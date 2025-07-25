// Derived from code by jatinchowdhury18 (2020)
// Licensed under the BSD 3-Clause License
// https://github.com/jatinchowdhury18/RTNeural
#pragma once

#include "common.h"
#include "json.hpp"
#include "model_weights.h"
#include "xsimd/xsimd.hpp"
#include <string>
#include <vector>

template <typename T, std::size_t Alignment = xsimd::default_arch::alignment()>
class Model
{
private:
  // Model parameters
  // no bias and use conjugate symmetry
  static constexpr int d_model = 16;
  static constexpr int d_state = 64;
  static constexpr int exp_f = 2;
  static constexpr int d_inner = exp_f * d_model;
  static constexpr int d_inner_2 = 2 * d_inner;
  static constexpr int ssm_size = d_state / 2;
  static constexpr int num_layers = 2;

  // SIMD types and alignment
  using v_type = xsimd::simd_type<T>;
  static constexpr std::size_t alignment = Alignment;
  static constexpr int v_size = static_cast<int>(v_type::size);
  static constexpr int v_d_model = ceil_div(d_model, v_size);
  static constexpr int v_d_inner = ceil_div(d_inner, v_size);
  static constexpr int v_d_inner_2 = ceil_div(d_inner_2, v_size);
  static constexpr int v_ssm_size = ceil_div(ssm_size, v_size);

  // buffers for intermediate results
  alignas(alignment) v_type tmp[v_d_model];
  alignas(alignment) v_type res1[v_d_model];
  alignas(alignment) v_type res2[v_d_inner];
  alignas(alignment) v_type mamba_proj[v_d_inner_2];
  alignas(alignment) v_type u[v_d_inner];
  alignas(alignment) v_type y[v_d_inner];
  alignas(alignment) v_type Bu_real[v_ssm_size];
  alignas(alignment) v_type Bu_imag[v_ssm_size];

  alignas(alignment) v_type BL_real[v_ssm_size];
  alignas(alignment) v_type BL_imag[v_ssm_size];

  alignas(alignment) v_type v_input;
  alignas(alignment) v_type v_tmp_RMS;
  T output;
  T sum_RMS;

  // linear projection buffers
  alignas(alignment) T scalar_in[v_size] = {T(0)};
  alignas(alignment) T scalar_in2[v_size] = {T(0)};
  static constexpr int v_size_d_model = std::min(v_size, d_model);
  static constexpr int v_size_d_inner = std::min(v_size, d_inner);
  static constexpr int v_size_ssm_size = std::min(v_size, ssm_size);

  // Model weights
  alignas(alignment) v_type in_proj[v_d_model];
  alignas(alignment) v_type out_proj[v_d_model];

  alignas(alignment) v_type in_proj_mamba[num_layers][d_model][v_d_inner_2]; // [layer][in][v_out]
  alignas(alignment) v_type out_proj_mamba[num_layers][d_inner][v_d_model];

  alignas(alignment) v_type A_real[num_layers][v_ssm_size];
  alignas(alignment) v_type A_imag[num_layers][v_ssm_size];
  alignas(alignment) v_type B_real[num_layers][d_inner][v_ssm_size];
  alignas(alignment) v_type B_imag[num_layers][d_inner][v_ssm_size];
  alignas(alignment) v_type C_real[num_layers][ssm_size][v_d_inner];
  alignas(alignment) v_type C_imag[num_layers][ssm_size][v_d_inner];
  alignas(alignment) v_type D[num_layers][v_d_inner];

  alignas(alignment) v_type inv_dt[num_layers][v_ssm_size];
  alignas(alignment) v_type dt[v_ssm_size];
  alignas(alignment) v_type dA_real[num_layers][v_ssm_size];
  alignas(alignment) v_type dA_imag[num_layers][v_ssm_size];
  alignas(alignment) v_type dB_real[num_layers][d_inner][v_ssm_size];
  alignas(alignment) v_type dB_imag[num_layers][d_inner][v_ssm_size];

  alignas(alignment) v_type norm[num_layers][v_d_model];
  T eps[num_layers];

  // Hidden state
  alignas(alignment) v_type hidden_real[num_layers][v_ssm_size];
  alignas(alignment) v_type hidden_imag[num_layers][v_ssm_size];

  // Plugin loading
  std::string lastError;

public:
  Model() noexcept
  {
    // Initialize all to zero
    const v_type zero = v_type(T(0));

    // buffers
    for (int i = 0; i < v_d_model; ++i)
      tmp[i] = res1[i] = zero;
    for (int i = 0; i < v_d_inner_2; ++i)
      mamba_proj[i] = zero;
    for (int i = 0; i < v_d_inner; ++i)
      u[i] = res2[i] = y[i] = zero;
    for (int i = 0; i < v_ssm_size; ++i)
      Bu_real[i] = Bu_imag[i] = zero;
    v_input = v_tmp_RMS = zero;
    for (int i = 0; i < v_size; ++i)
      scalar_in[i] = scalar_in2[i] = T(0);
    output = sum_RMS = T(0);

    // weights & parameters
    for (int i = 0; i < v_d_model; ++i)
      in_proj[i] = out_proj[i] = zero;
    for (int L = 0; L < num_layers; ++L)
    {
      eps[L] = T(0);
      for (int i = 0; i < d_model; ++i)
        for (int j = 0; j < v_d_inner_2; ++j)
          in_proj_mamba[L][i][j] = zero;

      for (int i = 0; i < d_inner; ++i)
        for (int j = 0; j < v_d_model; ++j)
          out_proj_mamba[L][i][j] = zero;

      for (int i = 0; i < v_ssm_size; ++i)
      {
        A_real[L][i] = A_imag[L][i] = inv_dt[L][i] = zero;
        hidden_real[L][i] = hidden_imag[L][i] = zero;
        dA_real[L][i] = dA_imag[L][i] = zero;
      }

      for (int i = 0; i < d_inner; ++i)
        for (int j = 0; j < v_ssm_size; ++j)
        {
          B_real[L][i][j] = B_imag[L][i][j] = zero;
          dB_real[L][i][j] = dB_imag[L][i][j] = zero;
        }

      for (int i = 0; i < ssm_size; ++i)
        for (int j = 0; j < v_d_inner; ++j)
          C_real[L][i][j] = C_imag[L][i][j] = zero;

      for (int i = 0; i < v_d_inner; ++i)
        D[L][i] = zero;
      for (int i = 0; i < v_d_model; ++i)
        norm[L][i] = zero;
    }
  }

  bool initFromWeights() noexcept
  {
    try
    {
      loadWeightsFromFile();
      return true;
    }
    catch (const std::exception& e)
    {
      lastError = e.what();
      return false;
    }
    catch (...)
    {
      lastError = "Unknown error loading weights";
      return false;
    }
  }

  inline void reset() noexcept
  {
    const v_type zero = v_type(T(0));
    for (int i = 0; i < num_layers; ++i)
      for (int j = 0; j < v_ssm_size; ++j)
        hidden_real[i][j] = hidden_imag[i][j] = zero;
  }

  const std::string& getLastError() const noexcept { return lastError; }

  // Process a single sample through the neural network
  inline T processSample(const T& input, const v_type (&gamma)[v_d_model], const v_type (&beta)[v_d_model]) noexcept
  {
    v_input = xsimd::batch<T, xsimd::default_arch>(input);
    output = T(0);

    // in proj
    for (int i = 0; i < v_d_model; ++i)
    {
      tmp[i] = in_proj[i] * v_input;
    }

    for (int i = 0; i < num_layers; ++i)
    {
      // Residual connection
      for (int j = 0; j < v_d_model; ++j)
      {
        res1[j] = tmp[j];
      }

      // FiLM conditioning
      for (int j = 0; j < v_d_model; ++j)
      {
        tmp[j] = gamma[j] * tmp[j] + beta[j];
      }

      // RMS norm
      v_tmp_RMS = v_type(T(0));
      for (int j = 0; j < v_d_model; ++j)
      {
        v_tmp_RMS += tmp[j] * tmp[j];
      }
      sum_RMS = xsimd::reduce_add(v_tmp_RMS) / static_cast<T>(d_model); // expects d_model is a multiple of v_size
      v_tmp_RMS = v_type(T(1) / std::sqrt(eps[i] + sum_RMS));

      for (int j = 0; j < v_d_model; ++j)
      {
        tmp[j] = norm[i][j] * tmp[j] * v_tmp_RMS;
      }

      // Mamba in proj
      for (int j = 0; j < v_d_inner_2; ++j)
      {
        mamba_proj[j] = v_type(T(0));
      }

      for (int j = 0; j < v_d_model; ++j)
      {
        tmp[j].store_aligned(scalar_in);
        for (int k = 0; k < v_d_inner_2; ++k)
        {
          for (int l = 0; l < v_size_d_model; ++l)
          {
            mamba_proj[k] += scalar_in[l] * in_proj_mamba[i][j * v_size + l][k];
          }
        }
      }

      // silu
      for (int j = 0; j < v_d_inner_2; ++j)
      {
        mamba_proj[j] = mamba_proj[j] / (v_type(T(1)) + xsimd::exp(-mamba_proj[j]));
      }

      // chunk
      for (int j = 0; j < v_d_inner; ++j)
      {
        u[j] = mamba_proj[j];
        res2[j] = mamba_proj[j + v_d_inner];
      }

      /* ================ S5 ================ */
      // h[n] = Ah[n - 1] + Bu[n]
      // y[n] = real(Ch[n]) + Du[n]

      // Bu[n]
      for (int j = 0; j < v_ssm_size; ++j)
      {
        Bu_real[j] = v_type(T(0));
        Bu_imag[j] = v_type(T(0));
      }

      for (int j = 0; j < v_d_inner; ++j)
      {
        u[j].store_aligned(scalar_in);
        for (int k = 0; k < v_ssm_size; ++k)
        {
          for (int l = 0; l < v_size_d_inner; ++l)
          {
            Bu_real[k] += scalar_in[l] * dB_real[i][j * v_size + l][k];
            Bu_imag[k] += scalar_in[l] * dB_imag[i][j * v_size + l][k];
          }
        }
      }

      // h[n]
      for (int j = 0; j < v_ssm_size; ++j)
      {
        auto tmp1 = hidden_real[i][j];
        auto tmp2 = hidden_imag[i][j];
        hidden_real[i][j] = tmp1 * dA_real[i][j] - tmp2 * dA_imag[i][j] + Bu_real[j];
        hidden_imag[i][j] = tmp1 * dA_imag[i][j] + tmp2 * dA_real[i][j] + Bu_imag[j];
      }

      // y[n]
      for (int j = 0; j < v_d_inner; ++j)
      {
        y[j] = D[i][j] * u[j];
      }

      for (int j = 0; j < v_ssm_size; ++j)
      {
        hidden_real[i][j].store_aligned(scalar_in);
        hidden_imag[i][j].store_aligned(scalar_in2);
        for (int k = 0; k < v_d_inner; ++k)
        {
          for (int l = 0; l < v_size_ssm_size; ++l)
          {
            y[k] += v_type(T(2)) * (scalar_in[l] * C_real[i][j * v_size + l][k] - scalar_in2[l] * C_imag[i][j * v_size + l][k]); // use conj_sym
          }
        }
      }
      /* ==================================== */

      // Residual connection
      for (int j = 0; j < v_d_inner; ++j)
      {
        y[j] *= res2[j];
      }

      // mamba out proj
      for (int j = 0; j < v_d_model; ++j)
      {
        tmp[j] = v_type(T(0));
      }

      for (int j = 0; j < v_d_inner; ++j)
      {
        y[j].store_aligned(scalar_in);
        for (int k = 0; k < v_d_model; ++k)
        {
          for (int l = 0; l < v_size_d_inner; ++l)
          {
            tmp[k] += scalar_in[l] * out_proj_mamba[i][j * v_size + l][k];
          }
        }
      }

      // Residual connection
      for (int j = 0; j < v_d_model; ++j)
      {
        tmp[j] += res1[j];
      }
    }

    // out proj
    for (int i = 0; i < v_d_model; ++i)
    {
      output += xsimd::reduce_add(tmp[i] * out_proj[i]);
    }

    return output;
  }

  void discretize_bilinear(const T& sr) noexcept
  {
    // Discretize the continuous-time A and B variables for all layers
    for (int i = 0; i < num_layers; ++i)
    {
      for (int j = 0; j < v_ssm_size; ++j)
      {
        dt[j] = v_type(T(48000)) / v_type(T(sr)) * xsimd::log(v_type(T(1)) + xsimd::exp(inv_dt[i][j]));
      }

      // dA
      for (int j = 0; j < v_ssm_size; ++j)
      {
        auto dt_div_2 = dt[j] / v_type(T(2));
        auto denom_c = v_type(T(1)) - dt_div_2 * A_real[i][j];
        auto denom_d = -dt_div_2 * A_imag[i][j];
        auto denom = denom_c * denom_c + denom_d * denom_d;

        BL_real[j] = denom_c / denom;
        BL_imag[j] = -denom_d / denom;

        auto tmp1 = v_type(T(1)) + dt_div_2 * A_real[i][j];
        auto tmp2 = dt_div_2 * A_imag[i][j];
        dA_real[i][j] = BL_real[j] * tmp1 - BL_imag[j] * tmp2;
        dA_imag[i][j] = BL_real[j] * tmp2 + BL_imag[j] * tmp1;
      }

      // dB
      for (int j = 0; j < v_ssm_size; ++j)
      {
        for (int k = 0; k < d_inner; ++k)
        {
          dB_real[i][k][j] = BL_real[j] * dt[j] * B_real[i][k][j] - BL_imag[j] * dt[j] * B_imag[i][k][j];
          dB_imag[i][k][j] = BL_real[j] * dt[j] * B_imag[i][k][j] + BL_imag[j] * dt[j] * B_real[i][k][j];
        }
      }
    }
  }

private:
  // Load weights from the embedded model_weights.h file
  void loadWeightsFromFile()
  {
    // Read the embedded JSON data from model_weights.h
    // model_weights_json is declared as: unsigned char model_weights_json[]
    // model_weights_json_len is declared as: unsigned int model_weights_json_len

    // Create string from the embedded binary data
    std::string json_string(reinterpret_cast<const char*>(model_weights_json), model_weights_json_len);
    nlohmann::json model_data = nlohmann::json::parse(json_string);

    // FilM weights at layers 0 and 1
    auto in_proj_weights = model_data["layers"][2]["weights"][0];
    auto out_proj_weights = model_data["layers"][num_layers + 3]["weights"][0];

    std::vector<T> in_proj_flat_weights(d_model);
    std::vector<T> out_proj_flat_weights(d_model);
    for (int i = 0; i < d_model; ++i)
    {
      in_proj_flat_weights[i] = static_cast<T>(in_proj_weights[i][0]);
      out_proj_flat_weights[i] = static_cast<T>(out_proj_weights[0][i]);
    }
    set_values<T, alignment>(in_proj_flat_weights, in_proj, d_model, v_d_model);
    set_values<T, alignment>(out_proj_flat_weights, out_proj, d_model, v_d_model);

    for (int i = 0; i < num_layers; ++i)
    {
      auto mamba_in_proj_weights = model_data["layers"][i + 3]["parameters"]["mamba"]["in_proj"]["weights"];
      auto A_real_weights = model_data["layers"][i + 3]["parameters"]["mamba"]["A_real"];
      auto A_imag_weights = model_data["layers"][i + 3]["parameters"]["mamba"]["A_imag"];
      auto B_real_weights = model_data["layers"][i + 3]["parameters"]["mamba"]["B_real"];
      auto B_imag_weights = model_data["layers"][i + 3]["parameters"]["mamba"]["B_imag"];
      auto C_real_weights = model_data["layers"][i + 3]["parameters"]["mamba"]["C_real"];
      auto C_imag_weights = model_data["layers"][i + 3]["parameters"]["mamba"]["C_imag"];
      auto D_weights = model_data["layers"][i + 3]["parameters"]["mamba"]["D"];
      auto inv_dt_weights = model_data["layers"][i + 3]["parameters"]["mamba"]["inv_dt"];
      auto norm_weights = model_data["layers"][i + 3]["parameters"]["norm"]["weight"];
      auto eps_weight = model_data["layers"][i + 3]["parameters"]["norm"]["eps"];
      auto mamba_out_proj_weights = model_data["layers"][i + 3]["parameters"]["mamba"]["out_proj"]["weights"];

      std::vector<T> mamba_in_proj_flat_weights(d_inner_2);
      for (int j = 0; j < d_model; ++j)
      {
        for (int k = 0; k < d_inner_2; ++k)
        {
          mamba_in_proj_flat_weights[k] = static_cast<T>(mamba_in_proj_weights[k][j]);
        }
        set_values<T, alignment>(mamba_in_proj_flat_weights, in_proj_mamba[i][j], d_inner_2, v_d_inner_2);
      }

      std::vector<T> mamba_out_proj_flat_weights(d_model);
      for (int j = 0; j < d_inner; ++j)
      {
        for (int k = 0; k < d_model; ++k)
        {
          mamba_out_proj_flat_weights[k] = static_cast<T>(mamba_out_proj_weights[k][j]);
        }
        set_values<T, alignment>(mamba_out_proj_flat_weights, out_proj_mamba[i][j], d_model, v_d_model);
      }

      std::vector<T> norm_flat_weights(d_model);
      for (int j = 0; j < d_model; ++j)
      {
        norm_flat_weights[j] = static_cast<T>(norm_weights[j]);
      }
      set_values<T, alignment>(norm_flat_weights, norm[i], d_model, v_d_model);
      eps[i] = static_cast<T>(eps_weight);

      std::vector<T> A_real_flat_weights(ssm_size);
      std::vector<T> A_imag_flat_weights(ssm_size);
      for (int j = 0; j < ssm_size; ++j)
      {
        A_real_flat_weights[j] = static_cast<T>(A_real_weights[j]);
        A_imag_flat_weights[j] = static_cast<T>(A_imag_weights[j]);
      }
      set_values<T, alignment>(A_real_flat_weights, A_real[i], ssm_size, v_ssm_size);
      set_values<T, alignment>(A_imag_flat_weights, A_imag[i], ssm_size, v_ssm_size);

      std::vector<T> B_real_flat_weights(ssm_size);
      std::vector<T> B_imag_flat_weights(ssm_size);
      for (int j = 0; j < d_inner; ++j)
      {
        for (int k = 0; k < ssm_size; ++k)
        {
          B_real_flat_weights[k] = static_cast<T>(B_real_weights[k][j]);
          B_imag_flat_weights[k] = static_cast<T>(B_imag_weights[k][j]);
        }
        set_values<T, alignment>(B_real_flat_weights, B_real[i][j], ssm_size, v_ssm_size);
        set_values<T, alignment>(B_imag_flat_weights, B_imag[i][j], ssm_size, v_ssm_size);
      }

      std::vector<T> C_real_flat_weights(d_inner);
      std::vector<T> C_imag_flat_weights(d_inner);
      for (int j = 0; j < ssm_size; ++j)
      {
        for (int k = 0; k < d_inner; ++k)
        {
          C_real_flat_weights[k] = static_cast<T>(C_real_weights[k][j]);
          C_imag_flat_weights[k] = static_cast<T>(C_imag_weights[k][j]);
        }
        set_values<T, alignment>(C_real_flat_weights, C_real[i][j], d_inner, v_d_inner);
        set_values<T, alignment>(C_imag_flat_weights, C_imag[i][j], d_inner, v_d_inner);
      }

      std::vector<T> D_flat_weights(d_inner);
      for (int j = 0; j < d_inner; ++j)
      {
        D_flat_weights[j] = static_cast<T>(D_weights[j]);
      }
      set_values<T, alignment>(D_flat_weights, D[i], d_inner, v_d_inner);

      std::vector<T> inv_dt_flat_weights(ssm_size);
      for (int j = 0; j < ssm_size; ++j)
      {
        inv_dt_flat_weights[j] = static_cast<T>(inv_dt_weights[j]);
      }
      set_values<T, alignment>(inv_dt_flat_weights, inv_dt[i], ssm_size, v_ssm_size);
    }
  }
};