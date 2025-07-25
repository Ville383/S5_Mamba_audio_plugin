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
class FiLM
{
private:
  // FiLM parameters
  static constexpr int c_in = 2;
  static constexpr int d_hidden = 4;
  static constexpr int d_model = 16;
  static constexpr int d_model_2 = 2 * d_model;

  // SIMD types and alignment
  using v_type = xsimd::simd_type<T>;
  static constexpr std::size_t alignment = Alignment;
  static constexpr int v_size = static_cast<int>(v_type::size);
  static constexpr int v_c_in = ceil_div(c_in, v_size);
  static constexpr int v_d_hidden = ceil_div(d_hidden, v_size);
  static constexpr int v_d_model = ceil_div(d_model, v_size);
  static constexpr int v_d_model_2 = ceil_div(d_model_2, v_size);

  // FiLM layers: Linear -> ReLU -> Linear
  alignas(alignment) v_type in_proj[c_in][v_d_hidden];
  alignas(alignment) v_type out_proj[d_hidden][v_d_model_2];
  alignas(alignment) v_type in_bias[v_d_hidden];
  alignas(alignment) v_type out_bias[v_d_model_2];

  // buffers for intermediate results
  alignas(alignment) v_type tmp1[v_d_hidden];
  alignas(alignment) v_type tmp2[v_d_model_2];

  // preallocated inputs
  alignas(alignment) v_type v_in0;
  alignas(alignment) v_type v_in1;

  // linear projection buffers
  alignas(alignment) T scalar_in[v_size] = {};
  static constexpr int v_size_d_hidden = std::min(v_size, d_hidden);

  // plugin loading
  std::string lastError;

public:
  FiLM() noexcept
  {
    // Initialize all to zero
    auto zero = v_type(T(0));
    for (int i = 0; i < c_in; ++i)
      for (int j = 0; j < v_d_hidden; ++j)
        in_proj[i][j] = zero;
    for (int i = 0; i < d_hidden; ++i)
      for (int j = 0; j < v_d_model_2; ++j)
        out_proj[i][j] = zero;
    for (int i = 0; i < v_d_hidden; ++i)
      in_bias[i] = zero;
    for (int i = 0; i < v_d_model_2; ++i)
      out_bias[i] = zero;
    for (int i = 0; i < v_d_hidden; ++i)
      tmp1[i] = zero;
    for (int i = 0; i < v_d_model_2; ++i)
      tmp2[i] = zero;
    v_in0 = zero;
    v_in1 = zero;
    for (int i = 0; i < v_size; ++i)
      scalar_in[i] = T(0);
    for (int i = 0; i < v_d_model; ++i)
    {
      gamma[i] = zero;
      beta[i] = zero;
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

  const std::string& getLastError() const noexcept { return lastError; }

  // Process conditioning input and update gamma and beta
  inline void processSample(const float& input1, const float& input2) noexcept
  {
    // reuse preallocated SIMD registers
    v_in0 = xsimd::batch<T, xsimd::default_arch>(input1);
    v_in1 = xsimd::batch<T, xsimd::default_arch>(input2);

    // layer 1: Linear + ReLU
    for (int i = 0; i < v_d_hidden; ++i)
    {
      tmp1[i] = in_proj[0][i] * v_in0 + in_proj[1][i] * v_in1 + in_bias[i];
      tmp1[i] = xsimd::max(tmp1[i], v_type(T(0)));
    }

    // layer 2: Linear
    for (int i = 0; i < v_d_model_2; ++i)
      tmp2[i] = out_bias[i];

    for (int i = 0; i < v_d_hidden; ++i)
    {
      tmp1[i].store_aligned(scalar_in);
      for (int j = 0; j < v_d_model_2; ++j)
        for (int k = 0; k < v_size_d_hidden; ++k)
          tmp2[j] += scalar_in[k] * out_proj[i * v_size + k][j];
    }

    // split into gamma and beta
    for (int i = 0; i < v_d_model; ++i)
    {
      gamma[i] = tmp2[i];
      beta[i] = tmp2[i + v_d_model];
    }
  }

  // output buffers
  alignas(alignment) v_type gamma[v_d_model];
  alignas(alignment) v_type beta[v_d_model];

private:
  void loadWeightsFromFile()
  {
    std::string json_string(reinterpret_cast<const char*>(model_weights_json), model_weights_json_len);
    nlohmann::json model_data = nlohmann::json::parse(json_string);

    auto W1 = model_data["layers"][0]["weights"][0];
    auto B1 = model_data["layers"][0]["weights"][1];
    auto W2 = model_data["layers"][1]["weights"][0];
    auto B2 = model_data["layers"][1]["weights"][1];

    // layer weights
    std::vector<T> flat1(d_hidden);
    for (int i = 0; i < c_in; ++i)
    {
      for (int j = 0; j < d_hidden; ++j)
        flat1[j] = static_cast<T>(W1[j][i]);
      set_values<T, alignment>(flat1, in_proj[i], d_hidden, v_d_hidden);
    }

    std::vector<T> flat2(d_model_2);
    for (int i = 0; i < d_hidden; ++i)
    {
      for (int j = 0; j < d_model_2; ++j)
        flat2[j] = static_cast<T>(W2[j][i]);
      set_values<T, alignment>(flat2, out_proj[i], d_model_2, v_d_model_2);
    }

    // biases
    std::vector<T> fb1(d_hidden);
    for (int i = 0; i < d_hidden; ++i)
      fb1[i] = static_cast<T>(B1[i]);
    set_values<T, alignment>(fb1, in_bias, d_hidden, v_d_hidden);

    std::vector<T> fb2(d_model_2);
    for (int i = 0; i < d_model_2; ++i)
      fb2[i] = static_cast<T>(B2[i]);
    set_values<T, alignment>(fb2, out_bias, d_model_2, v_d_model_2);
  }
};