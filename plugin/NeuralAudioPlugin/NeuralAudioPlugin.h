#pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "Model.h"
#include "FiLM.h"
#include <array>

const int kNumPresets = 1;

enum EParams
{
  kDrive = 0,
  kTone,
  kNumParams
};

using namespace iplug;
using namespace igraphics;

class NeuralAudioPlugin final : public Plugin
{
public:
  NeuralAudioPlugin(const InstanceInfo& info);

#if IPLUG_DSP // http://bit.ly/2S64BDd
  void ProcessBlock(sample** inputs, sample** outputs, int nFrames) override;
#endif
  void OnReset() override;
private:
  FiLM<float, 16> mFilm;                  // 16 bytes alignment for SIMD operations
  std::array<Model<float, 16>, 2> mModel; // two models, one per channel
  bool mModelsOK = false;
  std::string mModelError;

  double mLastSampleRate = 0.0;
};