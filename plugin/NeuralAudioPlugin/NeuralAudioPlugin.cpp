#include "NeuralAudioPlugin.h"
#include "IPlug_include_in_plug_src.h"
#include "IControls.h"

NeuralAudioPlugin::NeuralAudioPlugin(const InstanceInfo& info)
: iplug::Plugin(info, MakeConfig(kNumParams, kNumPresets))
, mFilm()
, mModel()
{
  GetParam(kDrive)->InitDouble("Drive", 0., 0., 100.0, 0.01, "%");
  GetParam(kTone)->InitDouble("Tone", 0., 0., 100.0, 0.01, "%");



#if IPLUG_EDITOR
  mMakeGraphicsFunc = [&]() { return MakeGraphics(*this, PLUG_WIDTH, PLUG_HEIGHT, PLUG_FPS, GetScaleForScreen(PLUG_WIDTH, PLUG_HEIGHT)); };
  mLayoutFunc = [&](IGraphics* pGraphics) {
    // Attach resizer
    pGraphics->AttachCornerResizer(EUIResizerMode::Scale, false);

    const IBitmap bitmap = pGraphics->LoadBitmap(BACKGROUND);
    pGraphics->AttachControl(new IBitmapControl(pGraphics->GetBounds(), bitmap, 0));

    // Load font
    pGraphics->LoadFont("Roboto-Regular", ROBOTO_FN);
    // Get canvas bounds
    const IRECT b = pGraphics->GetBounds();
    // Centered knobs side-by-side
    IRECT knobRect = b.GetCentredInside(100);
    pGraphics->AttachControl(new IVKnobControl(knobRect.GetHShifted(-70), kDrive));
    pGraphics->AttachControl(new IVKnobControl(knobRect.GetHShifted(70), kTone));
  };
#endif

  mModelsOK = mFilm.initFromWeights();
  if (!mModelsOK)
  {
    mModelError = "FiLM load failed: " + mFilm.getLastError();
  }

  for (int ch = 0; ch < 2 && mModelsOK; ++ch)
  {
    if (!mModel[ch].initFromWeights())
    {
      mModelsOK = false;
      mModelError = "Model[" + std::to_string(ch) + "] load failed: " + mModel[ch].getLastError();
    }
  }

  if (!mModelsOK)
  {
    DBGMSG("NeuralAudioPlugin initialization error: %s", mModelError.c_str());
  }
  else
  {
    DBGMSG("NeuralAudioPlugin initialized successfull");
  }
}

void NeuralAudioPlugin::OnReset()
{
  const double sr = GetSampleRate();

  if (!mModelsOK)
    return;

  if (sr != mLastSampleRate)
  {
    for (int ch = 0; ch < 2; ++ch)
    {
      mModel[ch].discretize_bilinear((float)sr);
      DBGMSG("Model[%d] discretized at %f Hz", ch, sr);
    }
    mLastSampleRate = sr;
  }

  for (int ch = 0; ch < 2; ++ch)
  {
    mModel[ch].reset();
  }

}

#if IPLUG_DSP
void NeuralAudioPlugin::ProcessBlock(sample** inputs, sample** outputs, int nFrames)
{
  const int nChans = NOutChansConnected();

  if (!mModelsOK || nChans > 2)
  {
    for (int s = 0; s < nFrames; s++)
    {
      for (int c = 0; c < nChans; c++)
      {
        outputs[c][s] = inputs[c][s];
      }
    }
    return;
  }

  const float c1 = GetParam(kDrive)->Value() / 100. * 2. - 1.;
  const float c2 = GetParam(kTone)->Value() / 100. * 2. - 1.;
  mFilm.processSample(c1, c2);

  for (int s = 0; s < nFrames; s++)
  {
    for (int c = 0; c < nChans; c++)
    {
      float input = inputs[c][s];
      outputs[c][s] = mModel[c].processSample(input, mFilm.gamma, mFilm.beta);
    }
  }
}
#endif
