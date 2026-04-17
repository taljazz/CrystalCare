using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Tanh-based wave shaping for soft saturation.
/// Produces gentle harmonic distortion without hard clipping.
/// Port of jit_wave_shaping() from SoundGenerator.py.
/// </summary>
public static class WaveShaper
{
    // Tanh soft saturation — clamp to [-1, 1], apply tanh(factor * x) * 1.2.
    // Produces gentle harmonic distortion without hard clipping.
    #region Wave Shaping

    /// <summary>
    /// Apply tanh wave shaping: y = (528/432) * tanh(shapeFactor * clamp(x, -1, 1))
    /// Output scale is the Love Frequency ratio — the same 528/432 ratio used
    /// in Lemurian Quartz's heart chakra bridge.
    /// </summary>
    public static void Process(Span<float> signal, float shapeFactor = 2.5f)
    {
        for (int i = 0; i < signal.Length; i++)
        {
            float clamped = System.Math.Clamp(signal[i], -1.0f, 1.0f);
            signal[i] = MathF.Tanh(shapeFactor * clamped) * SacredConstants.WAVE_SHAPER_SCALE;
        }
    }

    #endregion

    // Peak normalization with headroom factor — scales signal so peak * factor = 1.0.
    #region Normalization

    /// <summary>
    /// Normalize signal by peak amplitude with a scale factor for headroom.
    /// Port of jit_normalize_signal().
    /// </summary>
    public static void Normalize(Span<float> signal, float scaleFactor = 1.3f)
    {
        float maxVal = 0f;
        for (int i = 0; i < signal.Length; i++)
            maxVal = MathF.Max(maxVal, MathF.Abs(signal[i]));

        if (maxVal > 0f)
        {
            float gain = 1.0f / (maxVal * scaleFactor);
            for (int i = 0; i < signal.Length; i++)
                signal[i] *= gain;
        }
    }

    #endregion

    // Tanh-based pan curve processing — soft-clips the pan value to [-0.8, 0.8]
    // for gentle stereo bounds that never hit hard left/right.
    #region Pan Curve

    /// <summary>
    /// Pan curve processing: clamp(tanh(x * scale), clipMin, clipMax).
    /// Port of jit_pan_curve_tanh().
    /// </summary>
    public static void PanCurveTanh(Span<float> panCurve, float scale = 0.6f,
        float clipMin = -0.8f, float clipMax = 0.8f)
    {
        for (int i = 0; i < panCurve.Length; i++)
            panCurve[i] = System.Math.Clamp(MathF.Tanh(panCurve[i] * scale), clipMin, clipMax);
    }

    #endregion
}
