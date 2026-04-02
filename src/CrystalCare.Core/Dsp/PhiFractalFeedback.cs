using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// PHI-spaced fractal micro-echoes for self-similar harmonic enrichment.
/// Adds golden-ratio-delayed copies at decreasing amplitudes,
/// creating organic overtone structure without altering the fundamental.
///
/// Port of AudioProcessor.phi_fractal_feedback() from SoundGenerator.py.
/// </summary>
public static class PhiFractalFeedback
{
    /// <summary>
    /// Apply PHI-fractal feedback to a signal.
    /// Default: 3 echo depths at 5% amplitude per echo.
    /// Base delay: sampleRate / PHI = ~29,708 samples (~0.619s at 48kHz).
    /// </summary>
    public static float[] Process(ReadOnlySpan<float> signal, int sampleRate = 48000,
        int depth = 3, float factor = 0.05f)
    {
        var result = new float[signal.Length];
        signal.CopyTo(result);

        int baseDelay = (int)(sampleRate / SacredConstants.PHI); // ~29,708 samples

        for (int d = 0; d < depth; d++)
        {
            int delay = baseDelay * (d + 1);
            if (delay >= signal.Length) break;

            float amplitude = MathF.Pow(factor, d + 1);

            // Add delayed copy
            for (int i = delay; i < signal.Length; i++)
                result[i] += signal[i - delay] * amplitude;
        }

        return result;
    }
}
