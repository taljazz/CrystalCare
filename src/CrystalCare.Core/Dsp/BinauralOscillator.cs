using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Binaural beat oscillator for stereo frequency pairs.
/// Each pair produces a different frequency in left and right channels,
/// creating a perceptual beat at the difference frequency.
///
/// Port of AudioProcessor.binaural_oscillator() and batch_binaural_oscillator().
/// </summary>
public static class BinauralOscillator
{
    /// <summary>
    /// Generate a single binaural pair. Returns [samples, 2] array.
    /// </summary>
    public static float[,] Generate(ReadOnlySpan<double> t, float leftFreq, float rightFreq)
    {
        var result = new float[t.Length, 2];
        for (int i = 0; i < t.Length; i++)
        {
            // Double precision phase, cast sin result to float
            result[i, 0] = (float)System.Math.Sin(SacredConstants.TWO_PI_D * leftFreq * t[i]);
            result[i, 1] = (float)System.Math.Sin(SacredConstants.TWO_PI_D * rightFreq * t[i]);
        }
        return result;
    }

    /// <summary>
    /// Generate multiple binaural pairs. Returns array of [samples, 2] results.
    /// </summary>
    public static float[][,] BatchGenerate(ReadOnlySpan<double> t,
        (float left, float right)[] freqPairs, CancellationToken ct = default)
    {
        var results = new float[freqPairs.Length][,];
        for (int p = 0; p < freqPairs.Length; p++)
        {
            if (ct.IsCancellationRequested)
            {
                results[p] = new float[t.Length, 2];
                continue;
            }
            results[p] = Generate(t, freqPairs[p].left, freqPairs[p].right);
        }
        return results;
    }
}
