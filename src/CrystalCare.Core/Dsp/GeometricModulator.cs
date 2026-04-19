using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Geometric modulation: sum of sine waves at sacred ratio frequencies.
/// Port of AudioProcessor.geometric_modulation() from SoundGenerator.py.
/// </summary>
public static class GeometricModulator
{
    /// <summary>
    /// Compute frequency modulation from a set of geometric ratios.
    /// result[i] = sum over all ratios: modulationIndex * sin(2*PI * ratio * t[i])
    /// </summary>
    public static float[] Compute(ReadOnlySpan<double> t, Dictionary<string, float> ratios,
        float modulationIndex = 0.2f, CancellationToken ct = default)
    {
        if (ct.IsCancellationRequested)
            return new float[t.Length];

        var ratioValues = ratios.Values.ToArray();
        var result = new float[t.Length];

        for (int r = 0; r < ratioValues.Length; r++)
        {
            double ratio = ratioValues[r];
            for (int i = 0; i < t.Length; i++)
                result[i] += modulationIndex * (float)System.Math.Sin(SacredConstants.TWO_PI_D * ratio * t[i]);
        }

        return result;
    }

    /// <summary>
    /// Compute modulation for a chunk using pre-computed schedule data.
    /// Used by the streaming pipeline.
    /// </summary>
    public static float[] ComputeChunk(ReadOnlySpan<double> t, float[] ratioValues,
        float modulationIndex)
    {
        var result = new float[t.Length];
        for (int r = 0; r < ratioValues.Length; r++)
        {
            double ratio = ratioValues[r];
            for (int i = 0; i < t.Length; i++)
                result[i] += modulationIndex * (float)System.Math.Sin(SacredConstants.TWO_PI_D * ratio * t[i]);
        }
        return result;
    }
}
