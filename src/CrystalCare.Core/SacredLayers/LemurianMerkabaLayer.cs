using CrystalCare.Core.Dsp;
using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.SacredLayers;

/// <summary>
/// Lemurian Merkaba Layer — 5th sacred layer.
///
/// Sonic Merkaba (Jonathan Goldman): 4 frequencies forming a sacred geometric
/// sound structure. Pythagorean 3:4:5 ratios + golden ratio, anchored to 432 Hz.
///
/// Frequencies: 324 Hz (earth), 432 Hz (heart), 540 Hz (expression), 698.4 Hz (transcendence)
/// PHI-weighted amplitudes: keynote strongest, transcendence most subtle.
/// Heart coherence breath at 0.1 Hz (10-second cycle — HeartMath rhythm).
///
/// Scale: 0.0006
/// Fade: 55 seconds (Fibonacci)
/// </summary>
public sealed class LemurianMerkabaLayer : SacredLayerBase
{
    // Configuration: 55s Fibonacci fade, 5% breath depth at 0.1 Hz (HeartMath
    // heart coherence), 0.0006 output scale. BreathBeforeFade = true (Group B).
    #region Layer Configuration

    protected override float FadeSeconds => 55.0f;
    protected override float BreathCenter => 0.975f;
    protected override float BreathDepth => 0.025f;
    protected override float BreathFreq => SacredConstants.BREATH_HEART_COHERENCE; // 0.1 Hz HeartMath
    protected override float OutputScale => 0.0006f;
    protected override bool BreathBeforeFade => true;

    #endregion

    // Generates the Sonic Merkaba: 4 frequencies (324/432/540/698.4 Hz) from the
    // Pythagorean 3:4:5 triangle + PHI, with PHI-weighted amplitudes and organic
    // phase wobble from simplex noise. Flower of Life geometric modulation enriches
    // the Merkaba structure at 0.0006 sub-perceptual scale.
    #region Signal Generation

    protected override float[] GenerateSignal(ReadOnlySpan<double> tChunk,
        float totalDuration, int n)
    {
        var simplex = Simplex.Value!;

        // Organic phase wobble from simplex noise — scaled time stays small for float
        var tScaled = new float[n];
        for (int i = 0; i < n; i++)
            tScaled[i] = (float)(tChunk[i] * 0.03);
        var wobble = simplex.GenerateNoise(tScaled, 0.7f, 0.3f);
        for (int i = 0; i < n; i++)
            wobble[i] *= 0.015f;

        // Generate 4 Merkaba tones — double precision phase for long-session stability
        var freqs = SacredConstants.MERKABA_FREQS;
        var phases = SacredConstants.MERKABA_PHASES;
        var weights = SacredConstants.MERKABA_WEIGHTS;

        var merkaba = new float[n];
        for (int f = 0; f < 4; f++)
        {
            double freq = freqs[f];
            float phase = phases[f];
            for (int i = 0; i < n; i++)
                merkaba[i] += weights[f] * (float)System.Math.Sin(
                    SacredConstants.TWO_PI_D * freq * tChunk[i] + phase + wobble[i]);
        }

        // Flower of Life geometric modulation — sacred geometry enriching the Merkaba
        var flowerRatios = FrequencyManager.FlowerOfLifeRatios.Values.ToArray();
        var geoMod = GeometricModulator.ComputeChunk(tChunk, flowerRatios, 0.0006f);
        for (int i = 0; i < n; i++)
            merkaba[i] *= (1.0f + geoMod[i]);

        return merkaba;
    }

    #endregion
}
