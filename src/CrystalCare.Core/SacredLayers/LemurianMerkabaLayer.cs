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
    protected override float FadeSeconds => 55.0f;
    protected override float BreathCenter => 0.975f;
    protected override float BreathDepth => 0.025f;
    protected override float BreathFreq => 0.1f;
    protected override float OutputScale => 0.0006f;
    protected override bool BreathBeforeFade => true;

    protected override float[] GenerateSignal(ReadOnlySpan<float> tChunk,
        float totalDuration, int n)
    {
        var simplex = Simplex.Value!;

        // Organic phase wobble from simplex noise
        var tScaled = new float[n];
        for (int i = 0; i < n; i++)
            tScaled[i] = tChunk[i] * 0.03f;
        var wobble = simplex.GenerateNoise(tScaled, 0.7f, 0.3f);
        for (int i = 0; i < n; i++)
            wobble[i] *= 0.015f;

        // Generate 4 Merkaba tones
        var freqs = SacredConstants.MERKABA_FREQS;
        var phases = SacredConstants.MERKABA_PHASES;
        var weights = SacredConstants.MERKABA_WEIGHTS;

        var merkaba = new float[n];
        for (int f = 0; f < 4; f++)
        {
            for (int i = 0; i < n; i++)
                merkaba[i] += weights[f] * MathF.Sin(
                    SacredConstants.TWO_PI * freqs[f] * tChunk[i] + phases[f] + wobble[i]);
        }

        // Flower of Life geometric modulation — sacred geometry enriching the Merkaba
        var flowerRatios = FrequencyManager.FlowerOfLifeRatios.Values.ToArray();
        var geoMod = GeometricModulator.ComputeChunk(tChunk, flowerRatios, 0.0006f);
        for (int i = 0; i < n; i++)
            merkaba[i] *= (1.0f + geoMod[i]);

        return merkaba;
    }
}
