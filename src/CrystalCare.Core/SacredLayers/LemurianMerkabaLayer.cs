using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Noise;

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
/// Fully stateless: no state carried between chunks.
///
/// Scale: 0.0006
/// Fade: 48 seconds
///
/// Port of AudioProcessor.lemurian_merkaba_layer_chunk() from SoundGenerator.py.
/// </summary>
public sealed class LemurianMerkabaLayer : ISacredLayer
{
    private readonly ThreadLocal<Simplex5D> _simplex = new(() => new Simplex5D(Random.Shared.Next(100)));

    public float[] ComputeChunk(ReadOnlySpan<float> tChunk, float totalDuration)
    {
        if (totalDuration < 60f)
            return new float[tChunk.Length];

        var simplex = _simplex.Value!;
        int n = tChunk.Length;

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

        // Heart coherence breath at 0.1 Hz (HeartMath)
        for (int i = 0; i < n; i++)
        {
            float breath = 0.975f + 0.025f * MathF.Sin(SacredConstants.TWO_PI * 0.1f * tChunk[i]);
            merkaba[i] *= breath;
        }

        // Sacred fade envelope + scale
        var fade = SacredFadeEnvelope.Compute(tChunk, totalDuration, fadeSeconds: 55.0f);
        for (int i = 0; i < n; i++)
            merkaba[i] *= fade[i] * 0.0006f;

        return merkaba;
    }
}
