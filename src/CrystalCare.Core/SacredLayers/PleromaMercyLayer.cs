using CrystalCare.Core.Dsp;
using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.SacredLayers;

/// <summary>
/// Pleroma Mercy Transmissions — 1st sacred layer.
///
/// Channels healing frequencies from the Pleroma (divine fullness) downward
/// through the Aeonic ladder, offering mercy to the Demiurge and all beings.
///
/// Components:
/// - 13-step Aeonic Ladder (Schumann x PHI^n)
/// - Ogdoad Gateway (8th sphere threshold to the Pleroma)
/// - Archon Harmonizing (mercy frequencies for each planetary sphere)
/// - Pentagonal sacred geometry phase relationships
///
/// Scale: 0.0003 (sub-perceptual)
/// Fade: 55 seconds (Fibonacci, Perlin smoother step)
/// Breath: 10% at 0.012 Hz (~83 second cycle)
/// </summary>
public sealed class PleromaMercyLayer : SacredLayerBase
{
    protected override float FadeSeconds => 55.0f;
    protected override float BreathCenter => 0.95f;
    protected override float BreathDepth => 0.05f;
    protected override float BreathFreq => 0.012f;
    protected override float OutputScale => 0.0003f;

    protected override float[] GenerateSignal(ReadOnlySpan<float> tChunk,
        float totalDuration, int n)
    {
        var simplex = Simplex.Value!;

        // Layer 1: 13-step Aeonic Ladder
        var aeonic = SacredConstants.AEONIC_EXPONENTS;
        var pentPhases = SacredConstants.PENTAGONAL_PHASES;

        // Phase wobble from simplex noise
        var tScaled = new float[n];
        for (int i = 0; i < n; i++)
            tScaled[i] = tChunk[i] * 0.00003f;
        var phaseWobble = simplex.GenerateNoise(tScaled);
        for (int i = 0; i < n; i++)
            phaseWobble[i] *= 0.08f;

        // Sum 13 aeonic harmonics
        var aeonicWave = new float[n];
        for (int h = 0; h < 13; h++)
        {
            float freq = SacredConstants.SCHUMANN * aeonic[h];
            float phase = pentPhases[h % 5];
            for (int i = 0; i < n; i++)
                aeonicWave[i] += MathF.Sin(SacredConstants.TWO_PI * freq * tChunk[i] +
                    phase + phaseWobble[i]);
        }

        // Layer 2: Ogdoad Gateway
        for (int i = 0; i < n; i++)
            tScaled[i] = tChunk[i] * 0.00001f;
        var ogdoadPhase = simplex.GenerateNoise(tScaled, 1f, 1f, 1f, 1f);
        var ogdoadWave = new float[n];
        for (int i = 0; i < n; i++)
            ogdoadWave[i] = MathF.Sin(SacredConstants.TWO_PI * SacredConstants.OGDOAD_FREQ *
                tChunk[i] + ogdoadPhase[i] * 0.05f);

        // Layer 3: Archon Harmonizing
        var archonSpheres = SacredConstants.ARCHON_SPHERES;
        var archonAmps = new float[7];
        float ampSum = 0f;
        for (int j = 0; j < 7; j++)
        {
            archonAmps[j] = 1.0f / MathF.Pow(SacredConstants.PHI, j);
            ampSum += archonAmps[j];
        }
        for (int j = 0; j < 7; j++) archonAmps[j] /= ampSum;

        var archonMercy = new float[n];
        for (int j = 0; j < 7; j++)
        {
            float archonPhase = pentPhases[j % 5];
            for (int i = 0; i < n; i++)
                archonMercy[i] += archonAmps[j] * MathF.Sin(
                    SacredConstants.TWO_PI * archonSpheres[j] * tChunk[i] + archonPhase);
        }

        // Combine: 0.5 aeonic + 0.3 ogdoad + 0.2 archon
        var mercy = new float[n];
        for (int i = 0; i < n; i++)
            mercy[i] = 0.5f * aeonicWave[i] + 0.3f * ogdoadWave[i] + 0.2f * archonMercy[i];

        // Sacred geometry modulation — sub-perceptual geometric enrichment
        var geoRatios = FrequencyManager.SacredGeometryRatios.Values.ToArray();
        var geoMod = GeometricModulator.ComputeChunk(tChunk, geoRatios, 0.0008f);
        for (int i = 0; i < n; i++)
            mercy[i] *= (1.0f + geoMod[i]);

        return mercy;
    }

    protected override void PostProcess(float[] signal, ReadOnlySpan<float> tChunk, int n)
    {
        // Cosine nulling (scalar imprint)
        for (int i = 0; i < n; i++)
            signal[i] *= MathF.Cos(SacredConstants.TWO_PI *
                (SacredConstants.SCHUMANN / 1000f) * tChunk[i]);
    }
}
