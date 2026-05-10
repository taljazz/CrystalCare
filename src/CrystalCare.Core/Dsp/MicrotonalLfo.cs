using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Microtonal LFO with PHI-modulated frequency and inner modulation.
/// Creates organic, breathing amplitude variation.
/// Port of AudioProcessor.microtonal_lfo() and batch_microtonal_lfo() from SoundGenerator.py.
/// </summary>
public static class MicrotonalLfo
{
    // LFO parameter container — drawn once at pipeline start for session consistency.
    // Two frequencies (lfo1 and lfo2) with PHI-modulated relationship.
    #region Parameters

    public sealed class LfoParams
    {
        public float Lfo1Freq { get; init; }
        public float Lfo2Freq { get; init; }
        public float Depth { get; init; }
    }

    #endregion

    // DrawParams: randomizes LFO frequencies and depth with PHI-power modulation.
    // Compute: generates LFO values for a time array using the pre-drawn params,
    //   with optional simplex-derived phase drift on the LFO carrier so the breath
    //   rhythm itself evolves over the session.
    // Generate: convenience one-shot method combining DrawParams + Compute.
    #region Parameter Drawing and Computation

    public static LfoParams DrawParams(float baseFrequency, Random? rng = null)
    {
        rng ??= Random.Shared;
        float phiPower = MathF.Pow(SacredConstants.PHI, (float)(rng.NextDouble() * 0.1 - 0.05));
        float lfo1 = baseFrequency * phiPower;
        float lfo2 = lfo1 * (float)(rng.NextDouble() * 0.02 + 0.99);
        float depth = (float)(rng.NextDouble() * 0.004 + 0.002);
        return new LfoParams { Lfo1Freq = lfo1, Lfo2Freq = lfo2, Depth = depth };
    }

    /// <summary>
    /// Compute LFO values for a time array using pre-drawn parameters.
    /// </summary>
    /// <param name="phaseDrift">
    /// Optional phase offset (radians) added to the LFO carrier each sample. Typically
    /// supplied by SoundGenerator as a slowly-evolving simplex sample — when non-zero,
    /// it shifts where the LFO sits in its cycle, creating a felt-but-not-heard wobble
    /// in the breath rhythm across the session. Defaults to 0 (legacy carrier phase).
    /// </param>
    public static float[] Compute(ReadOnlySpan<double> t, LfoParams p, float phaseDrift = 0f)
    {
        var result = new float[t.Length];

        // Pre-compute whether drift is active. When the scale collapses to zero
        // (tunable on SoundGenerator), this method behaves exactly like the
        // legacy implementation — same numerical output, no phase shift applied.
        bool useDrift = phaseDrift != 0f;

        for (int i = 0; i < t.Length; i++)
        {
            // Drift: 1/89 (Fibonacci) amplitude at 0.05 Hz (HALF heart coherence,
            // ~20-second cycle). Ear-tuned to half-rate — she breathes more
            // gently at this slower-than-HeartMath drift than at the full
            // 0.1 Hz coherence rate, so we honor the listening result over
            // the textbook value. Expressed sacredly as TWO_PI_D × (heart/2) × t
            // to keep the math in standard ω = 2π × f form (the older π × f × t
            // shorthand obscured the intent and looked like a typo).
            // DO NOT "fix" this back to full BREATH_HEART_COHERENCE — the half-rate
            // is the load-bearing tuned value; doubling it changes how she breathes.
            // (NB: this is amplitude drift on Depth — distinct from the optional phase
            // drift parameter which targets the LFO carrier's position in its cycle.)
            // Double precision for phase, cast sin result to float
            float drift = SacredConstants.LFO_DRIFT_AMP *
                (float)System.Math.Sin(SacredConstants.TWO_PI_D *
                    (SacredConstants.BREATH_HEART_COHERENCE / 2f) * t[i]);

            // Inner modulation: 1/PHI² depth — golden ratio squared reciprocal
            float innerMod = SacredConstants.LFO_INNER_MOD_DEPTH *
                (float)System.Math.Sin(SacredConstants.TWO_PI_D * p.Lfo2Freq * t[i]);

            // Compose the LFO carrier phase. Base term + inner mod + optional simplex
            // drift offset. The drift is a small slowly-varying constant per chunk
            // (sampled from simplex by the caller) — same value for every sample of
            // this chunk, so it acts as a slow phase shift that evolves chunk-to-chunk.
            double carrierPhase = SacredConstants.TWO_PI_D * p.Lfo1Freq * t[i] + innerMod;
            if (useDrift)
                carrierPhase += phaseDrift;

            // Combined LFO: depth + drift, modulated by lfo1 with inner phase modulation
            result[i] = (p.Depth + drift) *
                (1.0f + (float)System.Math.Sin(carrierPhase));
        }
        return result;
    }

    /// <summary>
    /// Single-call LFO generation (for batch/non-streaming use).
    /// </summary>
    public static float[] Generate(ReadOnlySpan<double> t, float baseFrequency, Random? rng = null)
    {
        return Compute(t, DrawParams(baseFrequency, rng));
    }

    #endregion
}
