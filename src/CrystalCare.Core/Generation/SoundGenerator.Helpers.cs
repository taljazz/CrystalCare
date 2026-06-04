using CrystalCare.Core.Dsp;
using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Generation;

/// <summary>
/// SoundGenerator helper methods — frequency set construction, modulation scheduling,
/// and normalization gain estimation. Split from the main file for readability.
/// </summary>
public sealed partial class SoundGenerator
{
    // Builds the 13-frequency set from a base frequency using three different
    // mathematical relationships: 6 PHI exponents, 3 ratio-1.3 exponents,
    // and 4 subharmonics. Each frequency gets slight random jitter (±2-5%)
    // so every session has a unique tonal character.
    #region Frequency Set Building

    /// <summary>
    /// Build the 13-frequency set from the base frequency.
    /// 6 PHI exponents + 3 ratio-1.3 exponents + 4 subharmonics = 13 total.
    /// Each gets random jitter for organic uniqueness.
    /// </summary>
    private float[] BuildFrequencySet(float baseFreq)
    {
        var freqs = new List<float>();

        // 6 PHI exponents: base * PHI^0 through PHI^5 (with ±2% jitter)
        for (int i = 0; i < 6; i++)
            freqs.Add(baseFreq * SacredConstants.PHI_EXPONENTS_6[i] *
                (float)(_rng.NextDouble() * 0.04 + 0.98));

        // 3 ratio-1.3 exponents: base * 1.3^0, 1.3^1, 1.3^2 (with ±2% jitter)
        for (int i = 0; i < 3; i++)
            freqs.Add(baseFreq * SacredConstants.RATIO_1_3_EXPONENTS_3[i] *
                (float)(_rng.NextDouble() * 0.04 + 0.98));

        // 4 subharmonics: base / 2, 4, 8, 16 (with ±5% jitter)
        for (int i = 0; i < 4; i++)
            freqs.Add(baseFreq / SacredConstants.SUBHARMONIC_DIVISORS[i] *
                (float)(_rng.NextDouble() * 0.1 + 0.95));

        return freqs.ToArray();
    }

    #endregion

    // Builds the modulation schedule that determines which sacred geometry ratio sets
    // are active during which time segments, and at what modulation depth.
    // Dimensional mode cycles through 7 phases then an "all ratios" finale.
    // Normal mode uses Fibonacci-timed intervals cycling through random ratio sets.
    #region Modulation Schedule

    /// <summary>
    /// Build the modulation schedule — time segments paired with ratio sets and depths.
    /// Dimensional mode: 7 phases + all-ratios finale.
    /// Normal mode: Fibonacci-timed intervals with random ratio set selection.
    /// </summary>
    private List<(int start, int end, float[] ratioValues, float modIndex, int headFade, int tailFade)>
        BuildModulationSchedule(float duration, int totalSamples, int sampleRate,
            int[] intervalDurations, bool dimensionalMode, CancellationToken ct)
    {
        var schedule = new List<(int, int, float[], float, int, int)>();
        int current = 0;

        if (dimensionalMode)
        {
            // Dimensional Journey: 9 equal-duration phases, one per dimension (1D-9D).
            // Each phase uses its dimension-specific ratio set and modulation intensity
            // from DimensionalJourney.DIMENSIONS. This replaces the prior implementation
            // which iterated a `sel` dimension index but never used it — every phase
            // just picked a RANDOM ratio set, and 3D was missing entirely. The dimension
            // labels were decorative; the audio didn't actually distinguish dimensions.
            //
            // Now: 1D → minimal (Metatron √2 only — foundational anchor)
            //      2D → fibonacci_set (PHI-based — Atlantean cosmic)
            //      3D → triple_helix (DNA 1.0/1.2/1.4 — the physical body)  ← FILLED IN (was missing)
            //      4D → flower_of_life (soft 1.3/1.5/2.5 — astral flow)
            //      5D → sacred_geometry (full Pythagorean)
            //      6D → combined (sacred + flower — bridging structure)
            //      7D → fractal_set (transcendental π and e — soul/causal)
            //      8D → taygetan (Pleiadian / Taygetan ratios)
            //      9D → all_blended (every distinct ratio sounding together — Source culmination)
            //
            // headFade/tailFade samples enable smootherstep crossfades between phases:
            // middle phases get both edges faded; the first phase opens at full
            // (no head fade) so the session starts cleanly at 1D; the last phase
            // closes at full (no tail fade) so the session ends in pure 9D.
            int dimCount = DimensionalJourney.DIMENSIONS.Length;
            int phaseLen = totalSamples / dimCount;
            int xfadeSamples = (int)(phaseLen * DimensionalJourney.CROSSFADE_FRACTION);
            for (int d = 0; d < dimCount; d++)
            {
                if (ct.IsCancellationRequested) break;

                // Look up the dimension's specific ratio set + modulation intensity
                var dim = DimensionalJourney.DIMENSIONS[d];
                var ratios = DimensionalJourney.ResolveRatios(dim);
                float modIndex = dim.ModIndex;

                // Last phase extends to totalSamples to absorb any integer-division
                // remainder so the schedule covers the full session with no silence
                // gap at the end.
                int end = (d == dimCount - 1)
                    ? totalSamples
                    : global::System.Math.Min(current + phaseLen, totalSamples);

                // First phase: no head fade (open at full 1D presence).
                // Last phase: no tail fade (end at full 9D presence).
                // Middle phases: both edges fade (smootherstep crossfade to neighbors).
                int headFade = (d == 0) ? 0 : xfadeSamples;
                int tailFade = (d == dimCount - 1) ? 0 : xfadeSamples;

                schedule.Add((current, end, ratios, modIndex, headFade, tailFade));
                current = end;
            }
        }
        else
        {
            // Normal mode: cycle through Fibonacci-timed intervals [34, 55, 89, 144].
            // headFade/tailFade = 0 so the legacy non-dimensional behavior is preserved
            // exactly (no smootherstep weighting — each segment plays at full modIndex
            // for its entire duration, instant switch at boundaries as before).
            float remaining = duration;
            int intervalCount = 0;
            while (remaining > 0 && !ct.IsCancellationRequested)
            {
                // Select interval duration from the Fibonacci list (cycles)
                float interval = global::System.Math.Min(
                    intervalDurations[intervalCount % intervalDurations.Length], remaining);
                int segSamples = (int)(sampleRate * interval);

                // Pick a random ratio set with weighted probability
                var ratioSet = _frequencyManager.SelectRandomRatioSet(ct);
                float modIndex = (float)(_rng.NextDouble() * 0.05 + 0.2);
                int end = global::System.Math.Min(current + segSamples, totalSamples);
                schedule.Add((current, end, ratioSet.Values.ToArray(), modIndex, 0, 0));
                current = end;
                remaining -= interval;
                intervalCount++;
            }
        }

        return schedule;
    }

    /// <summary>
    /// Compute modulation values for a single chunk from the pre-built schedule.
    /// Only processes schedule segments that overlap with this chunk's sample range.
    /// Each active ratio contributes a sine wave at modulationIndex amplitude.
    ///
    /// Each segment may have head/tail fade samples (used by Dimensional Journey
    /// mode for smootherstep crossfades between adjacent dimensions). When non-zero,
    /// the first headFade samples ramp 0 → 1 via Perlin smootherstep, and the last
    /// tailFade samples ramp 1 → 0 via smootherstep. For non-dimensional modes the
    /// fade samples are 0 so the segment plays at full modIndex throughout (exact
    /// legacy behavior preserved).
    /// </summary>
    private static float[] ComputeModulationChunk(ReadOnlySpan<double> tChunk,
        int chunkOffset, int chunkSamples,
        List<(int start, int end, float[] ratioValues, float modIndex, int headFade, int tailFade)> schedule)
    {
        var result = new float[chunkSamples];
        int chunkEnd = chunkOffset + chunkSamples;

        // Only process schedule segments that overlap with this chunk
        foreach (var (start, end, ratioValues, modIndex, headFade, tailFade) in schedule)
        {
            if (start >= chunkEnd || end <= chunkOffset) continue;

            // Compute local sample range within this chunk
            int localStart = global::System.Math.Max(0, start - chunkOffset);
            int localEnd = global::System.Math.Min(chunkSamples, end - chunkOffset);

            // Phase length (used for the tail-fade weight calculation)
            int phaseLength = end - start;

            // Pre-decide whether this phase has any crossfade. If both are 0
            // (non-dimensional mode), skip the per-sample weight branch entirely
            // for negligible-cost hot-path execution.
            bool hasCrossfade = headFade > 0 || tailFade > 0;

            // Sum sine waves for each ratio in this segment — double precision phase.
            // With crossfade active, each sample's contribution is weighted by the
            // smootherstep-position so adjacent dimensions blend seamlessly at boundaries.
            for (int r = 0; r < ratioValues.Length; r++)
            {
                double ratio = ratioValues[r];
                for (int i = localStart; i < localEnd; i++)
                {
                    float weight = 1.0f;
                    if (hasCrossfade)
                    {
                        // Position within phase: 0 at start, phaseLength at end
                        int posInPhase = (chunkOffset + i) - start;

                        if (headFade > 0 && posInPhase < headFade)
                        {
                            // First headFade samples ramp 0 → 1 via smootherstep
                            float x = posInPhase / (float)headFade;
                            weight = Smootherstep(x);
                        }
                        else if (tailFade > 0 && posInPhase >= phaseLength - tailFade)
                        {
                            // Last tailFade samples ramp 1 → 0 via smootherstep
                            float x = (phaseLength - posInPhase) / (float)tailFade;
                            weight = Smootherstep(x);
                        }
                    }

                    result[i] += weight * modIndex *
                        (float)System.Math.Sin(SacredConstants.TWO_PI_D * ratio * tChunk[i]);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Perlin smootherstep: 6t⁵ − 15t⁴ + 10t³. Smooth at both endpoints
    /// (zero first and second derivatives), no overshoot. Used by the
    /// modulation crossfade in dimensional mode and by every other crossfade
    /// curve in the codebase (Crystalline, Taygetan ratio bias). Kept local
    /// here so ComputeModulationChunk doesn't reach across files.
    /// </summary>
    private static float Smootherstep(float t)
    {
        t = global::System.Math.Clamp(t, 0f, 1f);
        return t * t * t * (t * (t * 6f - 15f) + 10f);
    }

    #endregion

    // Estimates the peak amplitude of the generated harmonics to compute
    // a normalization gain factor. Uses a quarter-period LFO sample to catch
    // the peak, then divides by NormalizationHeadroom (default PHI^0.75 ≈ 1.452,
    // tunable on SoundGenerator) so dynamic moments can breathe organically.
    #region Normalization

    /// <summary>
    /// Estimate normalization gain by generating a short test signal.
    /// Runs harmonics (and optionally Taygetan binaural overlay) through wave shaping
    /// at full envelope to find the peak, then returns 1/(peak × NormalizationHeadroom)
    /// — the headroom is configurable on SoundGenerator (default PHI^0.75 for ~0.689
    /// post-norm peak; was PHI for ~0.618).
    /// </summary>
    /// <param name="binauralPairs">
    /// Optional Stage 7 Taygetan binaural pairs. When non-null, each pair is synthesized
    /// into the test signal at the SAME envelope × 0.015 mix scale used in the runtime
    /// pipeline, so the peak estimate reflects the actual amplitude reaching Stage 8/10.
    /// Passing null preserves the original harmonic-only test (correct for Modes 0–4
    /// and Dimensional, since they don't activate Stage 7 binaural injection).
    /// </param>
    private float EstimateNormGain(float[] frequencies, float[] modDepths,
        MicrotonalLfo.LfoParams lfoParams, int sampleRate,
        (float Left, float Right)[]? binauralPairs = null)
    {
        // Generate a half-second test signal starting at the LFO quarter-period (peak)
        int estSamples = global::System.Math.Min(sampleRate / 2, 24000);
        double quarterPeriod = 1.0 / (4.0 * System.Math.Max(lfoParams.Lfo1Freq, 0.001));

        // Build test time array offset to the LFO peak — double precision
        var estT = new double[estSamples];
        for (int i = 0; i < estSamples; i++)
            estT[i] = quarterPeriod + (double)i / sampleRate;

        // Generate test harmonics at full envelope with LFO modulation.
        // Use waveShape: Triangle to match the runtime pipeline (triangle is the
        // default for the harmonic field in v5.1+). Without this, the test signal
        // would be sine while the real signal is triangle, and normGain would be
        // calibrated to the wrong waveform — same class of bug as the old Stage 7
        // omission. Triangle and sine share the same peak amplitude (±1.0) so the
        // gain estimate stays well-matched.
        var estEnv = new float[estSamples];
        Array.Fill(estEnv, 1.0f);
        var estLfo = MicrotonalLfo.Compute(estT, lfoParams);
        var estWave = HarmonicGenerator.GenerateHarmonics(estT, frequencies, estEnv, estLfo, modDepths,
            waveShape: WaveShape.Triangle);

        // Stage 7 simulation — REMOVED in Path 4+ architecture. The Taygetan
        // binaural sync is now woven directly through the Stage 6 harmonic field
        // (separate L/R freq arrays per voice, schedule-driven amp scaling).
        // The test signal already reflects the Taygetan tone formation because
        // it was generated from `frequencies` (which is the Taygetan freq set
        // when Taygetan mode is active) by the call above.
        //
        // The `binauralPairs` parameter is kept for API compatibility but no
        // longer needed for amplitude estimation — peak comes naturally from
        // the harmonic-field test signal. The breath peak (1 + 50%) is implicit
        // because we test at the LFO quarter-period (LFO peak); breath cycle
        // is similar magnitude. Real-world peak should fit comfortably under
        // the post-norm target.
        _ = binauralPairs;  // intentionally unused; preserved in signature

        // Wave-shape the same drive (2.5) used in runtime Stage 8 — saturation matters
        // for peak measurement because tanh limits peaks regardless of input amplitude.
        WaveShaper.Process(estWave, 2.5f);

        // Find peak amplitude (clamped to a floor of 0.001 to avoid div-by-zero)
        float peak = 0.001f;
        for (int i = 0; i < estSamples; i++)
            peak = MathF.Max(peak, MathF.Abs(estWave[i]));

        // Return gain using the configured headroom multiplier.
        // Default PHI^0.75 (~1.452) targets a ~0.689 post-norm peak — a tad off the
        // original PHI (~0.618 peak) so reverb summation and deep LFO modulation can
        // express their dynamics without the harness pulling them back. Lower the
        // value on SoundGenerator.NormalizationHeadroom for a hotter mix; raise it
        // toward PHI for the legacy conservative target.
        return 1.0f / (peak * NormalizationHeadroom);
    }

    #endregion
}
