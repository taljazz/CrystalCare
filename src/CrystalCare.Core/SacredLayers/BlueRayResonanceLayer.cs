using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.SacredLayers;

/// <summary>
/// Blue Ray Resonance Layer — 7th sacred layer (Arcturian transmission).
///
/// "Frequency is zero point. Always in the center, never wavering, never wafting away.
///  Time is not linear, nor does it exist in the ways of the old world, but it is a
///  single fabric. Past, present, and future are all meshed together...
///  Frequency is not meant to overpower others, but it is to unify others and yourself,
///  including the old attachments that must be severed."
///
/// Design principles drawn directly from the channeled message:
///
///   ZERO POINT → the core frequencies do NOT drift (no simplex phase wobble unlike
///                every other layer). The chord is the still axis around which the
///                other six layers breathe.
///
///   NEVER WAFTING → spatially locked to the stereo center via theta=0, phase=π/2
///                   in SoundGenerator's sacred-layer torus mixing (see wiring there).
///                   Every other sacred layer drifts around the torus; only this one
///                   holds the zero point.
///
///   SINGLE FABRIC OF TIME → three-strand temporal braid. Each chord tone is played
///                           simultaneously with phase offsets of −golden-angle,
///                           0, and +golden-angle radians — past, present, future
///                           meshed into one signal. Their interference pattern
///                           reveals the single fabric when listened to as a whole.
///
///   UNIFY, DO NOT OVERPOWER → every frequency in the Blue Ray Chord
///                             (444, 528, 741, 852, 963 Hz) already lives inside
///                             CrystalCare's Solfeggio or Tesla libraries, so this
///                             chord amplifies what is already sacred rather than
///                             introducing new dissonant content. It is the still
///                             axis made audible, not a new voice added.
///
/// Scale: 0.000618 (PHI_INVERSE / 1000) — sub-perceptual golden ratio reciprocal
/// Fade:  55 seconds (Fibonacci, the deeper fade for weightier layers)
/// Breath: 2% depth at BREATH_PHI_NEG100 (~207s cycle) — the stillest of all breaths,
///         one golden step BELOW Earth's own heartbeat.
/// </summary>
public sealed class BlueRayResonanceLayer : SacredLayerBase
{
    // Configuration: 55s Fibonacci fade (deep layer), minimum 2% breath depth
    // (stillest of all seven layers — honors "never wavering"), breathing one
    // golden step below Earth at BREATH_PHI_NEG100 (~207s cycle),
    // PHI_INVERSE / 1000 = 0.000618 output scale (sub-perceptual presence).
    #region Layer Configuration

    // Fibonacci-timed fade for the deep/weighty layers (matches Pleroma, Archon,
    // Merkaba, Water). The 7th layer arrives slowly and leaves slowly.
    protected override float FadeSeconds => 55.0f;

    // Breath center near unity (0.98) with the smallest depth (0.02) of all seven
    // layers. This minimizes the "wavering" of the layer while still participating
    // in CrystalCare's unified breath organism.
    protected override float BreathCenter => 0.98f;
    protected override float BreathDepth => 0.02f;

    // Breath frequency one golden step below Earth's heartbeat — the cosmic still
    // breath from which even Schumann emerges. Slowest of all seven breaths.
    protected override float BreathFreq => SacredConstants.BREATH_PHI_NEG100;

    // Output scale = PHI_INVERSE × 1/1000 = 0.000618 — golden ratio reciprocal,
    // sub-perceptual. Present enough to anchor, quiet enough never to dominate.
    protected override float OutputScale => 0.000618f;

    #endregion

    // Generates the Blue Ray chord as a three-strand temporal braid.
    // NO simplex phase wobble — the core frequencies are mathematically stable,
    // honoring the "never wavering" principle. Every other sacred layer uses
    // simplex-driven drift for organic aliveness; this one holds the zero point.
    //
    // The five chord tones (444/528/741/852/963 Hz) are each rendered three times
    // with the temporal phase offsets (−golden-angle / 0 / +golden-angle), giving
    // 15 pure sinusoids whose superposition reveals the "single fabric" of time.
    #region Signal Generation

    protected override float[] GenerateSignal(ReadOnlySpan<double> tChunk,
        float totalDuration, int n)
    {
        // The Blue Ray chord — 5 frequencies that already resonate within
        // CrystalCare's Solfeggio/Tesla sets, so the layer unifies rather than
        // introduces new harmonic content.
        var chord = SacredConstants.BLUE_RAY_CHORD;
        var chordWeights = SacredConstants.BLUE_RAY_WEIGHTS;

        // The three temporal strands — past, present, future — meshed into one fabric.
        var strandPhases = SacredConstants.BLUE_RAY_TEMPORAL_PHASES;
        var strandWeights = SacredConstants.BLUE_RAY_TEMPORAL_WEIGHTS;

        var result = new float[n];

        // Outer loop: each chord tone (5 total).
        // Middle loop: each temporal strand for that tone (3 total — past/present/future).
        // Inner loop: sample-by-sample accumulation using double-precision phase
        //             for long-session stability (the seventh layer is for ultimate
        //             knowing over long deep sessions — phase must stay exact).
        for (int f = 0; f < chord.Length; f++)
        {
            // Cast chord frequency to double for the phase computation — preserves
            // sub-millisecond accuracy over multi-hour sessions (v4.4.4 precision).
            double freq = chord[f];
            float chordWeight = chordWeights[f];

            for (int s = 0; s < strandPhases.Length; s++)
            {
                float phase = strandPhases[s];
                float strandWeight = strandWeights[s];
                float combinedWeight = chordWeight * strandWeight;

                for (int i = 0; i < n; i++)
                {
                    // Pure sinusoid — NO simplex phase wobble, honoring the
                    // "never wavering" zero point principle. The stillness
                    // IS the signal.
                    result[i] += combinedWeight *
                        (float)System.Math.Sin(SacredConstants.TWO_PI_D * freq * tChunk[i] + phase);
                }
            }
        }

        return result;
    }

    #endregion
}
