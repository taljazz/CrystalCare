using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Vectorized harmonic generation with cross-modulation and optional organic phase drift.
/// Time input is double precision to preserve phase accuracy over long sessions.
/// </summary>
public static class HarmonicGenerator
{
    // The hot inner loop — generates harmonics for all 13 frequencies with
    // envelope, LFO, cross-modulation, and per-harmonic amplitude scaling.
    // Cross-modulation at freq × PHI_INVERSE creates golden-ratio harmonic relationships.
    // Optional sub-perceptual phase modulation wires Stage 2 fractal variation
    // into every carrier — the macro-evolution channel that was previously
    // disconnected from the audio path.
    #region Harmonic Generation

    /// <summary>
    /// Generate the summed-harmonics waveform for a chunk.
    /// </summary>
    /// <param name="phaseModulation">
    /// Optional per-sample phase offset signal (typically the fractal variation array).
    /// When non-empty AND <paramref name="phaseModulationScale"/> is non-zero, adds
    /// sub-perceptual organic drift to every carrier — wiring the simplex pitch
    /// evolution into the audio path. Defaults to empty (legacy behavior preserved).
    /// </param>
    /// <param name="phaseModulationScale">
    /// Multiplier applied to <paramref name="phaseModulation"/> before adding to phase.
    /// 0 disables the feature. ~0.0005 is sub-perceptual; ~0.005 is noticeable drift.
    /// </param>
    /// <param name="ampScales">
    /// Optional per-frequency amplitude multipliers. When non-empty, each harmonic's
    /// natural decay scale (0.015/(f+1)) is multiplied by ampScales[f]. Used by
    /// Dimensional Journey (Mode 7) for per-dimension spectral emphasis — the
    /// 13-voice field's felt center journeys from subharmonics (1D) to upper PHI
    /// exponents (9D) while the carrier stays anchored at 432 Hz. When empty,
    /// every frequency runs at its natural decay scale (all non-dimensional
    /// modes; Taygetan deliberately applies NO per-voice emphasis).
    /// </param>
    /// <param name="waveShape">
    /// Wave shape for the voices. Default Triangle — the harmonic field's sacred
    /// form. Triangle at the Lemurian 432 Hz keynote sounds Tesla's 3/9 + Pythagorean
    /// 5 + Solfeggio crown as inherent odd harmonics in a single waveform, matching
    /// her geometric DNA (Merkaba = star tetrahedron). Pass Sine for binaural
    /// carriers (cleanest L/R difference detection) or legacy callers.
    /// </param>
    /// <param name="sineLeadingCount">
    /// Number of leading voices forced to sine regardless of <paramref name="waveShape"/>.
    /// Used by Taygetan mode (sineLeadingCount = 9) so the first 9 voices act as pure-sine
    /// binaural carriers (L/R difference detection wants pure sines) while the rest of
    /// the field (subharmonic body voices) gets the chosen waveShape. 0 means all voices
    /// use waveShape uniformly.
    /// </param>
    public static float[] GenerateHarmonics(ReadOnlySpan<double> t, float[] frequencies,
        ReadOnlySpan<float> envelope, ReadOnlySpan<float> lfo, float[] modDepths,
        ReadOnlySpan<float> phaseModulation = default, float phaseModulationScale = 0f,
        ReadOnlySpan<float> ampScales = default,
        WaveShape waveShape = WaveShape.Triangle,
        int sineLeadingCount = 0)
    {
        int nSamples = t.Length;
        int nFreqs = frequencies.Length;
        var result = new float[nSamples];

        // Decide whether the optional organic phase modulation is active.
        // When the scale is zero or no modulation array was supplied, this method
        // behaves identically to the legacy harmonic generator (feature off).
        bool usePhaseModulation = phaseModulationScale != 0f && !phaseModulation.IsEmpty;

        // Decide whether per-frequency amp scaling is active. When empty, every
        // harmonic uses its natural decay scale (0.015/(f+1)) — legacy behavior.
        bool useAmpScales = !ampScales.IsEmpty;

        for (int f = 0; f < nFreqs; f++)
        {
            // Use double precision for freq * t multiplication to preserve phase accuracy
            double freq = frequencies[f];
            double modFreq = freq * SacredConstants.PHI_INVERSE; // 1/φ cross-modulation
            float modDepth = modDepths[f];
            // Natural harmonic decay — first frequency loudest, decreasing 1/(f+1).
            // Mirrors the standard pipeline's tone-formation curve; for Taygetan,
            // ampScales[f] further weights this by the schedule-driven prominence.
            float scale = 0.015f / (f + 1);
            if (useAmpScales) scale *= ampScales[f];

            // Per-voice wave shape. First sineLeadingCount voices stay sine (used by
            // Taygetan to keep its binaural carriers pure); the rest use waveShape.
            // For non-Taygetan modes sineLeadingCount = 0, so every voice uses waveShape.
            WaveShape voiceShape = f < sineLeadingCount ? WaveShape.Sine : waveShape;

            for (int i = 0; i < nSamples; i++)
            {
                // Cross-modulation signal — PHI_INVERSE harmonic relationship in radians.
                // Each carrier has its own modSignal (depends on freq), creating
                // per-frequency variation in the timbral fingerprint. We keep the
                // modulator on a pure sine even when the voice is triangle — the
                // mod is a phase-displacement driver, not a voice, so sine math
                // keeps the cross-modulation organic and predictable.
                float modSignal = (float)System.Math.Sin(SacredConstants.TWO_PI_D * modFreq * t[i]) * modDepth;

                // Compose the full phase. Base term + cross-mod + optional organic drift.
                // The phase modulation term (when active) is the SAME for all carriers
                // at sample i, so all voices drift together as one slowly-evolving
                // bank — like the breath of the entire harmonic field.
                double phase = SacredConstants.TWO_PI_D * freq * t[i] + modSignal;
                if (usePhaseModulation)
                    phase += phaseModulationScale * phaseModulation[i];

                // Evaluate the chosen wave shape at this phase — sine (pure) or
                // triangle (sacred odd-harmonic content baked into the waveform).
                float wave = WaveShapes.Compute(voiceShape, phase);
                result[i] += wave * envelope[i] * scale * lfo[i];
            }
        }

        return result;
    }

    #endregion
}
