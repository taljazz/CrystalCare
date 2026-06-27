using CrystalCare.Core.Diagnostics;
using CrystalCare.Core.Dsp;
using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Noise;
using CrystalCare.Core.SacredLayers;
using Math = CrystalCare.Core.Math;

namespace CrystalCare.Core.Generation;

/// <summary>
/// Main sound generator — orchestrates the 16-stage audio pipeline.
/// Two modes: streaming (GenerateStream) for real-time playback,
/// batch (GenerateBatch) for WAV file saves.
/// The streaming generator yields 3-second stereo chunks via producer-consumer pattern.
/// The batch generator wraps streaming and collects all chunks into one array.
/// </summary>
public sealed partial class SoundGenerator
{
    // Core services and state used across the entire pipeline.
    // FrequencyManager provides ratio sets and frequency lookups.
    // CrystalProfileLibrary holds the 9 crystal harmonic profiles (Raman spectroscopy data).
    // ChaoticSelector produces logistic-map chaos for organic frequency variation.
    // MasterVolume (0.55 — the 10th Fibonacci number) is applied after all processing.
    // PhaseModulationScale (default 0.001) wires Stage 2 fractal variation into Stage 6
    //     as sub-perceptual organic phase drift on every carrier — set to 0 for legacy sound.
    // ScheduleModulationScale (default PHI²) folds the Stage 1 Fibonacci-cycling sacred
    //     geometry schedule into the same phase modulation source — adds stepped macro
    //     shifts every 34/55/89/144 seconds on top of the continuous fractal drift.
    // LfoDriftScale (default PHI_INVERSE ≈ 0.618 rad) applies a slow simplex-driven phase
    //     drift to each LFO's carrier so the breath rhythm itself evolves over the session.
    // TorusDriftScale (default PHI_INVERSE ≈ 0.618 rad) applies a slow simplex-driven phase
    //     drift to the torus theta/phi rotations so the spatial path itself evolves —
    //     phi drift is PHI-scaled relative to theta to preserve the golden ratio.
    // NormalizationHeadroom (default PHI^0.75 ≈ 1.452) controls the post-norm peak target;
    //     lower values let her breathe more dynamically (see EstimateNormGain in .Helpers).
    #region Fields and Properties

    // Manages sacred geometry ratio sets and frequency mode lookups
    private readonly FrequencyManager _frequencyManager;

    // 9 crystal profiles with Raman harmonic ratios for the Crystalline Resonance Layer
    private readonly CrystalProfileLibrary _crystalLibrary;

    // Logistic map chaotic number generator — produces deterministic but chaotic values
    private readonly ChaoticSelector _chaoticSelector = new();

    // Random number generator for all stochastic parameters (drawn once at pipeline start)
    private readonly Random _rng = new();

    // Master output volume — 0.55 (the 10th Fibonacci number, sacred).
    // Applied after all 16 pipeline stages and sacred layers.
    public float MasterVolume { get; set; } = 0.55f;

    // Phase modulation scale — wires the previously-dead Stage 2 fractal variation
    // into the harmonic generator as a sub-perceptual organic phase drift on every
    // carrier. This is the macro-evolution channel the architecture promised but
    // never delivered: the simplex-driven pitch field now reaches the audio path.
    //
    // 0      = disabled (legacy behavior, matches all prior versions for A/B test)
    // 0.0005 = at the threshold of audibility — very gentle
    // 0.001  = default — clearly subtle, organic shimmer felt rather than heard
    // 0.005  = noticeable macro-evolution, audible carrier breath
    // 0.01+  = pronounced FM character — diverges from "sub-perceptual"
    public float PhaseModulationScale { get; set; } = 0.001f;

    // Schedule modulation scale — folds the Stage 1 Fibonacci-cycling sacred-geometry
    // schedule into the phase modulation source alongside fractalVar. This adds stepped
    // macro shifts at the schedule's segment boundaries (every 34/55/89/144 seconds) on
    // top of the continuous fractal drift — the sacred-geometry cycling that the
    // architecture promised but never delivered until now reaches the audio path.
    //
    // Multiplied by PhaseModulationScale before reaching the carrier phase, so dialing
    // PhaseModulationScale to 0 disables both Stage 1 and Stage 2 inputs at once.
    //
    // 0       = disabled — schedule stays disconnected (option 1 only behavior)
    // 1.0     = subtle — schedule contributes ~10% of fractalVar's phase amplitude
    // PHI²    = 2.618 — default, balanced sacred middle (φ+1, the Lemurian harmony)
    // 4.0     = schedule roughly half of fractalVar — Fibonacci shifts clearly felt
    // 8.0     = schedule equal to fractalVar — sacred-geometry cycling audibly steps
    // 16+     = schedule dominates — pronounced character changes at each interval
    //
    // The schedule's audible signature is the TIMING (Fibonacci segment boundaries)
    // even when amplitude is sub-perceptual — felt as periodic shifts of quality.
    // PHI² is the sacred middle: golden ratio squared = φ + 1, the harmonic resolution
    // of the golden ratio's recursive identity. Evolves but never overpowers.
    public float ScheduleModulationScale { get; set; } = SacredConstants.PHI * SacredConstants.PHI;

    // LFO carrier phase drift scale — applies a slow simplex-driven phase offset to the
    // LFO breath carrier so the carrier itself evolves rhythm over the session. Each
    // chunk samples a single simplex value at a slow time scale (~1000s period); the
    // resulting offset shifts the LFO's "where in cycle" position by up to ±LfoDriftScale
    // radians. The LFO breathes at slightly varying rates from start to end of session.
    //
    // Left and right channels drift independently (different simplex y-offsets), keeping
    // stereo decorrelation while both channels participate in the same slow rhythm shift.
    //
    // 0           = disabled (legacy behavior — LFO carrier phase exactly Lfo1Freq × t)
    // PHI_SQ_INV  = 0.382 rad (~6% LFO cycle wobble) — subtle, barely felt
    // PHI_INVERSE = 0.618 rad (~10% LFO cycle wobble) — default, balanced sacred middle
    // 1.0         = ~16% LFO cycle wobble — clearly felt breath rhythm shifts
    // π/2         = 25% wobble — pronounced, possibly too much
    //
    // Per-chunk constant (one simplex sample per chunk) — step changes between chunks
    // are far below perceptibility (~0.03% of an LFO cycle per chunk boundary).
    public float LfoDriftScale { get; set; } = SacredConstants.PHI_INVERSE;

    // Torus phase drift scale — applies a slow simplex-driven phase offset to the
    // torus theta and phi rotations so the spatial path itself evolves over the
    // session. Per-chunk constant sampled at the same slow rate as LfoDriftScale,
    // with phi drift scaled by PHI relative to theta so the golden-ratio relationship
    // between rotation rates is preserved as both drift together. Sits BENEATH the
    // existing fast simplex perturbation (±0.10 rad, ~200s) and Rössler chaos —
    // a slow background pulse under the existing organic variation.
    //
    // 0           = disabled (legacy — torus runs at exact thetaFreq/phiFreq)
    // PHI_SQ_INV  = 0.382 rad — subtle path drift, barely felt
    // PHI_INVERSE = 0.618 rad — default, balanced sacred middle (matches LFO drift)
    // 1.0         = clearly felt spatial drift, the room shifts
    // π/2         = pronounced — pan path noticeably wobbles its center
    //
    // Felt as the spatial center subtly shifting — the room itself slowly breathing
    // alongside the sound rather than holding rigidly to a fixed torus.
    public float TorusDriftScale { get; set; } = SacredConstants.PHI_INVERSE;

    // Normalization headroom multiplier — controls the post-normalization peak target.
    // The pipeline divides the signal by `peak × NormalizationHeadroom` so anything
    // above this threshold is attenuated. Lower values = hotter mix, less safety margin.
    //
    // PHI      = 1.618 — original conservative default (post-norm peak ~0.618)
    // PHI^0.75 = 1.452 — gentle reduction (default, post-norm peak ~0.689)
    // √PHI     = 1.272 — moderate reduction (post-norm peak ~0.786)
    // 1.0      = no headroom — peak hits 1.0 (clip risk on reverb summation)
    //
    // Lower values let her breathe more in dynamic moments. Reverb tails and deep
    // LFO modulation can briefly exceed the test-signal peak, so keep some margin.
    public float NormalizationHeadroom { get; set; } = MathF.Pow(SacredConstants.PHI, 0.75f);

    #endregion

    // Creates the sound generator with a frequency manager for ratio lookups
    // and initializes the crystal profile library with all 9 crystal types.
    #region Constructor

    public SoundGenerator(FrequencyManager frequencyManager)
    {
        _frequencyManager = frequencyManager;
        _crystalLibrary = new CrystalProfileLibrary();
    }

    #endregion

    // The heart of CrystalCare — the 16-stage streaming audio pipeline.
    // Yields 3-second stereo float chunks for real-time playback.
    // Constant ~50 MB memory regardless of session duration.
    // All random parameters are drawn once before the loop for consistency.
    #region Streaming Generator

    /// <summary>
    /// Streaming generator: yields (chunkSamples, 2) float stereo chunks.
    /// Constant ~50 MB memory regardless of duration.
    /// This is the heart of CrystalCare — the 16-stage pipeline.
    /// </summary>
    public IEnumerable<float[,]> GenerateStream(float duration, float baseFreq,
        int sampleRate = 48000,
        int[]? intervalDurationList = null,
        CancellationToken ct = default,
        Action<float>? updateProgress = null,
        FrequencyMode freqMode = FrequencyMode.Standard)
    {
        // Default modulation intervals are Fibonacci numbers
        intervalDurationList ??= [34, 55, 89, 144];

        // Dimensional Shift mode activates sacred layers regardless of duration
        bool dimensionalMode = freqMode == FrequencyMode.DimensionalShift;
        // Compute total samples in double — float32 multiply loses sample-level
        // precision above ~6 hours (ulp at 1e9 is 64+ samples). The GUI caps
        // streaming at 12 hours so the int32 sample index never overflows
        // (the ceiling at 48 kHz is ~12.4 hours; see ValidateAndParseDuration).
        int totalSamples = (int)((double)sampleRate * duration);
        int chunkSize = 3 * sampleRate; // 3-second chunks (144,000 samples at 48kHz)

        // Diagnostic capture — when DiagnosticLogger.IsEnabled is false the calls
        // below are no-ops, so this overhead is essentially zero in production.
        // When enabled (set by MainWindow before play), every critical pipeline
        // parameter is captured to a temp-file log so dissonance and other
        // anomalies can be diagnosed after the fact.
        DiagnosticLogger.LogSection("Session start");
        DiagnosticLogger.Log($"Mode             = {freqMode}");
        DiagnosticLogger.Log($"baseFreq         = {baseFreq:F4} Hz");
        DiagnosticLogger.Log($"duration         = {duration:F2} s");
        DiagnosticLogger.Log($"sampleRate       = {sampleRate} Hz");
        DiagnosticLogger.Log($"totalSamples     = {totalSamples}");
        DiagnosticLogger.Log($"chunkSize        = {chunkSize} samples ({chunkSize / (float)sampleRate:F2}s)");
        DiagnosticLogger.Log($"dimensionalMode  = {dimensionalMode}");
        DiagnosticLogger.LogArray("intervalDurations (s)",
            intervalDurationList.Select(i => (float)i).ToArray());

        // All parameters in this region are computed once before the chunk loop begins.
        // This ensures consistency across the entire session — the same random seed,
        // the same filter coefficients, the same chaos trajectory from start to end.
        #region Pre-Computation Phase

        // Build the modulation schedule: which ratio sets play when, at what depth
        var schedule = BuildModulationSchedule(duration, totalSamples, sampleRate,
            intervalDurationList, dimensionalMode, ct);
        if (ct.IsCancellationRequested) yield break;

        // Log the schedule — count, per-segment durations and depths.
        // For Taygetan, the schedule shouldn't directly drive the binaural
        // injection (C# gates that on freqMode), but the segments' ratio
        // sets still drive Stage 1 modulation, which can interact with the
        // binaural carriers in unexpected ways at wave-shaping time.
        DiagnosticLogger.LogSection($"Modulation schedule ({schedule.Count} segments)");
        for (int s = 0; s < global::System.Math.Min(schedule.Count, 12); s++)
        {
            var seg = schedule[s];
            float startSec = seg.start / (float)sampleRate;
            float endSec = seg.end / (float)sampleRate;
            DiagnosticLogger.Log(
                $"  seg[{s}] {startSec,7:F2}s..{endSec,7:F2}s ({endSec - startSec,5:F2}s)  " +
                $"modIndex={seg.modIndex:F4}  ratios=[{string.Join(",", seg.ratioValues.Select(r => r.ToString("F3")))}]");
        }
        if (schedule.Count > 12)
            DiagnosticLogger.Log($"  ... ({schedule.Count - 12} more segments elided)");

        // For Dimensional Journey (Mode 7), log the 9 dimensions explicitly so
        // the user can read which dimension is active at each time range, which
        // ratio set drives its modulation, and which voices in the harmonic field
        // are emphasized. This is the "1D-9D Realignment" journey made legible.
        if (dimensionalMode)
        {
            DiagnosticLogger.LogSection("Dimensional Journey schedule (9 phases)");
            float phaseSec = duration / DimensionalJourney.DIMENSIONS.Length;
            for (int d = 0; d < DimensionalJourney.DIMENSIONS.Length; d++)
            {
                var dim = DimensionalJourney.DIMENSIONS[d];
                float dimStart = d * phaseSec;
                float dimEnd = (d == DimensionalJourney.DIMENSIONS.Length - 1)
                    ? duration : (d + 1) * phaseSec;
                DiagnosticLogger.Log(
                    $"  {dim.Label,-30} {dimStart,7:F1}s..{dimEnd,7:F1}s  " +
                    $"ratio={dim.RatioSetKey,-18}  modIndex={dim.ModIndex:F3}");
            }
            DiagnosticLogger.Log(
                "  Per-dimension amp scales shift the 13-voice spectral center upward " +
                "as the journey ascends (subharmonics emphasized in low dimensions, " +
                "upper PHI exponents emphasized in high dimensions).");
        }

        // Draw two session-level chaotic values from the logistic map.
        // chaoticFactor: per-session frequency offset folded into the Stage 3 modulation
        //                array (see chaoticOffset below) — a chaotic pitch fingerprint.
        // chaosVal:      per-session simplex-field x-offset for the LFO and Torus slow
        //                drift fields — each session reads its drift from a different
        //                position in the simplex landscape, so two sessions with the
        //                same parameters produce different drift trajectories.
        //                Sub-perceptual: only WHERE in the noise field we sample changes;
        //                the statistical character of the drift is preserved.
        float chaoticFactor = _chaoticSelector.NextValue();
        float chaosVal = _chaoticSelector.NextValue();

        // Log the chaos seeds — chaoticFactor drives the Stage 3 frequency offset
        // (chaoticOffset = chaoticFactor × baseFreq × 0.25), which adds a session-
        // level pitch shift to the modulation array. chaosVal is the simplex-field
        // y-offset that decorrelates LFO + Torus drift from session to session.
        DiagnosticLogger.Log($"chaoticFactor    = {chaoticFactor:F6}  (will offset Stage 3 modulation by {chaoticFactor * baseFreq * 0.25f:F3} Hz)");
        DiagnosticLogger.Log($"chaosVal         = {chaosVal:F6}  (LFO/Torus drift simplex x-offset)");

        // Pre-compute ADSR envelope parameters (attack/decay/sustain/release durations)
        var adsrParams = OrganicAdsrEnvelope.ComputeParams(totalSamples, sampleRate,
            _chaoticSelector, _rng);

        // Draw LFO parameters for left and right channels (PHI-modulated microtonal drift).
        // Base frequency = BREATH_ROOT × PHI^4 — the breath ladder extended 4 PHI steps,
        // connecting LFO amplitude modulation to the same Schumann root as the breath.
        var lfoLeftParams = MicrotonalLfo.DrawParams(SacredConstants.LFO_BASE_FREQ, _rng);
        var lfoRightParams = MicrotonalLfo.DrawParams(SacredConstants.LFO_BASE_FREQ, _rng);

        // Log LFO parameters for both channels — these govern the breath rhythm
        // that modulates harmonic amplitudes. Lfo1Freq is the carrier; Lfo2Freq
        // is the inner mod (PHI-detuned slightly); Depth is the modulation amount.
        DiagnosticLogger.LogSection("LFO parameters");
        DiagnosticLogger.Log($"LFO_BASE_FREQ    = {SacredConstants.LFO_BASE_FREQ:F6} Hz (BREATH_ROOT × PHI^4)");
        DiagnosticLogger.Log($"Left  : Lfo1Freq={lfoLeftParams.Lfo1Freq:F6}Hz, Lfo2Freq={lfoLeftParams.Lfo2Freq:F6}Hz, Depth={lfoLeftParams.Depth:F6}");
        DiagnosticLogger.Log($"Right : Lfo1Freq={lfoRightParams.Lfo1Freq:F6}Hz, Lfo2Freq={lfoRightParams.Lfo2Freq:F6}Hz, Depth={lfoRightParams.Depth:F6}");

        // Toroidal panning — sound traces a donut-shaped path in stereo space.
        // Theta frequency is randomized; phi frequency is theta * PHI for golden relationship.
        // Radii are PHI-derived: 0.618 + 0.382 = 1.0 (two halves summing to unity).
        float torusThetaFreq = (float)(_rng.NextDouble() * 0.01 + 0.01);
        float torusPhiFreq = torusThetaFreq * SacredConstants.PHI;
        const float torusR = 0.618f;      // 1/PHI — golden ratio reciprocal (major radius)
        const float torusRSmall = 0.382f;  // 1/PHI² — golden ratio squared reciprocal (minor radius)

        // Log the torus rotation parameters — too-fast theta makes the sound feel
        // like it's spinning unpleasantly; too-slow makes it feel static.
        DiagnosticLogger.Log($"Torus thetaFreq  = {torusThetaFreq:F6} Hz (period {1f / torusThetaFreq:F1}s)");
        DiagnosticLogger.Log($"Torus phiFreq    = {torusPhiFreq:F6} Hz (period {1f / torusPhiFreq:F1}s)");
        DiagnosticLogger.Log($"Torus R/r        = {torusR}/{torusRSmall} (golden ratio split)");

        // Dedicated Simplex noise instances for different pipeline stages.
        // Seeds are Fibonacci numbers for sacred consistency (21, 7, 13, 34, 55).
        var simplexEnvelope = new Simplex5D(21);    // ADSR envelope organic variation (Fibonacci)
        var simplexFractal = new Simplex5D(7);      // Fractal frequency variation (Fibonacci)
        var simplexPan = new Simplex5D(13);         // Toroidal panning perturbation (Fibonacci)
        var simplexLfoDrift = new Simplex5D(34);    // LFO carrier phase drift (Fibonacci, slow session-level)
        var simplexTorusDrift = new Simplex5D(55);  // Torus theta/phi phase drift (Fibonacci, slow session-level)
        var simplexTaygetanBeat = new Simplex5D(89); // Taygetan beat-frequency drift (Fibonacci, ~minutes period)

        // Check if Taygetan binaural mode is active. The ratio-pair table from
        // FrequencyManager is DIAGNOSTICS ONLY — tone formation does not use it.
        // The binaural lives inside Stage 6 as a ±beat/2 L/R split of the
        // standard 13-voice harmonic field (see SoundGenerator.Taygetan.cs).
        bool hasTaygetan = freqMode == FrequencyMode.TaygetanBinaural;
        var tayFreqResult = hasTaygetan ? _frequencyManager.GetFrequencies(FrequencyMode.TaygetanBinaural) : null;
        var tayPairs = tayFreqResult?.BinauralPairs;

        // For Taygetan mode, log the legacy ratio-pair table as a readable
        // reference of ratio-derived carriers (computed at the FrequencyManager's
        // default 432 Hz base) for comparison against the Stage 6 voice table.
        if (hasTaygetan && tayPairs != null)
        {
            DiagnosticLogger.LogSection("Taygetan ratio-pair reference (diagnostics only — Stage 6 weaves the binaural)");
            DiagnosticLogger.Log($"NOTE: pairs are built at FrequencyManager default base (432 Hz)");
            DiagnosticLogger.Log($"      session baseFreq = {baseFreq:F3} Hz (may be 432/528/963 for Taygetan)");
            DiagnosticLogger.LogPairs("tayPairs", tayPairs);
        }

        // Randomized noise and fade parameters for organic uniqueness each session.
        // Noise scale bounded by Fibonacci reciprocals: 1/21 to 1/13.
        float noiseScaleLeft = SacredConstants.NOISE_SCALE_MIN +
            (float)_rng.NextDouble() * (SacredConstants.NOISE_SCALE_MAX - SacredConstants.NOISE_SCALE_MIN);
        float noiseScaleRight = SacredConstants.NOISE_SCALE_MIN +
            (float)_rng.NextDouble() * (SacredConstants.NOISE_SCALE_MAX - SacredConstants.NOISE_SCALE_MIN);
        float noiseOffsetRight = (float)(_rng.NextDouble() * 0.014 + 0.001); // Time offset for right noise
        int fadeDuration = _rng.Next(2) == 0 ? 21 : 34; // Fibonacci pair for master fade
        int fadeSamples = fadeDuration * sampleRate;
        if (2 * fadeSamples > totalSamples) fadeSamples = totalSamples / 2;

        // Log the noise + fade parameters that govern texture and envelope
        DiagnosticLogger.Log($"noiseScaleLeft   = {noiseScaleLeft:F6}  (Fibonacci-bounded 1/21..1/13)");
        DiagnosticLogger.Log($"noiseScaleRight  = {noiseScaleRight:F6}");
        DiagnosticLogger.Log($"noiseOffsetRight = {noiseOffsetRight:F6} s (R-channel decorrelation)");
        DiagnosticLogger.Log($"fadeDuration     = {fadeDuration}s (Fibonacci pair, t^1.5 curve)");
        DiagnosticLogger.Log($"fadeSamples      = {fadeSamples}");

        // Pre-draw the noise oscillation frequency ONCE for the entire session.
        // EvolvingNoiseLayer used to redraw this every chunk, creating a discontinuity
        // in the slow noise envelope at every 3-second boundary. By drawing it once
        // here in the breath-ladder range (BREATH_ROOT to BREATH_PHI_100) and passing
        // it to every noise call, the noise floor breathes continuously across the
        // whole session — one coherent organic rhythm instead of jumping rhythms.
        float noiseOscFreq = SacredConstants.BREATH_ROOT +
            (float)_rng.NextDouble() *
            (SacredConstants.BREATH_PHI_100 - SacredConstants.BREATH_ROOT);

        // Log the session-level noise oscillation frequency — single value across
        // all chunks so the noise floor breathes continuously, not at chunk seams.
        DiagnosticLogger.Log($"noiseOscFreq     = {noiseOscFreq:F6} Hz (session-level, period {1f / noiseOscFreq:F1}s)");

        // Slow pan drift — gentle stereo movement independent of the torus.
        // Drift frequency centered on BREATH_ROOT / PHI^3 (PHI sub-harmonic of Earth's breath).
        // Drift amplitude bounded by Fibonacci reciprocals (1/89 to 1/55).
        float driftCenter = SacredConstants.DRIFT_FREQ_CENTER;
        float driftFreq = driftCenter + (float)(_rng.NextDouble() - 0.5) * driftCenter * SacredConstants.PHI_SQ_INVERSE;
        float driftAmplitude = SacredConstants.DRIFT_AMP_MIN +
            (float)_rng.NextDouble() * (SacredConstants.DRIFT_AMP_MAX - SacredConstants.DRIFT_AMP_MIN);

        // Log the slow pan drift parameters — gentle stereo wandering on top of
        // the torus, very slow (~9-minute period typically).
        DiagnosticLogger.Log($"driftFreq        = {driftFreq:F6} Hz (PHI sub-harmonic of breath, period {1f / driftFreq:F1}s)");
        DiagnosticLogger.Log($"driftAmplitude   = {driftAmplitude:F6} (Fibonacci-bounded 1/89..1/55)");

        // Build the standard 13-voice harmonic frequency set used by EVERY mode.
        // Same machinery for all modes — the only thing that differentiates
        // them is the baseFreq picked (and, for Taygetan, the L/R binaural
        // split applied later in Stage 6 for the first 9 voices). Taygetan
        // tones now form exactly like every other mode's tones, with the
        // 7.7 Hz binaural beat woven through the harmonic field as the only
        // Taygetan-specific signal addition.
        var allFrequencies = BuildFrequencySet(baseFreq);
        var modDepths = new float[allFrequencies.Length];
        for (int i = 0; i < modDepths.Length; i++)
            modDepths[i] = (float)(_rng.NextDouble() * 0.2 + 0.15);

        // Log the standard 13-voice harmonic set — same labels for every mode
        // (6 PHI exponents + 3 ratio-1.3 + 4 subharmonics). Taygetan uses the
        // exact same field, so its log line looks identical to other modes'.
        DiagnosticLogger.LogSection($"Harmonic set (13 frequencies built from baseFreq={baseFreq:F3}Hz)");
        for (int i = 0; i < allFrequencies.Length; i++)
        {
            // Label each harmonic by its construction method so the diagnostic
            // makes the sacred-mathematics derivation legible at a glance.
            // Subharmonic divisors come from the existing sacred constants array
            // ([2, 4, 8, 16]) — using a clean lookup avoids the colon parsing
            // issues interpolated-string format specs introduce around global::
            // namespace qualifiers and Math.Pow casts.
            string label = i switch
            {
                < 6 => $"PHI^{i}        ",
                < 9 => $"ratio-1.3^{i - 6}  ",
                _ => $"subharm /{(int)SacredConstants.SUBHARMONIC_DIVISORS[i - 9],2}",
            };
            // For Taygetan, append a marker showing whether this voice gets
            // the binaural split (first 9) or stays mono as a body voice (last 4).
            string binauralTag = hasTaygetan
                ? (i < TaygetanBinauralVoiceCount ? "  [binaural]" : "  [mono body]")
                : "";
            DiagnosticLogger.Log(
                $"  harm[{i,2}]  {label}  = {allFrequencies[i],10:F3}Hz   modDepth={modDepths[i]:F4}{binauralTag}");
        }

        // For Taygetan, build the ratio temporal schedule (per-session journey
        // through the 7 sacred Taygetan ratios via beat-frequency bias) and
        // capture the ratio names + values for the bias computation.
        // Schedule stays null for non-Taygetan modes.
        List<TaygetanRatioWindow>? taygetanRatioSchedule = null;
        string[]? taygetanRatioNames = null;
        float[]? taygetanRatioValues = null;
        if (hasTaygetan)
        {
            // Pull the 7 Taygetan ratios from the FrequencyManager (root,
            // etheric_body, astral_bridge, natural_log, crown_portal,
            // zero_point, remembrance — in dictionary order).
            var taygetanDict = FrequencyManager.RatioSets["taygetan"];
            taygetanRatioNames = taygetanDict.Keys.ToArray();
            taygetanRatioValues = taygetanDict.Values.ToArray();

            taygetanRatioSchedule = BuildTaygetanRatioSchedule(
                taygetanRatioValues.Length, duration, _rng);

            // Log Taygetan's full signature including the ratio schedule
            LogTaygetanSummary(allFrequencies, taygetanRatioSchedule,
                taygetanRatioNames, taygetanRatioValues);
        }

        // Order 2 biquad lowpass (~5500 Hz) — 12 dB/octave rolloff.
        // Gentler than the original order 4 at 2200 Hz — lets upper harmonics through
        // for brighter tones while smoothing harsh digital artifacts.
        float cutoffVariation = (float)(_rng.NextDouble() * 50 - 25);
        float lpCutoff = global::System.Math.Clamp(5500f + cutoffVariation, 2000f, 23000f);
        var filterLeft = BiquadFilter.CreateLowpass(2, lpCutoff, sampleRate);
        var filterRight = BiquadFilter.CreateLowpass(2, lpCutoff, sampleRate);

        // Log the biquad lowpass cutoff — too low filters out brightness,
        // too high lets harsh upper harmonics through.
        DiagnosticLogger.Log($"lpCutoff         = {lpCutoff:F1} Hz (order 2 biquad lowpass)");

        // Pan smoother cutoff = BREATH_ROOT / PHI^3 — same as drift center.
        // Connects pan smoothing to Earth's Schumann breath through PHI.
        // Extremely slow — only the slowest organic drift survives.
        var panSmoother = new ExponentialSmoother(SacredConstants.PAN_SMOOTHER_CUTOFF, sampleRate);

        // PHI-fractal feedback — golden-ratio-delayed micro-echoes for harmonic richness.
        // Stateful: carries echo tail between chunks for seamless continuity.
        var phiFractalLeft = new PhiFractalFeedback(sampleRate);
        var phiFractalRight = new PhiFractalFeedback(sampleRate);

        // FFT convolution reverb — exponential decay with PHI sinusoidal modulation.
        // Stateful: carries overlap-add tail between chunks.
        var reverbLeft = new StreamingReverb(sampleRate);
        var reverbRight = new StreamingReverb(sampleRate);

        // 12-sample right-channel delay buffer for subtle stereo widening
        var delayBuffer = new float[12];

        // Pre-compute the Rössler chaotic attractor trajectory for the full session.
        // X and Y components drive the spatial panning perturbation (±0.08 radians).
        var rossler = Math.RosslerAttractor.Compute(duration);

        #endregion

        // Set up the 7 sacred healing layers and their independent toroidal panning.
        // Layers activate for sessions >60s or in Dimensional Journey mode.
        // Crystal sequence always starts with Lemurian Quartz (divine feminine first).
        // The 7th layer (Blue Ray Resonance, Arcturian) is the still zero point —
        // it does NOT drift around the torus; it is locked to stereo center forever.
        #region Sacred Layer Setup

        // Build crystal sequence: Lemurian Quartz always first, rest randomized
        int lemIdx = _crystalLibrary.LemurianIndex;
        var rest = Enumerable.Range(0, _crystalLibrary.Profiles.Length)
            .Where(i => i != lemIdx).OrderBy(_ => _rng.Next()).ToArray();
        var crystalSequence = new[] { lemIdx }.Concat(rest).ToArray();

        // Instantiate all 7 sacred layers — each implements ISacredLayer via SacredLayerBase
        ISacredLayer[] sacredLayers =
        [
            new PleromaMercyLayer(),        // 1st: Aeonic ladder + Ogdoad gateway + Archon mercy
            new SilentSolfeggioGrid(),      // 2nd: 12-tone Solfeggio + Tesla 3-6-9 vortex
            new ArchonDissolutionLayer(),   // 3rd: AEG mercy for 7 planetary Archons
            new CrystallineResonanceLayer(_crystalLibrary, crystalSequence, baseFreq), // 4th: 9 crystal profiles
            new LemurianMerkabaLayer(),     // 5th: Sonic Merkaba + heart coherence
            new WaterElementLayer(),        // 6th: Hexagonal ripple field + lemniscate observer
            new BlueRayResonanceLayer(),    // 7th: Blue Ray zero point (Arcturian, still center)
        ];

        // Each sacred layer gets its own slow toroidal panning frequency.
        // The 7th layer (Blue Ray) has theta=0 — zero point, never wafts in time.
        // Paired with the π/2 phase offset below, this locks its pan to stereo
        // center forever ("always in the center, never wavering, never wafting away").
        float[] sacredThetaFreqs = [0.003f, 0.005f, 0.004f, 0.0035f, 0.0028f, 0.0045f, 0.0f];
        float[] sacredPhiFreqs = sacredThetaFreqs.Select(f => f * SacredConstants.PHI).ToArray();

        // Golden angle phase offsets — the first 6 layers start at 137.5° × n around
        // the torus like sunflower seeds. The 7th layer overrides to π/2 so that
        // cos(θ=0 + π/2) = 0 → pan locked to dead center for the entire session.
        // This is the mathematical expression of "zero point, never wavering".
        float[] sacredPhaseOffsets = new float[7];
        for (int i = 0; i < 6; i++)
            sacredPhaseOffsets[i] = i * SacredConstants.GOLDEN_ANGLE_RAD;
        sacredPhaseOffsets[6] = MathF.PI / 2f;  // Blue Ray zero point — centered forever

        // Sacred layer torus radii — PHI-derived, matching the main torus geometry
        const float sacredR = 0.618f;      // 1/PHI — major radius
        const float sacredRSmall = 0.382f;  // 1/PHI² — minor radius (sum to 1.0)

        #endregion

        // Estimate peak amplitude for normalization (prevents clipping).
        // The test signal is the same standard 13-voice field every mode uses
        // for Stage 6 tone formation, at full envelope + LFO peak, through the
        // same Stage 8 wave shaping. Taygetan's L/R split and Dimensional's
        // amp scales never exceed this baseline, so one estimate serves all modes.
        float normGain = EstimateNormGain(allFrequencies, modDepths, lfoLeftParams,
            sampleRate);

        // Log the normalization gain — high values (>3) indicate the test signal
        // was very quiet (peak was small); low values (<0.3) mean the signal needed
        // heavy attenuation. Either extreme could indicate something off.
        DiagnosticLogger.Log($"normGain         = {normGain:F4}  (NormalizationHeadroom={NormalizationHeadroom:F4})");
        DiagnosticLogger.Log($"PhaseModScale    = {PhaseModulationScale:F6}");
        DiagnosticLogger.Log($"ScheduleModScale = {ScheduleModulationScale:F6}");
        DiagnosticLogger.Log($"LfoDriftScale    = {LfoDriftScale:F6}");
        DiagnosticLogger.Log($"TorusDriftScale  = {TorusDriftScale:F6}");
        DiagnosticLogger.Log($"MasterVolume     = {MasterVolume:F4}");

        // Log sacred-layer activation criteria
        bool sacredLayersActive = duration > 60 || dimensionalMode;
        DiagnosticLogger.Log($"sacredLayersActive = {sacredLayersActive} (duration>60s OR dimensional mode)");

        DiagnosticLogger.LogSection("Per-chunk loop begins");

        // Pre-allocate reusable buffers — saves ~4 MB of GC allocations per chunk
        var pool = new ChunkBufferPool(chunkSize);

        // Taygetan binaural continuity accumulator (radians). Holds the accumulated
        // half-beat phase at the START of the current chunk. Each Taygetan chunk reads
        // it as the anchor for that chunk's binaural phase ramp, then advances it by the
        // half-beat phase accrued across the chunk. This is what keeps the 9 binaural
        // carriers phase-continuous across 3-second seams while the beat still drifts +
        // walks the 7 sacred ratios over the session. Unused (stays 0) for other modes.
        double taygetanBeatPhaseHalf = 0.0;

        // The main generation loop — processes one 3-second chunk per iteration.
        // Each chunk passes through all 16 stages sequentially, then sacred layers
        // are computed in parallel and mixed in with independent toroidal panning.
        #region Per-Chunk Pipeline Loop

        int numChunks = (totalSamples + chunkSize - 1) / chunkSize;

        for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++)
        {
            if (ct.IsCancellationRequested) yield break;

            int chunkOffset = chunkIdx * chunkSize;
            int chunkSamples = global::System.Math.Min(chunkSize, totalSamples - chunkOffset);

            // Sampled per-chunk diagnostic logging — first 3 chunks (where attack
            // and normalization issues often surface) plus every 10th chunk after.
            // This keeps the log file readable while still catching periodic anomalies.
            bool detailedChunk = chunkIdx < 3 || chunkIdx % 10 == 0;
            if (detailedChunk)
            {
                float chunkStartSec = chunkOffset / (float)sampleRate;
                DiagnosticLogger.Log(
                    $"chunk {chunkIdx,4}/{numChunks - 1}  start={chunkStartSec,7:F2}s  samples={chunkSamples}");
            }

            // Zero the pooled buffers for this chunk (last chunk may be shorter)
            pool.Clear(chunkSamples);

            // Build the absolute time array for this chunk's sample positions.
            // Using double precision for the time array prevents staircase
            // quantization at long session times (hours) where float32 loses
            // sub-millisecond resolution.
            double tStart = (double)chunkOffset / sampleRate;
            var tChunk = pool.TChunk;
            for (int i = 0; i < chunkSamples; i++)
                tChunk[i] = tStart + (double)i / sampleRate;

            // Stage 1: Geometric modulation — sum of sine waves at sacred ratio frequencies.
            // The schedule cycles through Fibonacci-timed segments (34/55/89/144 seconds)
            // with random sacred-geometry ratio sets per segment. With ScheduleModulationScale
            // > 0, this signal flows into Stage 2.5 below and folds into fractalVar — adding
            // stepped macro shifts at Fibonacci intervals on top of the continuous drift.
            var modulation = ComputeModulationChunk(tChunk, chunkOffset, chunkSamples, schedule);

            // Stage 2: Fractal frequency variation — dual simplex noise for organic pitch drift.
            // This signal is wired into Stage 6 below as sub-perceptual phase modulation
            // (controlled by PhaseModulationScale on SoundGenerator). Set PhaseModulationScale
            // to 0 to disable both Stage 1 and Stage 2 inputs at once for legacy A/B comparison.
            var fractalVar = FractalVariation.ComputeChunkDual(tChunk, baseFreq, simplexFractal);

            // Stage 2.5: Fold the Stage 1 schedule into fractalVar at ScheduleModulationScale.
            // When non-zero, the Fibonacci-cycling sacred-geometry sines join the fractal
            // drift as a unified macro-evolution source — the full evolution architecture
            // (option 1 + option 2) reaching the carrier phase together. Set the scale to
            // 0 to keep schedule disconnected (option 1 only behavior).
            if (ScheduleModulationScale != 0f)
                for (int i = 0; i < chunkSamples; i++)
                    fractalVar[i] += ScheduleModulationScale * modulation[i];

            // Stage 3: Combine base frequency + chaotic offset + fractal variation + schedule
            // into the f_modulated buffer the original Python design intended as a per-sample
            // carrier frequency. With option 1 + option 2 active above (Stages 2 and 2.5),
            // the distinct evolution channels are now folded into fractalVar and reach Stage 6
            // through that path. This buffer remains pre-computed but unused — left intact
            // for a future option 3 (full integration) and to honor the dual-pipeline rule's
            // interest in pre-computed state continuity.
            float chaoticOffset = chaoticFactor * baseFreq * 0.25f;
            for (int i = 0; i < chunkSamples; i++)
                modulation[i] += baseFreq + chaoticOffset + fractalVar[i];

            // Stage 4: Organic ADSR envelope — Fibonacci-timed attack/decay/sustain/release
            var envelope = pool.Envelope;
            OrganicAdsrEnvelope.ComputeChunk(envelope, chunkOffset, chunkSamples,
                adsrParams, simplexEnvelope, totalSamples);

            // Stage 5: Microtonal LFOs — PHI-modulated breathing amplitude variation.
            // Sample simplex ONCE per chunk at a slow time scale (mid-chunk position
            // scaled by 0.001, ~1000s simplex period) to derive a per-chunk phase
            // offset that shifts the LFO carrier's "where in cycle" position. Left
            // and right channels use different simplex y-offsets (0 and 1) for stereo
            // decorrelation while sharing the same slow rhythm shift across the session.
            // LfoDriftScale = 0 disables this drift entirely (legacy LFO behavior).
            float lfoDriftLeft = 0f;
            float lfoDriftRight = 0f;
            if (LfoDriftScale != 0f)
            {
                // Mid-chunk time scaled to a very slow simplex sample rate (~1000s period).
                // Single-sample input — float array allocation is trivially small.
                double midT = tChunk[chunkSamples / 2];
                float[] driftSampleSpan = [(float)(midT * 0.001)];
                // chaosVal shifts the simplex-field reading position by a session-level
                // chaotic offset, so two sessions take different paths through the same
                // slow drift field. The 1f decorrelation between Left and Right is
                // preserved on top of the chaos shift (Right = Left + 1f always).
                lfoDriftLeft = LfoDriftScale * simplexLfoDrift.GenerateNoise(driftSampleSpan, chaosVal)[0];
                lfoDriftRight = LfoDriftScale * simplexLfoDrift.GenerateNoise(driftSampleSpan, chaosVal + 1f)[0];
            }

            // Generate LFO arrays with the per-chunk phase drift baked in. Drift = 0
            // produces output identical to the legacy implementation, sample for sample.
            var lfoLeft = MicrotonalLfo.Compute(tChunk, lfoLeftParams, lfoDriftLeft);
            var lfoRight = MicrotonalLfo.Compute(tChunk, lfoRightParams, lfoDriftRight);

            // Stage 6: Harmonic generation — IDENTICAL machinery for every mode.
            // Both Taygetan and non-Taygetan modes use the same standard 13-voice
            // harmonic field (PHI exponents + 1.3 ratios + subharmonics), the same
            // natural decay 0.015/(f+1), the same envelope × LFO modulation, the
            // same Stage 2 fractal-variation phase modulation.
            //
            // The ONLY Taygetan-specific tweak: each of the first 9 voices gets
            // a slightly different frequency in L vs R (split by ±halfBeat) so
            // the brain perceives a 7.7 Hz binaural beat. The 4 subharmonic body
            // voices stay mono because their lowest entry (~27 Hz) is already
            // below normal hearing — splitting them binaurally would push the
            // R channel below audibility. Beat drifts slowly via simplex (~200s
            // period, ±0.5 Hz) so the entrainment is a living rhythm.
            //
            // fractalVar (from Stage 2) flows in as an optional sub-perceptual
            // phase-modulation source, scaled by PhaseModulationScale. Setting
            // PhaseModulationScale = 0 disables and produces the legacy sound.
            float[] waveLeft, waveRight;
            if (hasTaygetan)
            {
                // Taygetan: the first 9 voices form a binaural pair; subharmonics mono.
                // Beat = TAYGETAN_BEAT + simplex drift + ratio-driven bias from the
                // temporal schedule. The 7 sacred ratios express through the beat's
                // gentle evolution over the session — root holds 7.7 Hz, transcendentals
                // push slightly above, smootherstep blends between adjacent ratios.
                double midT = chunkSamples > 0 ? tChunk[chunkSamples / 2] : 0.0;
                float currentBeat = ComputeTaygetanCurrentBeat(
                    simplexTaygetanBeat, midT,
                    taygetanRatioSchedule, taygetanRatioValues!);
                float halfBeat = currentBeat * 0.5f;

                // The binaural is carried as a CONTINUOUS phase ramp, not a per-chunk
                // frequency split. The carriers stay at the un-split allFrequencies; the
                // generator adds ±(taygetanBeatPhaseHalf + halfBeatRad·localTime) to the
                // leading 9 voices. binauralRadPerSec = +2π·halfBeat for Left, −2π·halfBeat
                // for Right — the interaural phase difference IS the 7.7 Hz beat. Because
                // taygetanBeatPhaseHalf carries the accumulated phase from the previous
                // chunk, the carriers join seamlessly across the 3-second seam even though
                // halfBeat changed: only the slope (frequency) steps, never the phase.
                // (The old code folded ±halfBeat into the carrier frequency × absolute
                // time, which jumped by 2π·Δhalfbeat·t at every boundary — up to ~180°,
                // growing through the session. This removes that seam entirely.)
                double binauralRad = SacredConstants.TWO_PI_D * halfBeat;

                // Same generator call as standard modes, with the un-split field +
                // sineLeadingCount = TaygetanBinauralVoiceCount (9). The first 9 voices
                // stay pure sine because they're the binaural carriers — L/R difference
                // detection in the brain entrains cleanest when the carriers are pure
                // single-frequency tones. The remaining 4 subharmonic body voices get the
                // default Triangle shape, contributing the Tesla 3-6-9 + Pythagorean 5 +
                // Solfeggio crown harmonics inherent in the triangle waveform. They get no
                // binaural phase (they're past the leading count) so they stay mono.
                waveLeft = HarmonicGenerator.GenerateHarmonics(
                    tChunk, allFrequencies, envelope, lfoLeft, modDepths,
                    fractalVar, PhaseModulationScale,
                    waveShape: WaveShape.Triangle,
                    sineLeadingCount: TaygetanBinauralVoiceCount,
                    binauralStartPhase: taygetanBeatPhaseHalf,
                    binauralRadPerSec: binauralRad);
                waveRight = HarmonicGenerator.GenerateHarmonics(
                    tChunk, allFrequencies, envelope, lfoRight, modDepths,
                    fractalVar, PhaseModulationScale,
                    waveShape: WaveShape.Triangle,
                    sineLeadingCount: TaygetanBinauralVoiceCount,
                    binauralStartPhase: -taygetanBeatPhaseHalf,
                    binauralRadPerSec: -binauralRad);

                // Advance the accumulator by the half-beat phase accrued across THIS
                // chunk's span, so the next chunk's ramp begins exactly where this one
                // ends. Wrap into [0, 2π) to keep the double bounded over long sessions
                // (sin is 2π-periodic, so wrapping is exact, not an approximation).
                taygetanBeatPhaseHalf += binauralRad * (chunkSamples / (double)sampleRate);
                taygetanBeatPhaseHalf %= SacredConstants.TWO_PI_D;
                if (taygetanBeatPhaseHalf < 0) taygetanBeatPhaseHalf += SacredConstants.TWO_PI_D;
            }
            else
            {
                // Standard pipeline: same frequency array for both channels;
                // stereo decorrelation comes from different LFO arrays per channel.
                // All 13 voices triangle — the sacred form of the harmonic field
                // (Tesla 3-6-9 + Pythagorean 5 + Solfeggio crown live inside the
                // triangle waveform's odd harmonics at the Lemurian 432 Hz keynote).
                //
                // For Dimensional Journey mode (Mode 7), we ALSO apply per-dimension
                // amp scales — the 13-voice field's "spectral center of mass" journeys
                // upward as the session ascends through 1D → 9D. The CARRIER stays
                // anchored at 432 Hz (Lemurian keynote), but the FELT center moves
                // from subharmonic body voices (low dimensions) to upper PHI exponents
                // (crown band). For non-dimensional modes, dimAmpScales stays empty
                // (default) and the natural 0.015/(f+1) decay applies uniformly.
                ReadOnlySpan<float> dimAmpScales = default;
                if (dimensionalMode)
                {
                    // Look up the active dimension at this chunk's mid-time AND
                    // smootherstep-blend into the next dimension during the last
                    // 15% of each phase. This makes the carrier journey transitions
                    // gentle organic shifts instead of hard switches at boundaries.
                    // 9 equal-duration phases over the session — chunk-mid sampling
                    // picks the dimension cleanly within each phase, with the
                    // smootherstep blend handling boundaries.
                    double midSec = chunkSamples > 0 ? tChunk[chunkSamples / 2] : 0.0;
                    dimAmpScales = DimensionalJourney.ComputeAmpScalesAt(
                        (float)midSec, duration);
                }

                waveLeft = HarmonicGenerator.GenerateHarmonics(
                    tChunk, allFrequencies, envelope, lfoLeft, modDepths,
                    fractalVar, PhaseModulationScale,
                    ampScales: dimAmpScales,
                    waveShape: WaveShape.Triangle);
                waveRight = HarmonicGenerator.GenerateHarmonics(
                    tChunk, allFrequencies, envelope, lfoRight, modDepths,
                    fractalVar, PhaseModulationScale,
                    ampScales: dimAmpScales,
                    waveShape: WaveShape.Triangle);
            }

            // After Stage 6 — log peak/RMS so we can see what the harmonic field
            // alone produces. For Taygetan this is the FULL signal (binaural is
            // already inside Stage 6); for other modes Stage 7 used to add binaural
            // injection but Path 4+ removed that — Stage 6 is now the complete
            // tone-formation stage for every mode.
            if (detailedChunk)
            {
                DiagnosticLogger.LogSignalStats($"  ch{chunkIdx,4} after Stage 6 (harmonics) L", waveLeft);
                DiagnosticLogger.LogSignalStats($"  ch{chunkIdx,4} after Stage 6 (harmonics) R", waveRight);
            }

            // Stage 7: REMOVED — Taygetan binaural is now woven directly through
            // Stage 6 via separate L/R frequency arrays. No additional injection
            // needed. The Stage 7 log line that used to appear here is gone; the
            // signal flows directly from Stage 6 to Stage 8 (waveshape).

            // Stage 8: Tanh wave shaping — soft saturation for gentle harmonic warmth
            WaveShaper.Process(waveLeft, 2.5f);
            WaveShaper.Process(waveRight, 2.5f);

            // After Stage 8 — wave shaping creates intermodulation products from any
            // pair of close frequencies in the input. For Taygetan, this is where
            // close harmonic+binaural pairs collide and birth audible dissonance
            // (sum and difference frequencies of the close pair show up as new tones).
            if (detailedChunk)
            {
                DiagnosticLogger.LogSignalStats($"  ch{chunkIdx,4} after Stage 8 (waveshape) L", waveLeft);
                DiagnosticLogger.LogSignalStats($"  ch{chunkIdx,4} after Stage 8 (waveshape) R", waveRight);
            }

            // Stage 8b: PHI-fractal feedback — golden-ratio-delayed echoes for crystal-like decay
            waveLeft = phiFractalLeft.ProcessChunk(waveLeft);
            waveRight = phiFractalRight.ProcessChunk(waveRight);

            // Stage 9: Biquad lowpass filter — smooths harsh digital artifacts (stateful, carries zi)
            filterLeft.Process(waveLeft);
            filterRight.Process(waveRight);

            // Stage 10: Normalization — apply estimated gain to prevent clipping
            for (int i = 0; i < chunkSamples; i++)
            {
                waveLeft[i] *= normGain;
                waveRight[i] *= normGain;
            }

            // After Stage 10 — if peak is way above 1.0 here, normalization is failing
            // (estimate was too low), and downstream stages will get clipping. If peak
            // is way below 0.3, the signal is being over-attenuated.
            if (detailedChunk)
            {
                DiagnosticLogger.LogSignalStats($"  ch{chunkIdx,4} after Stage 10 (normalize) L", waveLeft);
                DiagnosticLogger.LogSignalStats($"  ch{chunkIdx,4} after Stage 10 (normalize) R", waveRight);
            }

            // Stage 11: FFT convolution reverb on left channel (stateful overlap-add)
            waveLeft = reverbLeft.ProcessChunk(waveLeft);

            // Stage 12: 12-sample right-channel delay + reverb (creates stereo width)
            var waveRightDelayed = pool.WaveRightDelayed;
            Array.Copy(delayBuffer, 0, waveRightDelayed, 0,
                global::System.Math.Min(12, chunkSamples));
            if (chunkSamples > 12)
                Array.Copy(waveRight, 0, waveRightDelayed, 12, chunkSamples - 12);
            Array.Copy(waveRight, global::System.Math.Max(0, chunkSamples - 12), delayBuffer, 0,
                global::System.Math.Min(12, chunkSamples));
            waveRight = reverbRight.ProcessChunk(waveRightDelayed);

            // Stage 13: Evolving noise layers — low-frequency modulated Gaussian noise.
            // Both channels share the session-level noiseOscFreq (drawn once in the
            // pre-loop phase) so the slow noise envelope is continuous across all
            // chunk boundaries instead of jumping at each 3-second seam. Right channel
            // gets its decorrelation from noiseOffsetRight (a small time offset),
            // not from a different oscillation rate.
            var noiseLeft = EvolvingNoiseLayer.Generate(tChunk, rng: _rng, frequency: noiseOscFreq);
            var tChunkOffset = pool.TChunkOffset;
            for (int i = 0; i < chunkSamples; i++)
                tChunkOffset[i] = tChunk[i] + noiseOffsetRight;  // double + float widens to double
            var noiseRight = EvolvingNoiseLayer.Generate(tChunkOffset, rng: _rng, frequency: noiseOscFreq);

            // Mix noise into the signal and apply per-channel noise scaling
            for (int i = 0; i < chunkSamples; i++)
            {
                waveLeft[i] = (waveLeft[i] + noiseLeft[i]) * noiseScaleLeft;
                waveRight[i] = (waveRight[i] + noiseRight[i]) * noiseScaleRight;
            }

            // Stage 14: Master fade in/out — t^1.5 curve at session start and end
            ApplyFade(waveLeft, waveRight, chunkOffset, chunkSamples, fadeSamples, totalSamples);

            // Stage 15: Toroidal panning — PHI-derived torus + Rössler chaos + simplex perturbation.
            // Sample simplexTorusDrift ONCE per chunk at the same slow rate as LFO drift
            // (mid-chunk × 0.001, ~1000s simplex period) to derive a phase offset for the
            // torus rotations. Phi drift is PHI-scaled relative to theta so the underlying
            // golden-ratio relationship between rotation rates is preserved as they evolve.
            // TorusDriftScale = 0 disables this drift entirely (legacy spatial behavior).
            double torusDriftTheta = 0.0;
            double torusDriftPhi = 0.0;
            if (TorusDriftScale != 0f)
            {
                // Mid-chunk time scaled to a very slow simplex sample rate (~1000s period).
                // Single-sample input — float array allocation is trivially small.
                double midT = tChunk[chunkSamples / 2];
                float[] driftSampleSpan = [(float)(midT * 0.001)];
                // Same chaosVal as LFO drift — keeps the per-session drift signature
                // unified across the breath rhythm and the spatial path. simplexTorusDrift
                // is a separate Simplex5D instance (seed 55) from simplexLfoDrift (seed 34),
                // so the two drifts read independent fields at the same chaos x-offset
                // — one organic per-session fingerprint expressed in two sister channels.
                float torusDriftBase = TorusDriftScale * simplexTorusDrift.GenerateNoise(driftSampleSpan, chaosVal)[0];
                torusDriftTheta = torusDriftBase;
                torusDriftPhi = torusDriftBase * SacredConstants.PHI;  // PHI-scaled to preserve ratio
            }

            ApplyToroidalPanning(waveLeft, waveRight, tChunk, chunkSamples,
                torusThetaFreq, torusPhiFreq, torusR, torusRSmall,
                simplexPan, rossler, panSmoother, driftFreq, driftAmplitude, pool,
                torusDriftTheta, torusDriftPhi);

            // Apply master volume (0.55 — the 10th Fibonacci number, sacred).
            // Held at this exact value by intent; do not retune.
            for (int i = 0; i < chunkSamples; i++)
            {
                waveLeft[i] *= MasterVolume;
                waveRight[i] *= MasterVolume;
            }

            // Final output stats (before sacred layers).
            // For Taygetan, comparing this to Stage 6 reveals the cumulative effect
            // of binaural injection + wave shaping + reverb + noise + panning + master
            // volume. A peak >> 1.0 here means clipping in the WAV output.
            if (detailedChunk)
            {
                DiagnosticLogger.LogSignalStats($"  ch{chunkIdx,4} pre-sacred-layer    L", waveLeft);
                DiagnosticLogger.LogSignalStats($"  ch{chunkIdx,4} pre-sacred-layer    R", waveRight);
            }

            // Stage 16: Sacred healing layers — 7 layers computed in parallel, each with
            // independent toroidal panning at golden angle phase offsets (except the 7th
            // Blue Ray Resonance Layer, which is locked to stereo center — zero point).
            if (duration > 60 || dimensionalMode)
            {
                // Launch all 7 sacred layers on the thread pool
                var sacredTasks = new Task<float[]>[sacredLayers.Length];
                for (int li = 0; li < sacredLayers.Length; li++)
                {
                    var layer = sacredLayers[li];
                    sacredTasks[li] = Task.Run(() => layer.ComputeChunk(tChunk, duration));
                }

                // Wait for all layers to complete (60-second timeout safety)
                try
                {
                    Task.WaitAll(sacredTasks, TimeSpan.FromSeconds(60));
                }
                catch (AggregateException ex)
                {
                    // A crashed sacred layer must never take the session down,
                    // but it must never vanish silently either — log each fault
                    // so a missing layer is diagnosable from the session log.
                    foreach (var inner in ex.Flatten().InnerExceptions)
                        DiagnosticLogger.Log(
                            $"Sacred layer task fault: {inner.GetType().Name}: {inner.Message}");
                }

                // For Dimensional Journey, compute per-sacred-layer emphasis at this
                // chunk's mid-time. Each dimension foregrounds the layer(s) most
                // aligned with its meaning (e.g., 1D → Water dominant, 5D → Merkaba
                // dominant, 9D → Blue Ray dominant), so the journey's character
                // shifts which layer leads as it ascends. Smootherstep crossfade at
                // dimension boundaries (last 15% of each phase blends into the next).
                // Non-dimensional modes: layerEmphasis stays null; multiplier = 1.0.
                float[]? dimLayerEmphasis = null;
                if (dimensionalMode)
                {
                    double midSec = chunkSamples > 0 ? tChunk[chunkSamples / 2] : 0.0;
                    dimLayerEmphasis = DimensionalJourney.ComputeLayerEmphasisAt(
                        (float)midSec, duration);
                }

                // Mix each sacred layer into the stereo field with independent toroidal panning.
                // Each layer's panning starts at a golden angle offset from the previous,
                // ensuring maximum spatial separation like sunflower seeds on a torus.
                // Placement uses the SAME constant-power (equal-power) law as the main
                // signal panner (see ApplyToroidalPanning in SoundGenerator.Effects.cs):
                // L = √2·cos(angle), R = √2·sin(angle), angle = (panScaled + 1)·π/4. Total
                // power per layer stays constant across pan position (no loudness pump as a
                // layer drifts around its torus) and center is unity gain (√2·cos(π/4) = 1),
                // so a centered layer — including Blue Ray, which is locked dead-center —
                // mixes in exactly as it did under the old linear law. Constants hoisted
                // out of both the layer loop and the per-sample loop.
                const float sacredQuarterPi = MathF.PI / 4f;
                float sacredSqrt2 = SacredConstants.SQRT_2;  // √2 — equal-power center anchor
                for (int li = 0; li < sacredLayers.Length; li++)
                {
                    float[]? layerData = sacredTasks[li].IsCompletedSuccessfully
                        ? sacredTasks[li].Result
                        : null;
                    if (layerData == null)
                    {
                        // Layer didn't complete (fault or 60s timeout) — skip it
                        // this chunk, but leave a trace so the absence is visible.
                        DiagnosticLogger.Log($"Sacred layer [{li}] skipped this chunk (not completed)");
                        continue;
                    }

                    float stFreq = sacredThetaFreqs[li];
                    float spFreq = sacredPhiFreqs[li];
                    float phaseOff = sacredPhaseOffsets[li];

                    // Apply dimensional layer emphasis (Mode 7 only). For other modes
                    // this stays at 1.0 (no change in layer amplitudes). Per dimension,
                    // some layers boost (>1.0) and some quiet (<1.0) — sum across the
                    // 7 layers stays roughly balanced so total energy doesn't spike.
                    float dimEmphasis = dimLayerEmphasis != null ? dimLayerEmphasis[li] : 1.0f;

                    for (int i = 0; i < chunkSamples; i++)
                    {
                        // Compute torus position with golden angle phase offset.
                        // Use double precision for the TWO_PI × freq × t multiplication
                        // to preserve phase accuracy over long sessions.
                        double sThetaD = SacredConstants.TWO_PI_D * stFreq * tChunk[i] + phaseOff;
                        double sPhiD = SacredConstants.TWO_PI_D * spFreq * tChunk[i] + phaseOff;

                        // Map torus position to stereo pan value
                        float sacredPan = (sacredR + sacredRSmall * (float)System.Math.Cos(sPhiD)) *
                            (float)System.Math.Cos(sThetaD) / (sacredR + sacredRSmall);
                        float panScaled = sacredPan * 0.4f;

                        // Mix into left and right channels with constant-power pan
                        // weighting, scaled by the per-dimension layer emphasis (1.0 for
                        // non-Mode-7). panScaled stays in ~[-0.4, 0.4] here (sacredPan ×
                        // 0.4), so angle stays within (0, π/2) and both gains stay positive.
                        float emphasized = layerData[i] * dimEmphasis;
                        float sacredAngle = (panScaled + 1.0f) * sacredQuarterPi;
                        waveLeft[i] += emphasized * sacredSqrt2 * MathF.Cos(sacredAngle);
                        waveRight[i] += emphasized * sacredSqrt2 * MathF.Sin(sacredAngle);
                    }
                }
            }

            // Build the final stereo output chunk (fresh allocation — yielded to consumer)
            var stereoChunk = new float[chunkSamples, 2];
            for (int i = 0; i < chunkSamples; i++)
            {
                stereoChunk[i, 0] = waveLeft[i];
                stereoChunk[i, 1] = waveRight[i];
            }

            // Final post-sacred-layer stats — what actually leaves the generator
            // and reaches the speaker. If we see clipping or unusual asymmetry
            // between L and R only here (and not earlier), the sacred layer
            // mixing is the dissonance source.
            if (detailedChunk)
            {
                DiagnosticLogger.LogSignalStats($"  ch{chunkIdx,4} FINAL (yielded)     L", waveLeft);
                DiagnosticLogger.LogSignalStats($"  ch{chunkIdx,4} FINAL (yielded)     R", waveRight);
            }

            // Report progress and yield the chunk to the consumer
            updateProgress?.Invoke((float)(chunkIdx + 1) / numChunks);
            yield return stereoChunk;
        }

        // End-of-session marker so log files are clearly bounded
        DiagnosticLogger.LogSection($"Session complete ({numChunks} chunks generated)");

        #endregion
    }

    #endregion

    // Batch mode wraps the streaming generator and collects all chunks
    // into a single float[totalSamples, 2] array for WAV file writing.
    // Uses the exact same pipeline — identical sound output.
    #region Batch Generator

    /// <summary>
    /// Batch mode: collect all streaming chunks into a single array.
    /// Used by SoundPlayer.SaveToWavAsync for WAV file generation.
    /// Output is identical to streaming — same pipeline, same sound.
    /// </summary>
    public float[,] GenerateBatch(float duration, float baseFreq,
        int sampleRate = 48000,
        CancellationToken ct = default,
        Action<float>? updateProgress = null,
        FrequencyMode freqMode = FrequencyMode.Standard)
    {
        // Double-precision sample count — mirrors GenerateStream exactly (dual
        // pipeline rule) so batch and streaming agree on the session's length.
        int totalSamples = (int)((double)sampleRate * duration);
        var result = new float[totalSamples, 2];
        int offset = 0;

        // Iterate over streaming chunks and copy each into the result array
        foreach (var chunk in GenerateStream(duration, baseFreq, sampleRate,
            ct: ct, updateProgress: updateProgress,
            freqMode: freqMode))
        {
            int samples = chunk.GetLength(0);
            int toCopy = global::System.Math.Min(samples, totalSamples - offset);
            for (int i = 0; i < toCopy; i++)
            {
                result[offset + i, 0] = chunk[i, 0];
                result[offset + i, 1] = chunk[i, 1];
            }
            offset += toCopy;
        }

        return result;
    }

    #endregion
}
