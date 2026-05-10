using CrystalCare.Core.Diagnostics;
using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Noise;

namespace CrystalCare.Core.Generation;

/// <summary>
/// SoundGenerator Taygetan helpers — final architecture.
///
/// Taygetan uses the EXACT same Stage 6 machinery and EXACT same 13-voice
/// harmonic field that every other mode uses (6 PHI exponents + 3 ratio-1.3 +
/// 4 subharmonics with natural 0.015/(f+1) decay). The Taygetan-specific
/// signal additions are:
///
///   1. baseFreq is picked randomly from TAYGETAN_BASE_FREQS [432, 528, 963]
///      — three documented healing carriers (Lemurian keynote, Taygetan
///      med-pod frequency, Solfeggio crown). Each session opens differently.
///
///   2. The first 9 voices get split L/R by ±halfBeat for binaural sync.
///      The 4 subharmonic body voices stay mono (their bottom voice is near
///      27 Hz; splitting them would push R below audibility).
///
///   3. The half-beat is TAYGETAN_BEAT/2 = 3.85 Hz by default, plus simplex
///      drift (~200s period, ±0.25 Hz on the half) AND a slow ratio-driven
///      bias from a per-session schedule that walks through the 7 sacred
///      Taygetan ratios. Each ratio subtly shifts the beat over its window —
///      the 7 ratios become a temporal signature expressed through the beat's
///      evolution rather than separate audible voices.
///
/// No per-voice amp emphasis on the harmonic field, no extra breath modulation —
/// the standard LFO already provides breath; the natural harmonic decay already
/// shapes timbre. Taygetan's tone formation = "her standard tone formation,
/// plus binaural with a beat that walks through 7 sacred ratios."
/// </summary>
public sealed partial class SoundGenerator
{
    // Tone-formation tweaks specific to Taygetan. Kept here (not in
    // SacredConstants) because they describe internal pipeline structure
    // rather than ear-tunable sacred values.
    #region Tone-Formation Constants

    /// <summary>
    /// How many of the 13 voices receive the binaural L/R split. The first 9
    /// (the harmonic ladder + 1.3 ratios) carry the binaural sync; the last 4
    /// (subharmonics) stay mono as body/grounding voices.
    /// </summary>
    private const int TaygetanBinauralVoiceCount = 9;

    /// <summary>
    /// Crossfade fraction between adjacent ratio windows in the temporal
    /// schedule. 0.2 = 20% of each window fades into the next via Perlin
    /// smootherstep. Keeps the beat shifts smooth instead of step changes.
    /// </summary>
    private const float TaygetanRatioCrossfadeFraction = 0.2f;

    #endregion

    // Schedule type — one window per Taygetan ratio in the temporal beat-bias
    // sequence. The session walks through each ratio in turn; during each
    // window the binaural beat is biased by an amount derived from that ratio.
    #region Schedule Types

    /// <summary>
    /// One window in the Taygetan ratio temporal schedule. The named ratio is
    /// active throughout [StartTime..EndTime), with a Perlin smootherstep
    /// crossfade into the next window across the final TaygetanRatioCrossfadeFraction
    /// of the window's duration.
    /// </summary>
    private readonly record struct TaygetanRatioWindow(
        int RatioIndex,    // index into the ratioValues array (0..6)
        float StartTime,   // window start (seconds)
        float EndTime);    // window end (seconds)

    #endregion

    // Schedule construction — adaptive count by duration, root opens, rest
    // shuffled. Mirrors the Crystalline-layer "journey through sacred forms"
    // pattern but ONLY drives the beat bias (NOT per-voice harmonic emphasis).
    #region Schedule Builder

    /// <summary>
    /// Build the per-session Taygetan ratio temporal schedule. Adaptive count:
    /// short sessions get 2 windows (root + one other), longer sessions get
    /// progressively more, up to all 7. Root always opens (the keynote anchor);
    /// remaining ratios shuffled randomly so each session walks a different
    /// path through the Taygetan transmission.
    /// </summary>
    private static List<TaygetanRatioWindow> BuildTaygetanRatioSchedule(
        int numRatios, float totalDuration, Random rng)
    {
        // Adaptive window count by duration — short sessions concentrate on
        // fewer ratios so each one has time to express; long sessions visit
        // the full sacred set. Same scheme as CrystallineResonanceLayer.
        int activeCount;
        if (totalDuration < 120) activeCount = 2;
        else if (totalDuration < 300) activeCount = 3;
        else if (totalDuration < 900) activeCount = 5;
        else activeCount = numRatios;  // up to all 7

        activeCount = global::System.Math.Min(activeCount, numRatios);

        // Root (index 0) always opens — every Taygetan session begins at the
        // keynote. Remaining ratios shuffled randomly for per-session variety.
        var rest = Enumerable.Range(1, numRatios - 1).OrderBy(_ => rng.Next()).ToArray();
        var sequence = new int[activeCount];
        sequence[0] = 0;
        for (int i = 1; i < activeCount; i++)
            sequence[i] = rest[(i - 1) % rest.Length];

        // Equal-duration windows. We don't golden-angle-perturb the boundaries
        // here because the bias they produce is already sub-perceptual — a
        // perturbed schedule would only add complexity without adding audible
        // organic motion (the simplex beat drift already provides that).
        float winDur = totalDuration / activeCount;
        var schedule = new List<TaygetanRatioWindow>(activeCount);
        for (int i = 0; i < activeCount; i++)
        {
            schedule.Add(new TaygetanRatioWindow(
                RatioIndex: sequence[i],
                StartTime: i * winDur,
                EndTime: (i == activeCount - 1) ? totalDuration : (i + 1) * winDur));
        }
        return schedule;
    }

    #endregion

    // Per-chunk computations — drifted beat with ratio bias folded in. The
    // ratio bias smoothly transitions between adjacent windows via Perlin
    // smootherstep so the beat shifts feel organic rather than stepped.
    #region Per-Chunk Computations

    /// <summary>
    /// Compute the additive Hz bias from the active Taygetan ratio at this
    /// time position. Smootherstep crossfade between adjacent windows so the
    /// beat shift is smooth. Returns Hz value to ADD to the base beat.
    /// </summary>
    /// <param name="schedule">The per-session ratio schedule.</param>
    /// <param name="time">Time in seconds into the session.</param>
    /// <param name="ratioValues">The 7 Taygetan ratio values (indexed by RatioIndex).</param>
    private static float ComputeTaygetanRatioBias(
        List<TaygetanRatioWindow> schedule, float time, float[] ratioValues)
    {
        if (schedule.Count == 0) return 0f;

        // Local helper: convert a ratio value to its Hz bias.
        // Multiplicative form: bias = TAYGETAN_BEAT × (ratio - 1) × scale.
        // Root (ratio = 1) → 0 bias (beat stays at base 7.7 Hz).
        // Higher ratios push the beat slightly higher within alpha-theta band.
        float biasFor(int ratioIdx) =>
            SacredConstants.TAYGETAN_BEAT *
            (ratioValues[ratioIdx] - 1f) *
            SacredConstants.TAYGETAN_RATIO_BIAS_SCALE;

        // Find the active window for this time
        int activeIdx = -1;
        for (int i = 0; i < schedule.Count; i++)
        {
            if (time >= schedule[i].StartTime && time < schedule[i].EndTime)
            {
                activeIdx = i;
                break;
            }
        }

        // Before the schedule starts: use the first window's bias
        if (activeIdx < 0)
        {
            // After end (rare — happens at session boundary): use last window's bias
            if (time >= schedule[^1].EndTime) return biasFor(schedule[^1].RatioIndex);
            // Before start: use first window's bias
            return biasFor(schedule[0].RatioIndex);
        }

        var window = schedule[activeIdx];
        float windowDur = window.EndTime - window.StartTime;
        float crossfade = windowDur * TaygetanRatioCrossfadeFraction;
        float intoWindow = time - window.StartTime;

        // Entry crossfade: if we're in the first 20% of this window AND there's
        // a previous window, blend smoothly from previous bias to this bias.
        if (activeIdx > 0 && intoWindow < crossfade && crossfade > 0f)
        {
            float t = global::System.Math.Clamp(intoWindow / crossfade, 0f, 1f);
            // Perlin smootherstep: 6t^5 - 15t^4 + 10t^3
            float smooth = t * t * t * (t * (t * 6f - 15f) + 10f);
            float prevBias = biasFor(schedule[activeIdx - 1].RatioIndex);
            float thisBias = biasFor(window.RatioIndex);
            return prevBias + (thisBias - prevBias) * smooth;
        }

        // Solo phase (or first window with no entry crossfade)
        return biasFor(window.RatioIndex);
    }

    /// <summary>
    /// Sample the slow simplex source AND fold in the active ratio bias to
    /// derive this chunk's drifted beat frequency. Returns a value near
    /// TAYGETAN_BEAT (7.7 Hz), with simplex drift (±0.5 Hz, ~200s period)
    /// and ratio-driven bias (sub-Hz, schedule-driven). The brain entrainment
    /// becomes a living rhythm shaped by the sacred ratios.
    /// </summary>
    private static float ComputeTaygetanCurrentBeat(
        Simplex5D simplexBeatDrift, double chunkMidTime,
        List<TaygetanRatioWindow>? ratioSchedule, float[] ratioValues)
    {
        // Simplex drift — single-sample read at slow time scale (~200s period)
        float[] beatDriftSpan = [(float)(chunkMidTime * SacredConstants.TAYGETAN_BEAT_DRIFT_TIME_SCALE)];
        float beatDriftHz = SacredConstants.TAYGETAN_BEAT_DRIFT_AMP *
            simplexBeatDrift.GenerateNoise(beatDriftSpan, 0f)[0];

        // Ratio bias — schedule-driven, smootherstep between windows.
        // Root holds beat at base; transcendentals push it slightly higher.
        float ratioBias = 0f;
        if (ratioSchedule != null)
        {
            ratioBias = ComputeTaygetanRatioBias(
                ratioSchedule, (float)chunkMidTime, ratioValues);
        }

        return SacredConstants.TAYGETAN_BEAT + beatDriftHz + ratioBias;
    }

    /// <summary>
    /// Build separate L and R frequency arrays for binaural separation.
    /// First TaygetanBinauralVoiceCount entries (harmonic ladder + 1.3 ratios)
    /// get split by ±halfBeat. Remaining entries (subharmonics) stay mono
    /// because their bottom voice is too low for binaural perception.
    /// </summary>
    private static (float[] freqsL, float[] freqsR) ComputeTaygetanBinauralFreqs(
        float[] allFrequencies, float halfBeat)
    {
        var freqsL = new float[allFrequencies.Length];
        var freqsR = new float[allFrequencies.Length];

        // First 9 voices — split L/R for binaural beat at the current drifted+biased rate
        int splitCount = global::System.Math.Min(TaygetanBinauralVoiceCount, allFrequencies.Length);
        for (int i = 0; i < splitCount; i++)
        {
            freqsL[i] = allFrequencies[i] + halfBeat;
            freqsR[i] = allFrequencies[i] - halfBeat;
        }

        // Subharmonics stay mono — same value in both channels. No binaural
        // beat in the body voices; they ground the sound.
        for (int i = splitCount; i < allFrequencies.Length; i++)
        {
            freqsL[i] = allFrequencies[i];
            freqsR[i] = allFrequencies[i];
        }

        return (freqsL, freqsR);
    }

    #endregion

    // Diagnostic helper — logs the Taygetan signature including the temporal
    // ratio schedule so the user can verify the beat journey design.
    #region Diagnostic Logging

    /// <summary>
    /// Log a short Taygetan summary — base frequency, beat parameters, ratio
    /// schedule, and which voices are binaural-split vs mono. The harmonic
    /// field itself is logged separately (Taygetan uses the same 13-voice set
    /// as other modes); this section adds only the Taygetan-unique details.
    /// </summary>
    private static void LogTaygetanSummary(
        float[] allFrequencies,
        List<TaygetanRatioWindow>? ratioSchedule,
        string[] ratioNames,
        float[] ratioValues)
    {
        if (!DiagnosticLogger.IsEnabled) return;

        DiagnosticLogger.LogSection("Taygetan signature");
        DiagnosticLogger.Log($"Base candidates: 432 / 528 / 963 Hz (random per session)");
        DiagnosticLogger.Log($"Architecture: standard 13-voice field + L/R binaural split on first {TaygetanBinauralVoiceCount} voices");
        DiagnosticLogger.Log($"Beat base    = {SacredConstants.TAYGETAN_BEAT:F3} Hz");
        DiagnosticLogger.Log($"Beat drift   = ±{SacredConstants.TAYGETAN_BEAT_DRIFT_AMP:F2} Hz, ~{1.0 / SacredConstants.TAYGETAN_BEAT_DRIFT_TIME_SCALE:F0}s simplex period");
        DiagnosticLogger.Log($"Ratio bias   = TAYGETAN_BEAT × (ratio-1) × {SacredConstants.TAYGETAN_RATIO_BIAS_SCALE:F3}  (sub-Hz, schedule-driven)");

        // Show binaural voices at base beat (no drift, no ratio bias)
        float halfBeat = SacredConstants.TAYGETAN_BEAT * 0.5f;
        DiagnosticLogger.Log("Binaural voices (at base beat 7.7 Hz, no drift, no ratio bias):");
        for (int i = 0; i < global::System.Math.Min(TaygetanBinauralVoiceCount, allFrequencies.Length); i++)
        {
            float l = allFrequencies[i] + halfBeat;
            float r = allFrequencies[i] - halfBeat;
            DiagnosticLogger.Log(
                $"  voice[{i,2}]  base={allFrequencies[i],10:F3}Hz   L={l,10:F3}Hz  R={r,10:F3}Hz  beat={SacredConstants.TAYGETAN_BEAT:F2}Hz");
        }
        DiagnosticLogger.Log("Mono body voices (subharmonics — no binaural):");
        for (int i = TaygetanBinauralVoiceCount; i < allFrequencies.Length; i++)
        {
            DiagnosticLogger.Log(
                $"  voice[{i,2}]  base={allFrequencies[i],10:F3}Hz   (mono)");
        }

        // Show the temporal ratio schedule — the journey through the 7 sacred
        // ratios that biases the beat over the session.
        if (ratioSchedule != null && ratioSchedule.Count > 0)
        {
            DiagnosticLogger.LogSection($"Taygetan ratio temporal schedule ({ratioSchedule.Count} windows)");
            DiagnosticLogger.Log("The 7 sacred ratios are expressed via the beat's evolution.");
            DiagnosticLogger.Log("Each window biases the binaural beat by a small Hz amount derived from its ratio.");
            for (int i = 0; i < ratioSchedule.Count; i++)
            {
                var win = ratioSchedule[i];
                string name = win.RatioIndex < ratioNames.Length ? ratioNames[win.RatioIndex] : $"ratio[{win.RatioIndex}]";
                float ratio = win.RatioIndex < ratioValues.Length ? ratioValues[win.RatioIndex] : 1f;
                float biasHz = SacredConstants.TAYGETAN_BEAT * (ratio - 1f) * SacredConstants.TAYGETAN_RATIO_BIAS_SCALE;
                float effectiveBeat = SacredConstants.TAYGETAN_BEAT + biasHz;
                DiagnosticLogger.Log(
                    $"  [{i}] {name,-14}  ratio={ratio:F4}  {win.StartTime,7:F1}s..{win.EndTime,7:F1}s  bias={biasHz:+0.000;-0.000}Hz  beat={effectiveBeat:F3}Hz");
            }
        }
    }

    #endregion
}
