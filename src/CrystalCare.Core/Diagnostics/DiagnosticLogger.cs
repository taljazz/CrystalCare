using System.Text;

namespace CrystalCare.Core.Diagnostics;

/// <summary>
/// Lightweight file-backed diagnostic logger for tracking the audio pipeline's
/// internal state during a session — especially useful for diagnosing dissonance,
/// frequency conflicts, and amplitude anomalies in Taygetan or other binaural modes.
///
/// Default disabled — enable from the UI when you want to capture a session.
/// Output: %TEMP%/CrystalCare-diag-{timestamp}-{tag}.log
/// Thread-safe; safe to call from any thread (audio + UI).
///
/// Usage:
///     DiagnosticLogger.IsEnabled = true;
///     string path = DiagnosticLogger.Initialize("Taygetan-test");
///     DiagnosticLogger.Log("Hello");
///     DiagnosticLogger.LogArray("freqs", new[] { 1f, 2f, 3f });
///     DiagnosticLogger.LogPairs("pairs", binauralPairs);
///     DiagnosticLogger.LogProximityMatrix(...);
///     DiagnosticLogger.LogSignalStats("after Stage 6", waveSpan);
/// </summary>
public static class DiagnosticLogger
{
    // Internal state: a single lock guards file I/O across threads, and a
    // string-builder is reused for compact array formatting to avoid GC pressure.
    // Logging is fail-soft — IO exceptions are swallowed so audio never crashes.
    #region Internal State

    // Lock for file appends — multiple threads (audio, UI) may log concurrently
    private static readonly object _lock = new();

    // Reusable StringBuilder for array formatting (avoids per-call allocation)
    [ThreadStatic]
    private static StringBuilder? _scratch;

    // Path to the current log file (null = not yet initialized or disabled)
    private static string? _logPath;

    // Master enable flag — when false, every log method short-circuits to no-op
    private static bool _isEnabled;

    #endregion

    // Public API: enable/disable and the current log path. Setting IsEnabled to
    // false makes every Log call a cheap no-op, so leaving the hooks in production
    // code is safe — they cost nothing when the flag is off.
    #region Public Properties

    /// <summary>
    /// Master switch. When false, every Log/LogArray/LogPairs/etc. call returns
    /// immediately without touching disk or formatting strings. Default: false.
    /// </summary>
    public static bool IsEnabled
    {
        get => _isEnabled;
        set => _isEnabled = value;
    }

    /// <summary>Absolute path to the current log file, or null if disabled / not initialized.</summary>
    public static string? LogPath => _logPath;

    #endregion

    // Initialization: opens a fresh timestamped log file in the user's temp
    // directory and writes a header with session metadata. Returns the path so
    // the UI can surface it. Subsequent log calls append to this file.
    #region Initialization

    /// <summary>
    /// Open a fresh log file and write the header. Call once per session before
    /// the audio pipeline starts. Returns the absolute path to the new log file
    /// (or empty string if logging is disabled / IO failed).
    /// </summary>
    /// <param name="sessionTag">
    /// Short label identifying this session (e.g., "Taygetan", "Solfeggio").
    /// Sanitized into the filename so unsafe characters don't break paths.
    /// </param>
    public static string Initialize(string sessionTag)
    {
        // Bail out fast if logging is disabled — no file is created
        if (!_isEnabled)
            return string.Empty;

        // Build a timestamped filename; sanitize the tag so it's filesystem-safe
        string timestamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
        string safeTag = string.IsNullOrEmpty(sessionTag) ? "session" : Sanitize(sessionTag);
        _logPath = Path.Combine(
            Path.GetTempPath(),
            $"CrystalCare-diag-{timestamp}-{safeTag}.log");

        // Write the header so the file exists immediately even before any logging
        lock (_lock)
        {
            try
            {
                File.WriteAllText(_logPath,
                    $"=== CrystalCare Diagnostic Log ==={Environment.NewLine}" +
                    $"Session tag: {sessionTag}{Environment.NewLine}" +
                    $"Started:     {DateTime.Now:O}{Environment.NewLine}" +
                    $"Process:     {Environment.ProcessId}{Environment.NewLine}" +
                    $"Machine:     {Environment.MachineName}{Environment.NewLine}" +
                    Environment.NewLine);
            }
            catch
            {
                // IO never crashes audio; drop the path on error
                _logPath = null;
            }
        }

        return _logPath ?? string.Empty;
    }

    #endregion

    // Core logging primitives: a single line, a section divider, and array-shaped
    // helpers. Every method short-circuits when IsEnabled is false so log call
    // sites can stay in production code with negligible overhead.
    #region Core Logging Methods

    /// <summary>Append a single line with a millisecond-precision timestamp.</summary>
    public static void Log(string message)
    {
        if (!_isEnabled || _logPath == null) return;
        lock (_lock)
        {
            try
            {
                // Format: [HH:mm:ss.fff] message
                File.AppendAllText(_logPath,
                    $"[{DateTime.Now:HH:mm:ss.fff}] {message}{Environment.NewLine}");
            }
            catch { /* IO errors must never crash audio */ }
        }
    }

    /// <summary>Append a section divider — useful for visually separating stages.</summary>
    public static void LogSection(string title)
    {
        if (!_isEnabled) return;
        Log("");
        Log($"--- {title} ---");
    }

    /// <summary>Log a float array compactly. Truncates to maxItems if longer.</summary>
    public static void LogArray(string name, float[] arr, int maxItems = 20)
    {
        if (!_isEnabled || arr == null) return;

        // Use the thread-static scratch StringBuilder to avoid per-call allocations
        var sb = _scratch ??= new StringBuilder(256);
        sb.Clear();
        sb.Append(name).Append('[').Append(arr.Length).Append("] = [");

        if (arr.Length <= maxItems)
        {
            // Short array — write all values
            for (int i = 0; i < arr.Length; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append(arr[i].ToString("F4"));
            }
        }
        else
        {
            // Long array — write first half/last half with elision marker
            int half = maxItems / 2;
            for (int i = 0; i < half; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append(arr[i].ToString("F4"));
            }
            sb.Append(", ... (").Append(arr.Length - maxItems).Append(" elided), ..., ");
            for (int i = arr.Length - half; i < arr.Length; i++)
            {
                if (i > arr.Length - half) sb.Append(", ");
                sb.Append(arr[i].ToString("F4"));
            }
        }

        sb.Append(']');
        Log(sb.ToString());
    }

    /// <summary>Log a binaural-pair array: each pair as L/R with computed center + beat.</summary>
    public static void LogPairs(string name, (float Left, float Right)[] pairs)
    {
        if (!_isEnabled || pairs == null) return;
        Log($"{name}[{pairs.Length}]:");
        for (int i = 0; i < pairs.Length; i++)
        {
            // Beat = L - R (the perceptual difference frequency)
            // Center = midpoint between L and R (the "carrier" ear-fused frequency)
            float beat = pairs[i].Left - pairs[i].Right;
            float center = (pairs[i].Left + pairs[i].Right) / 2f;
            Log($"  [{i}] L={pairs[i].Left,10:F3}Hz  R={pairs[i].Right,10:F3}Hz  center={center,10:F3}Hz  beat={beat,7:F3}Hz");
        }
    }

    /// <summary>Compute and log peak/RMS of a signal slice — useful for stage-by-stage tracking.</summary>
    public static void LogSignalStats(string label, ReadOnlySpan<float> signal)
    {
        if (!_isEnabled) return;

        // Single-pass scan for min/max/sumSq; cheap (~144k ops for a 3s chunk at 48kHz)
        float min = float.MaxValue;
        float max = float.MinValue;
        double sumSq = 0;
        for (int i = 0; i < signal.Length; i++)
        {
            float v = signal[i];
            if (v < min) min = v;
            if (v > max) max = v;
            sumSq += (double)v * v;
        }
        // Peak amplitude = max(|min|, |max|); a crude clipping indicator
        float peak = MathF.Max(MathF.Abs(min), MathF.Abs(max));
        // RMS = sqrt(mean of squares); more representative of perceived loudness.
        // Use System.Math explicitly — bare `Math` resolves to the sibling
        // CrystalCare.Core.Math namespace from this file, which has no Sqrt.
        float rms = signal.Length > 0 ? (float)System.Math.Sqrt(sumSq / signal.Length) : 0f;
        Log($"{label}: peak={peak:F4}, rms={rms:F4}, min={min:F4}, max={max:F4}");
    }

    #endregion

    // Specialized helpers for the kinds of analyses we need to do on Taygetan
    // dissonance: cross-checking each generated harmonic against every binaural
    // carrier, flagging close-frequency proximities that produce audible beats.
    #region Specialized Diagnostics

    /// <summary>
    /// For Taygetan-style binaural diagnosis: report harmonic-vs-binaural-carrier
    /// proximities. Each generated harmonic is matched to its closest binaural
    /// carrier (left or right side); any gap below warnGapHz flags as dissonant.
    ///
    /// Reading the output: "harm[5]=752.3Hz closest pair[2].L gap=2.7Hz !" means
    /// harmonic 5 sits 2.7 Hz from a binaural carrier — they will produce an
    /// audible 2.7 Hz beat. Multiple flags = thick dissonance.
    /// </summary>
    public static void LogProximityMatrix(string label,
        float[] harmonics, (float Left, float Right)[] binauralPairs, float warnGapHz = 10f)
    {
        if (!_isEnabled || harmonics == null || binauralPairs == null) return;
        Log($"{label} — closest binaural carrier per harmonic (gap < {warnGapHz}Hz flagged !):");

        // For each harmonic, scan all binaural carriers (left + right of each pair)
        // and report the closest. Mark with '!' if it's within warnGapHz — that's
        // the audible-beat zone where dissonance is born.
        for (int h = 0; h < harmonics.Length; h++)
        {
            float harm = harmonics[h];
            float closestGap = float.MaxValue;
            int closestIdx = -1;
            string closestSide = "?";

            for (int p = 0; p < binauralPairs.Length; p++)
            {
                // Compare against both ears of this binaural pair
                float gL = MathF.Abs(harm - binauralPairs[p].Left);
                float gR = MathF.Abs(harm - binauralPairs[p].Right);
                if (gL < closestGap) { closestGap = gL; closestIdx = p; closestSide = "L"; }
                if (gR < closestGap) { closestGap = gR; closestIdx = p; closestSide = "R"; }
            }

            // Flag near-misses (likely dissonance source) with '!', otherwise space
            string flag = closestGap < warnGapHz ? "!" : " ";
            Log($"  {flag} harm[{h}]={harm,10:F3}Hz   closest pair[{closestIdx}].{closestSide}, gap={closestGap,8:F3}Hz");
        }
    }

    #endregion

    // Filesystem helpers — keeping the tag clean so it works as a filename
    // even when the user enters arbitrary text.
    #region Helpers

    /// <summary>Strip filesystem-incompatible characters from a tag string.</summary>
    private static string Sanitize(string raw)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var safe = new StringBuilder(raw.Length);
        foreach (char c in raw)
        {
            // Replace invalid chars and spaces with underscore for clean filenames
            safe.Append(invalid.Contains(c) || c == ' ' ? '_' : c);
        }
        return safe.ToString();
    }

    #endregion
}
