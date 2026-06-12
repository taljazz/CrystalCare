using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// PHI-spaced fractal micro-echoes for self-similar harmonic enrichment.
/// Adds golden-ratio-delayed copies at decreasing amplitudes,
/// creating organic overtone structure without altering the fundamental.
///
/// Streaming-safe: carries a tail buffer between chunks to prevent pops
/// at chunk boundaries. The tail holds samples that "spill over" from
/// delayed echoes and gets added to the start of the next chunk.
///
/// Port of AudioProcessor.phi_fractal_feedback() from SoundGenerator.py.
/// </summary>
public sealed class PhiFractalFeedback
{
    // Base delay = sampleRate / PHI (~29,708 samples at 48kHz).
    // 3 echo depths at PHI-spaced intervals with exponentially decreasing amplitude.
    // Tail buffer carries echo spillover between chunks for seamless continuity.
    // Work and tail buffers are reused across chunks (cleared, not reallocated) —
    // the previous per-chunk allocations were ~1 MB of GC churn per 3-second chunk.
    #region Fields and Constructor

    private readonly int _baseDelay;
    private readonly int _depth;
    private readonly float _factor;
    private float[] _tail;      // Carried between chunks
    private float[] _tailSwap;  // Scratch for building next tail (ping-pongs with _tail)
    private float[]? _work;     // Reusable working buffer (lazily sized to n + tailLen)

    // Default echo factor = 1/21 (Fibonacci reciprocal, ≈0.0476)
    public PhiFractalFeedback(int sampleRate = 48000, int depth = 3,
        float factor = SacredConstants.FRACTAL_ECHO_FACTOR)
    {
        _baseDelay = (int)(sampleRate / SacredConstants.PHI); // ~29,708 samples
        _depth = depth;
        _factor = factor;

        // Tail length = maximum delay = baseDelay * depth
        int maxDelay = _baseDelay * _depth;
        _tail = new float[maxDelay];
        _tailSwap = new float[maxDelay];
    }

    #endregion

    // Streaming mode — processes one chunk with echo tail carry between calls.
    // Adds previous tail to start of output, generates new echoes, saves new tail.
    #region Streaming Chunk Processing

    /// <summary>
    /// Process a chunk with PHI-fractal feedback, carrying echo tail between chunks.
    /// </summary>
    public float[] ProcessChunk(ReadOnlySpan<float> signal)
    {
        int n = signal.Length;
        int tailLen = _tail.Length;
        int workLen = n + tailLen;

        // Working buffer: signal + room for echo spillover. Reused across chunks —
        // lazily allocated on first call (chunks after the first are the same
        // length or shorter, so the buffer never needs to grow mid-session).
        if (_work == null || _work.Length < workLen)
            _work = new float[workLen];
        var work = _work;
        Array.Clear(work, 0, workLen);
        signal.CopyTo(work);

        // Add previous tail (echoes from last chunk)
        int addLen = global::System.Math.Min(tailLen, n);
        for (int i = 0; i < addLen; i++)
            work[i] += _tail[i];

        // If tail is longer than chunk, carry the remainder
        // (unlikely with 3s chunks and ~1.86s max delay, but safe)
        if (tailLen > n)
        {
            for (int i = n; i < tailLen; i++)
                work[i] += _tail[i];
        }

        // Add delayed echoes
        for (int d = 0; d < _depth; d++)
        {
            int delay = _baseDelay * (d + 1);
            float amplitude = MathF.Pow(_factor, d + 1);

            for (int i = delay; i < workLen; i++)
            {
                int srcIdx = i - delay;
                if (srcIdx < n) // Only echo from the current signal, not from tail echoes
                    work[i] += signal[srcIdx] * amplitude;
            }
        }

        // Extract output (first n samples — fresh array, returned to the caller)
        var result = new float[n];
        Array.Copy(work, result, n);

        // Save new tail (samples beyond chunk length) — build into the swap
        // buffer then ping-pong it with _tail, so no per-chunk allocation.
        var newTail = _tailSwap;
        Array.Clear(newTail, 0, tailLen);
        int copyLen = global::System.Math.Min(tailLen, workLen - n);
        if (copyLen > 0)
            Array.Copy(work, n, newTail, 0, copyLen);
        _tailSwap = _tail;
        _tail = newTail;

        return result;
    }

    #endregion

    // Batch mode (stateless) — processes the full signal in one pass.
    // No tail carry needed since there are no chunk boundaries.
    #region Batch Processing (Stateless)

    // Default echo factor = 1/21 (Fibonacci reciprocal)
    public static float[] Process(ReadOnlySpan<float> signal, int sampleRate = 48000,
        int depth = 3, float factor = SacredConstants.FRACTAL_ECHO_FACTOR)
    {
        var result = new float[signal.Length];
        signal.CopyTo(result);

        int baseDelay = (int)(sampleRate / SacredConstants.PHI);

        for (int d = 0; d < depth; d++)
        {
            int delay = baseDelay * (d + 1);
            if (delay >= signal.Length) break;

            float amplitude = MathF.Pow(factor, d + 1);

            for (int i = delay; i < signal.Length; i++)
                result[i] += signal[i - delay] * amplitude;
        }

        return result;
    }

    #endregion
}
