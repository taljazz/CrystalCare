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
    #region Fields and Constructor

    private readonly int _baseDelay;
    private readonly int _depth;
    private readonly float _factor;
    private float[] _tail; // Carried between chunks

    public PhiFractalFeedback(int sampleRate = 48000, int depth = 3, float factor = 0.05f)
    {
        _baseDelay = (int)(sampleRate / SacredConstants.PHI); // ~29,708 samples
        _depth = depth;
        _factor = factor;

        // Tail length = maximum delay = baseDelay * depth
        int maxDelay = _baseDelay * _depth;
        _tail = new float[maxDelay];
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

        // Working buffer: signal + room for echo spillover
        var work = new float[n + tailLen];
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

            for (int i = delay; i < n + tailLen; i++)
            {
                int srcIdx = i - delay;
                if (srcIdx < n) // Only echo from the current signal, not from tail echoes
                    work[i] += signal[srcIdx] * amplitude;
            }
        }

        // Extract output (first n samples)
        var result = new float[n];
        Array.Copy(work, result, n);

        // Save new tail (samples beyond chunk length)
        _tail = new float[tailLen];
        int copyLen = global::System.Math.Min(tailLen, work.Length - n);
        if (copyLen > 0)
            Array.Copy(work, n, _tail, 0, copyLen);

        return result;
    }

    #endregion

    // Batch mode (stateless) — processes the full signal in one pass.
    // No tail carry needed since there are no chunk boundaries.
    #region Batch Processing (Stateless)

    public static float[] Process(ReadOnlySpan<float> signal, int sampleRate = 48000,
        int depth = 3, float factor = 0.05f)
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
