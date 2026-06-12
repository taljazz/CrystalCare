using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Math;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Streaming overlap-add FFT convolution reverb.
/// Maintains tail buffer between chunks for seamless reverb continuation.
///
/// IR: exponential decay (0.75 rate) with PHI-frequency sinusoidal modulation.
/// Length: 2.618 seconds at sample rate (golden ratio duration).
///
/// Port of StreamingReverb class from SoundGenerator.py.
/// </summary>
public sealed class StreamingReverb
{
    // Impulse response (IR) and overlap tail buffer.
    // IR: 2.618 seconds (PHI² duration) with exponential decay + PHI sinusoidal modulation.
    // Tail buffer carries reverb spillover between chunks for seamless continuity.
    // The IR's forward FFT is cached after the first chunk (the IR never changes
    // during a session) — re-computing it per chunk was a third of all reverb CPU.
    #region Fields and Constructor

    private readonly float[] _ir;        // Pre-computed impulse response
    private float[] _tail;               // Overlap tail carried between chunks

    // Cached forward FFT of the IR, built lazily on the first ProcessChunk call
    // (sized from the first chunk's length — the largest the stream produces;
    // later chunks are the same length or shorter, never longer). Rebuilt only
    // if a larger chunk ever arrives.
    private System.Numerics.Complex[]? _kernelFft;

    public StreamingReverb(int sampleRate = 48000)
    {
        // IR length: 2.618 seconds (golden ratio)
        int irLength = (int)(sampleRate * 2.618f);

        // Build impulse response: exponential decay + PHI sinusoidal modulation.
        // Decay rate 0.75 = 3/4 (Pythagorean ratio from the Merkaba 3:4:5 triangle).
        // Decay multiplier = PHI (golden ratio shapes the reverb tail envelope).
        // Modulation amplitude = 1/13 (Fibonacci reciprocal).
        _ir = new float[irLength];
        float decayRate = 0.75f; // 3/4 Pythagorean ratio
        for (int i = 0; i < irLength; i++)
        {
            float t = (float)i / irLength;
            // Exponential decay shaped by PHI — golden ratio governs the reverb tail
            _ir[i] = MathF.Exp(-t * decayRate * SacredConstants.REVERB_DECAY_MULTIPLIER);
            // PHI sinusoidal modulation at 1/13 Fibonacci amplitude
            _ir[i] += SacredConstants.REVERB_MOD_AMP * MathF.Sin(SacredConstants.TWO_PI *
                SacredConstants.PHI * (float)i / irLength);
        }

        // Normalize IR
        float maxAbs = 0f;
        for (int i = 0; i < irLength; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(_ir[i]));
        if (maxAbs > 0f)
            for (int i = 0; i < irLength; i++)
                _ir[i] /= maxAbs;

        // Initialize tail buffer
        _tail = new float[irLength - 1];
    }

    #endregion

    // FFT-based overlap-add convolution — convolves each chunk with the IR,
    // adds the previous tail to the output, and saves the new tail for the next chunk.
    // Returns only the first sigLen samples (same length as input).
    #region Overlap-Add Convolution

    /// <summary>
    /// Process a chunk with overlap-add convolution.
    /// Returns output of same length as input, carrying tail to next call.
    /// </summary>
    public float[] ProcessChunk(ReadOnlySpan<float> chunk)
    {
        int sigLen = chunk.Length;

        // FFT convolve: output length = sigLen + irLen - 1.
        // The IR's forward FFT is computed once and cached — only the signal
        // side is FFT'd per chunk. Cache is sized from the largest chunk seen
        // (the first full 3-second chunk); the final shorter chunk of a session
        // reuses the same FFT size, which is exact for linear convolution.
        int resultLen = sigLen + _ir.Length - 1;
        int neededFftLen = FftConvolution.NextPowerOf2(resultLen);
        if (_kernelFft == null || neededFftLen > _kernelFft.Length)
            _kernelFft = FftConvolution.PrepareKernel(_ir, neededFftLen);

        var fullOutput = FftConvolution.ConvolveWithPreparedKernel(chunk, _kernelFft, resultLen);

        // Overlap-add: add previous tail to beginning of output
        int tailAdd = global::System.Math.Min(_tail.Length, fullOutput.Length);
        for (int i = 0; i < tailAdd; i++)
            fullOutput[i] += _tail[i];

        // Save new tail for next chunk
        int newTailLen = fullOutput.Length - sigLen;
        var newTail = new float[_ir.Length - 1];
        int copyLen = global::System.Math.Min(newTailLen, newTail.Length);
        if (copyLen > 0)
            Array.Copy(fullOutput, sigLen, newTail, 0, copyLen);
        _tail = newTail;

        // Return only the first sigLen samples
        var result = new float[sigLen];
        Array.Copy(fullOutput, result, sigLen);
        return result;
    }

    /// <summary>
    /// Reset tail buffer (for starting a new stream).
    /// </summary>
    public void Reset()
    {
        Array.Clear(_tail);
    }

    #endregion
}
