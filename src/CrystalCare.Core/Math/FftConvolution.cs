using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Complex32 = System.Numerics.Complex;

namespace CrystalCare.Core.Math;

/// <summary>
/// FFT-based convolution for reverb processing.
/// Port of scipy.signal.fftconvolve from SoundGenerator.py.
/// Uses MathNet.Numerics for FFT operations.
///
/// API shape: prepare-once / convolve-many. The reverb kernel (impulse response)
/// never changes during a session, so its forward FFT is computed ONCE via
/// PrepareKernel and reused for every chunk via ConvolveWithPreparedKernel.
/// The previous Convolve(signal, kernel) form re-FFT'd the constant kernel on
/// every call — one of the three large FFTs per convolution was pure waste.
/// Caching it cuts roughly a third of all reverb CPU with bit-identical output
/// for full-size chunks (the final short chunk of a session convolves through
/// the same cached FFT size, differing from a minimal-size FFT only at
/// float-rounding level — far below audibility, during the fade-out tail).
/// </summary>
public static class FftConvolution
{
    // Kernel preparation — forward-FFT the kernel once at a chosen FFT length.
    // The caller picks fftLen = NextPowerOf2(maxSignalLength + kernelLength - 1)
    // so every subsequent chunk (same length or shorter) convolves exactly.
    #region Kernel Preparation

    /// <summary>
    /// Zero-pad the kernel to fftLen and compute its forward FFT once.
    /// The returned spectrum is reusable across every ConvolveWithPreparedKernel
    /// call whose (signal.Length + kernel.Length - 1) fits within fftLen.
    /// </summary>
    /// <param name="kernel">The convolution kernel (reverb impulse response).</param>
    /// <param name="fftLen">
    /// FFT size — must be a power of two at least (maxSignalLength + kernel.Length − 1).
    /// Use NextPowerOf2 to compute it from the largest chunk the caller will process.
    /// </param>
    public static Complex32[] PrepareKernel(ReadOnlySpan<float> kernel, int fftLen)
    {
        // Zero-pad the kernel into the complex FFT buffer
        var kerComplex = new Complex32[fftLen];
        for (int i = 0; i < kernel.Length; i++)
            kerComplex[i] = new Complex32(kernel[i], 0);

        // Forward FFT — AsymmetricScaling matches scipy convention:
        // no normalization on forward, 1/N on inverse
        Fourier.Forward(kerComplex, FourierOptions.AsymmetricScaling);
        return kerComplex;
    }

    #endregion

    // Convolution against a prepared kernel spectrum — one forward FFT for the
    // signal, a spectral multiply, and one inverse FFT. The kernel's FFT cost
    // is paid once per session instead of once per chunk.
    #region Convolution

    /// <summary>
    /// Convolve a signal against a kernel spectrum prepared by PrepareKernel
    /// (equivalent to scipy.signal.fftconvolve mode='full', truncated to resultLen).
    /// </summary>
    /// <param name="signal">The input signal chunk.</param>
    /// <param name="kernelFft">Prepared kernel spectrum from PrepareKernel.</param>
    /// <param name="resultLen">
    /// Number of output samples to return — for full linear convolution this is
    /// (signal.Length + kernelLength − 1). Must satisfy resultLen ≤ kernelFft.Length.
    /// </param>
    public static float[] ConvolveWithPreparedKernel(ReadOnlySpan<float> signal,
        Complex32[] kernelFft, int resultLen)
    {
        int fftLen = kernelFft.Length;

        // Zero-pad the signal to the cached FFT length and convert to Complex
        var sigComplex = new Complex32[fftLen];
        for (int i = 0; i < signal.Length; i++)
            sigComplex[i] = new Complex32(signal[i], 0);

        // Forward FFT on the signal only — the kernel side is already cached
        Fourier.Forward(sigComplex, FourierOptions.AsymmetricScaling);

        // Element-wise multiplication in frequency domain
        for (int i = 0; i < fftLen; i++)
            sigComplex[i] *= kernelFft[i];

        // Inverse FFT
        Fourier.Inverse(sigComplex, FourierOptions.AsymmetricScaling);

        // Extract real parts of the linear-convolution region
        var result = new float[resultLen];
        for (int i = 0; i < resultLen; i++)
            result[i] = (float)sigComplex[i].Real;

        return result;
    }

    #endregion

    // Power-of-two sizing helper — public so callers (StreamingReverb) can size
    // the cached kernel FFT from their maximum chunk length.
    #region Sizing

    /// <summary>Smallest power of two ≥ n. Used to size the shared FFT length.</summary>
    public static int NextPowerOf2(int n)
    {
        int result = 1;
        while (result < n) result <<= 1;
        return result;
    }

    #endregion
}
