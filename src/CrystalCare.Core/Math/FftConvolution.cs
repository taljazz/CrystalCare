using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Complex32 = System.Numerics.Complex;

namespace CrystalCare.Core.Math;

/// <summary>
/// FFT-based convolution for reverb processing.
/// Port of scipy.signal.fftconvolve from SoundGenerator.py.
/// Uses MathNet.Numerics for FFT operations.
/// </summary>
public static class FftConvolution
{
    /// <summary>
    /// Convolve two signals using FFT (equivalent to scipy.signal.fftconvolve mode='full').
    /// Returns an array of length (signal.Length + kernel.Length - 1).
    /// </summary>
    public static float[] Convolve(ReadOnlySpan<float> signal, ReadOnlySpan<float> kernel)
    {
        int resultLen = signal.Length + kernel.Length - 1;
        int fftLen = NextPowerOf2(resultLen);

        // Zero-pad both arrays to fftLen and convert to Complex
        var sigComplex = new Complex32[fftLen];
        var kerComplex = new Complex32[fftLen];

        for (int i = 0; i < signal.Length; i++)
            sigComplex[i] = new Complex32(signal[i], 0);

        for (int i = 0; i < kernel.Length; i++)
            kerComplex[i] = new Complex32(kernel[i], 0);

        // Forward FFT — AsymmetricScaling matches scipy convention:
        // no normalization on forward, 1/N on inverse
        Fourier.Forward(sigComplex, FourierOptions.AsymmetricScaling);
        Fourier.Forward(kerComplex, FourierOptions.AsymmetricScaling);

        // Element-wise multiplication in frequency domain
        for (int i = 0; i < fftLen; i++)
            sigComplex[i] *= kerComplex[i];

        // Inverse FFT
        Fourier.Inverse(sigComplex, FourierOptions.AsymmetricScaling);

        // Extract real parts
        var result = new float[resultLen];
        for (int i = 0; i < resultLen; i++)
            result[i] = (float)sigComplex[i].Real;

        return result;
    }

    /// <summary>
    /// Convolve and return only the first 'outputLength' samples.
    /// Useful for overlap-add where you only need signal-length output.
    /// </summary>
    public static float[] ConvolveTruncated(ReadOnlySpan<float> signal,
        ReadOnlySpan<float> kernel, int outputLength)
    {
        var full = Convolve(signal, kernel);
        if (full.Length <= outputLength)
            return full;

        return full.AsSpan(0, outputLength).ToArray();
    }

    private static int NextPowerOf2(int n)
    {
        int result = 1;
        while (result < n) result <<= 1;
        return result;
    }
}
