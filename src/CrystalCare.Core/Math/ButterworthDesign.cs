using System.Numerics;

namespace CrystalCare.Core.Math;

/// <summary>
/// Butterworth IIR filter coefficient design (bilinear transform method).
/// Produces Second-Order Sections (SOS) format coefficients matching scipy.signal.butter.
///
/// The SOS format is numerically stable for stateful (zi-carry) filtering across chunks.
/// Each section is [b0, b1, b2, a0, a1, a2] where a0 is always 1.0.
/// </summary>
public static class ButterworthDesign
{
    /// <summary>
    /// Design a Butterworth low-pass filter in SOS format.
    /// Returns array of second-order sections: float[numSections, 6]
    /// where each row is [b0, b1, b2, 1.0, a1, a2].
    /// </summary>
    public static float[,] DesignLowpass(int order, float cutoffHz, float sampleRate)
    {
        try
        {
            return DesignLowpassCore(order, cutoffHz, sampleRate);
        }
        catch
        {
            // Fallback: unity passthrough filter (no filtering)
            // 1 section: b=[1,0,0], a=[1,0,0]
            var fallback = new float[1, 6];
            fallback[0, 0] = 1f; // b0
            fallback[0, 3] = 1f; // a0
            return fallback;
        }
    }

    private static float[,] DesignLowpassCore(int order, float cutoffHz, float sampleRate)
    {
        float nyquist = sampleRate / 2.0f;
        float normalizedCutoff = cutoffHz / nyquist;
        normalizedCutoff = global::System.Math.Clamp(normalizedCutoff, 0.01f, 0.95f);

        // Pre-warp the cutoff frequency for the bilinear transform
        float warpedCutoff = MathF.Tan(MathF.PI * normalizedCutoff);

        // Compute analog Butterworth poles (left half of s-plane unit circle)
        int numPoles = order;
        var analogPoles = new Complex[numPoles];
        for (int k = 0; k < numPoles; k++)
        {
            double angle = global::System.Math.PI * (2.0 * k + numPoles + 1) / (2.0 * numPoles);
            analogPoles[k] = new Complex(global::System.Math.Cos(angle), global::System.Math.Sin(angle));
            // Scale by warped cutoff
            analogPoles[k] *= warpedCutoff;
        }

        // Bilinear transform: z = (1 + s) / (1 - s)
        var digitalPoles = new Complex[numPoles];
        for (int k = 0; k < numPoles; k++)
        {
            digitalPoles[k] = (1.0 + analogPoles[k]) / (1.0 - analogPoles[k]);
        }

        // Digital zeros are all at z = -1 for lowpass Butterworth
        // (the analog prototype has zeros at infinity, bilinear maps infinity to -1)

        // Pair conjugate poles into second-order sections
        int numSections = (order + 1) / 2;
        var sos = new float[numSections, 6];

        var usedPoles = new bool[numPoles];
        int sectionIdx = 0;

        // Handle odd-order: one real pole becomes a first-order section stored as SOS
        if (order % 2 == 1)
        {
            // Find the real pole (closest to real axis)
            int realIdx = -1;
            double minImag = double.MaxValue;
            for (int k = 0; k < numPoles; k++)
            {
                if (global::System.Math.Abs(digitalPoles[k].Imaginary) < minImag)
                {
                    minImag = global::System.Math.Abs(digitalPoles[k].Imaginary);
                    realIdx = k;
                }
            }

            double p = digitalPoles[realIdx].Real;
            usedPoles[realIdx] = true;

            // First-order section as SOS: (1 + z^-1) / (1 - p*z^-1)
            // Gain: (1 - p) / 2 for unity DC gain
            double gain = (1.0 - p) / 2.0;
            sos[sectionIdx, 0] = (float)gain;       // b0
            sos[sectionIdx, 1] = (float)gain;       // b1
            sos[sectionIdx, 2] = 0f;                // b2
            sos[sectionIdx, 3] = 1f;                // a0
            sos[sectionIdx, 4] = (float)(-p);       // a1
            sos[sectionIdx, 5] = 0f;                // a2
            sectionIdx++;
        }

        // Pair remaining conjugate poles
        for (int k = 0; k < numPoles && sectionIdx < numSections; k++)
        {
            if (usedPoles[k]) continue;
            usedPoles[k] = true;

            // Find conjugate partner
            int conjugateIdx = -1;
            double minDist = double.MaxValue;
            for (int j = k + 1; j < numPoles; j++)
            {
                if (usedPoles[j]) continue;
                double dist = Complex.Abs(digitalPoles[k] - Complex.Conjugate(digitalPoles[j]));
                if (dist < minDist)
                {
                    minDist = dist;
                    conjugateIdx = j;
                }
            }

            if (conjugateIdx >= 0)
                usedPoles[conjugateIdx] = true;

            Complex p1 = digitalPoles[k];
            Complex p2 = conjugateIdx >= 0 ? digitalPoles[conjugateIdx] : Complex.Conjugate(p1);

            // Second-order section: (1 + 2*z^-1 + z^-2) / (1 + a1*z^-1 + a2*z^-2)
            // Denominator from poles: (1 - p1*z^-1)(1 - p2*z^-1)
            //   = 1 - (p1+p2)*z^-1 + p1*p2*z^-2
            double a1 = -(p1 + p2).Real;
            double a2 = (p1 * p2).Real;

            // Numerator: zeros at z = -1: (1 + z^-1)^2 = 1 + 2z^-1 + z^-2
            // Gain for unity DC: evaluate H(z=1) = (1+2+1)/(1+a1+a2) = 4/(1+a1+a2)
            // We want H(1) = 1, so gain = (1+a1+a2)/4
            double gain = (1.0 + a1 + a2) / 4.0;

            sos[sectionIdx, 0] = (float)gain;         // b0
            sos[sectionIdx, 1] = (float)(2.0 * gain); // b1
            sos[sectionIdx, 2] = (float)gain;         // b2
            sos[sectionIdx, 3] = 1f;                  // a0
            sos[sectionIdx, 4] = (float)a1;           // a1
            sos[sectionIdx, 5] = (float)a2;           // a2
            sectionIdx++;
        }

        return sos;
    }

    /// <summary>
    /// Design Butterworth and return (b, a) transfer function coefficients.
    /// Used for non-stateful filtering (batch mode).
    /// </summary>
    public static (float[] b, float[] a) DesignLowpassBA(int order, float cutoffHz, float sampleRate)
    {
        // For simplicity, use SOS and convert to transfer function
        // But for batch filtering, SOS is actually preferred anyway
        // This is a convenience wrapper
        var sos = DesignLowpass(order, cutoffHz, sampleRate);
        return SosToBA(sos);
    }

    private static (float[] b, float[] a) SosToBA(float[,] sos)
    {
        int numSections = sos.GetLength(0);
        float[] b = [1f];
        float[] a = [1f];

        for (int s = 0; s < numSections; s++)
        {
            float[] secB = [sos[s, 0], sos[s, 1], sos[s, 2]];
            float[] secA = [sos[s, 3], sos[s, 4], sos[s, 5]];
            b = ConvolvePolynomials(b, secB);
            a = ConvolvePolynomials(a, secA);
        }

        return (b, a);
    }

    private static float[] ConvolvePolynomials(float[] a, float[] b)
    {
        var result = new float[a.Length + b.Length - 1];
        for (int i = 0; i < a.Length; i++)
            for (int j = 0; j < b.Length; j++)
                result[i + j] += a[i] * b[j];
        return result;
    }
}
