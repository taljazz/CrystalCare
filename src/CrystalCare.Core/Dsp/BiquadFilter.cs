namespace CrystalCare.Core.Dsp;

/// <summary>
/// Cascaded biquad (second-order section) low-pass filter with state carry.
/// Uses Robert Bristow-Johnson's Audio EQ Cookbook formula — numerically stable
/// at any valid cutoff frequency.
///
/// Numerically stable lowpass for the main pipeline, avoiding bilinear
/// transform instability at extreme cutoff values.
///
/// Multiple biquads can be cascaded for higher-order filtering:
/// Order 4 = 2 cascaded biquads (each is 2nd order).
/// </summary>
public sealed class BiquadFilter
{
    // Factory method and internal state. Creates cascaded biquad sections
    // for Butterworth-characteristic lowpass filtering at any cutoff frequency.
    #region Factory and Fields

    private readonly BiquadSection[] _sections;

    /// <summary>
    /// Create a cascaded biquad low-pass filter.
    /// order must be even (2, 4, 6, ...) — each pair becomes one biquad section.
    /// </summary>
    public static BiquadFilter CreateLowpass(int order, float cutoffHz, float sampleRate)
    {
        // Number of second-order sections
        int numSections = order / 2;
        if (numSections < 1) numSections = 1;

        var sections = new BiquadSection[numSections];

        for (int i = 0; i < numSections; i++)
        {
            // Butterworth Q values for cascaded sections
            // For order N, the Q of the k-th section is:
            // Q = 1 / (2 * cos(PI * (2k + 1) / (2N)))
            float angle = MathF.PI * (2 * i + 1) / (2.0f * order);
            float q = 1.0f / (2.0f * MathF.Cos(angle));

            sections[i] = BiquadSection.CreateLowpass(cutoffHz, sampleRate, q);
        }

        return new BiquadFilter(sections);
    }

    private BiquadFilter(BiquadSection[] sections)
    {
        _sections = sections;
    }

    #endregion

    // In-place filtering with state carry between calls — essential for seamless
    // streaming across chunk boundaries (no clicks or discontinuities).
    #region Processing

    public void Process(Span<float> data)
    {
        foreach (var section in _sections)
            section.Process(data);
    }

    public void Reset()
    {
        foreach (var section in _sections)
            section.Reset();
    }

    #endregion

    // Single biquad section using Direct Form II Transposed implementation.
    // Audio EQ Cookbook formula — numerically stable at any valid cutoff.
    // State variables use double precision for accumulated filter accuracy.
    #region Biquad Section (Inner Class)

    /// <summary>
    /// Single biquad section (Direct Form II Transposed).
    /// Uses Audio EQ Cookbook formulas for guaranteed stability.
    /// </summary>
    private sealed class BiquadSection
    {
        private float _b0, _b1, _b2, _a1, _a2;
        private double _z1, _z2; // State variables (double for precision)

        /// <summary>
        /// Create a low-pass biquad using Audio EQ Cookbook formula.
        /// Stable for any cutoff from near-DC to near-Nyquist.
        /// </summary>
        public static BiquadSection CreateLowpass(float cutoffHz, float sampleRate, float q)
        {
            var section = new BiquadSection();

            float w0 = 2.0f * MathF.PI * cutoffHz / sampleRate;
            float cosW0 = MathF.Cos(w0);
            float sinW0 = MathF.Sin(w0);
            float alpha = sinW0 / (2.0f * q);

            float a0 = 1.0f + alpha;
            section._b0 = (1.0f - cosW0) / 2.0f / a0;
            section._b1 = (1.0f - cosW0) / a0;
            section._b2 = (1.0f - cosW0) / 2.0f / a0;
            section._a1 = (-2.0f * cosW0) / a0;
            section._a2 = (1.0f - alpha) / a0;

            return section;
        }

        /// <summary>
        /// Process samples in-place using Direct Form II Transposed.
        /// </summary>
        public void Process(Span<float> data)
        {
            double b0 = _b0, b1 = _b1, b2 = _b2, a1 = _a1, a2 = _a2;
            double z1 = _z1, z2 = _z2;

            for (int i = 0; i < data.Length; i++)
            {
                double x = data[i];
                double y = b0 * x + z1;
                z1 = b1 * x - a1 * y + z2;
                z2 = b2 * x - a2 * y;
                data[i] = (float)y;
            }

            _z1 = z1;
            _z2 = z2;
        }

        public void Reset()
        {
            _z1 = 0;
            _z2 = 0;
        }
    }

    #endregion
}
