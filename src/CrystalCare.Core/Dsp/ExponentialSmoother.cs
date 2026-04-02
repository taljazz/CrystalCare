namespace CrystalCare.Core.Dsp;

/// <summary>
/// Simple 1-pole exponential low-pass smoother with state carry.
/// Numerically stable at ANY cutoff frequency, including extremely low values
/// like 0.002 Hz that cause Butterworth bilinear transform to fail.
///
/// Used for the pan curve filter where Python uses butter(2, 0.002/Nyquist).
/// The 0.002 Hz cutoff makes panning extremely slow and organic — only the
/// slowest drift survives, giving the non-linear toroidal panning its
/// characteristic gentle, breathing quality.
///
/// y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
/// alpha = 1 - exp(-2*PI*cutoff/sampleRate)
/// </summary>
public sealed class ExponentialSmoother
{
    private readonly float _alpha;
    private float _state;
    private bool _initialized;

    /// <summary>
    /// Create a smoother with the given cutoff frequency.
    /// For 0.002 Hz at 48kHz, alpha ≈ 2.6e-7 (extremely slow smoothing).
    /// </summary>
    public ExponentialSmoother(float cutoffHz, float sampleRate)
    {
        _alpha = 1.0f - MathF.Exp(-2.0f * MathF.PI * cutoffHz / sampleRate);
        _state = 0f;
        _initialized = false;
    }

    /// <summary>
    /// Filter data in-place, carrying state between calls.
    /// </summary>
    public void Process(Span<float> data)
    {
        if (!_initialized && data.Length > 0)
        {
            _state = data[0];
            _initialized = true;
        }

        for (int i = 0; i < data.Length; i++)
        {
            _state += _alpha * (data[i] - _state);
            data[i] = _state;
        }
    }

    public void Reset()
    {
        _state = 0f;
        _initialized = false;
    }
}
