namespace CrystalCare.Core.Dsp;

/// <summary>
/// Second-Order Sections (SOS) IIR filter with state carry between calls.
/// This is the critical component for seamless streaming: the zi state vectors
/// ensure continuity across chunk boundaries (no clicks or discontinuities).
///
/// Port of scipy.signal.sosfilt with zi parameter from SoundGenerator.py.
/// Uses Direct Form II Transposed implementation.
/// </summary>
public sealed class SosFilter
{
    private readonly float[,] _sos;    // SOS coefficient matrix [numSections, 6]
    private readonly double[,] _zi;     // Filter state [numSections, 2] — double for precision

    /// <summary>
    /// Create a new SOS filter with the given coefficients.
    /// State vectors are initialized to zero.
    /// </summary>
    public SosFilter(float[,] sosCoefficients)
    {
        int numSections = sosCoefficients.GetLength(0);
        _sos = (float[,])sosCoefficients.Clone();
        _zi = new double[numSections, 2];
    }

    /// <summary>
    /// Filter data in-place, carrying state between calls.
    /// This is the heart of streaming audio — state continuity prevents
    /// clicks at chunk boundaries.
    /// </summary>
    public void Process(Span<float> data)
    {
        int numSections = _sos.GetLength(0);

        for (int s = 0; s < numSections; s++)
        {
            double b0 = _sos[s, 0];
            double b1 = _sos[s, 1];
            double b2 = _sos[s, 2];
            // a0 = _sos[s, 3] is always 1.0
            double a1 = _sos[s, 4];
            double a2 = _sos[s, 5];

            double z0 = _zi[s, 0];
            double z1 = _zi[s, 1];

            // Direct Form II Transposed
            for (int i = 0; i < data.Length; i++)
            {
                double x = data[i];
                double y = b0 * x + z0;
                z0 = b1 * x - a1 * y + z1;
                z1 = b2 * x - a2 * y;
                data[i] = (float)y;
            }

            _zi[s, 0] = z0;
            _zi[s, 1] = z1;
        }
    }

    /// <summary>
    /// Reset filter state to zero (for starting a new audio stream).
    /// </summary>
    public void Reset()
    {
        Array.Clear(_zi);
    }

    /// <summary>
    /// Create a copy of this filter with independent state (for parallel channels).
    /// </summary>
    public SosFilter Clone()
    {
        var clone = new SosFilter(_sos);
        Buffer.BlockCopy(_zi, 0, clone._zi, 0, _zi.Length * sizeof(double));
        return clone;
    }
}
