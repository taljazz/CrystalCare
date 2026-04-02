namespace CrystalCare.Core.Generation;

/// <summary>
/// Logistic map chaotic number generator.
/// Produces deterministic but chaotic sequences for organic frequency variation.
/// Port of ChaoticSelector from SoundGenerator.py.
/// </summary>
public sealed class ChaoticSelector
{
    private float _x;
    private readonly float _r;

    public ChaoticSelector(float seed = 0.75f, float r = 3.9f)
    {
        _x = seed;
        _r = r;
    }

    /// <summary>
    /// Generate next chaotic value in [0, 1] range using logistic map: x = r * x * (1 - x)
    /// </summary>
    public float NextValue()
    {
        _x = _r * _x * (1.0f - _x);
        return _x;
    }
}
