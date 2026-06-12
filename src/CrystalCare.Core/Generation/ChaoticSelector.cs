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

    /// <summary>
    /// Create a logistic-map chaos source. The seed defaults to a random point
    /// in the chaotic basin (0.25..0.75) so every app launch walks a different
    /// trajectory — previously the fixed 0.75 seed made the first session after
    /// every launch draw IDENTICAL chaos values (same chaosVal drift offset,
    /// same ADSR chaos), quietly contradicting "no two sessions the same by
    /// design". An explicit seed is still accepted for reproducible testing.
    /// At r = 3.9 the map is chaotic for essentially any seed in (0, 1); the
    /// 0.25..0.75 band just keeps the start well away from the fixed points.
    /// </summary>
    public ChaoticSelector(float? seed = null, float r = 3.9f)
    {
        _x = seed ?? (0.25f + (float)Random.Shared.NextDouble() * 0.5f);
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
