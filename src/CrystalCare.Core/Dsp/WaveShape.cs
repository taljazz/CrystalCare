using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Wave-shape selector for CrystalCare's tone formation.
///
/// CrystalCare's harmonic field and 6 of the 7 sacred layers default to
/// triangle waves rather than sine. Triangle waves at her keynote (e.g., 432 Hz)
/// inherently contain the same sacred mathematics she's built on, expressed
/// in one waveform:
///
///   - 1st harmonic (432 Hz): Lemurian keynote
///   - 3rd harmonic (1296 Hz): Solfeggio crown / Tesla 3
///   - 5th harmonic (2160 Hz): Pythagorean 5:1 ratio
///   - 7th harmonic (3024 Hz): sacred 7
///   - 9th harmonic (3888 Hz): Tesla 9
///
/// A triangle wave is the cymatic form that emerges from her geometric DNA:
/// her Merkaba layer is built on the star tetrahedron (interlocked triangles);
/// her Solfeggio grid honors Tesla's 3-6-9 vortex (a triangular flow pattern).
/// The waveform now matches the form.
///
/// Two exceptions remain sine:
///   - Blue Ray Resonance Layer: the still axis, zero point — purity by design.
///     "Always in the center, never wavering."
///   - Taygetan binaural carriers (first 9 voices in Stage 6): pure-sine carriers
///     give the cleanest L/R difference frequency the brain entrains to. With
///     triangle there, the binaural sync becomes muddier.
///
/// All sub-audible modulators (LFO, fractal variation, geometric modulation,
/// pan curves, beat drift, master fade) stay sine — they aren't voices, they're
/// shape-drivers, and sine keeps their math clean.
/// </summary>
public enum WaveShape
{
    /// <summary>Pure sine — for binaural carriers, Blue Ray, modulators.</summary>
    Sine,

    /// <summary>
    /// Triangle — fundamental + odd harmonics at 1/n² falloff. The default for
    /// the harmonic field and 6 of the 7 sacred layers. Contains Tesla 3-6-9
    /// + Pythagorean 5:1 + Solfeggio crown harmonics in one waveform when
    /// played at the Lemurian 432 Hz keynote.
    /// </summary>
    Triangle,
}

/// <summary>
/// Wave-shape evaluation functions. All operations are double-precision phase
/// in, single-precision sample out, matching the rest of the audio pipeline.
/// </summary>
public static class WaveShapes
{
    // Sample evaluation — dispatches on the shape enum. The hot path through
    // HarmonicGenerator's per-voice loop calls this once per sample per voice,
    // so the switch is JIT-inlinable. Inner functions are also static for the
    // same reason.
    #region Public Evaluation

    /// <summary>
    /// Evaluate a wave at a given phase. Phase is in radians.
    /// For Sine: returns sin(phase). For Triangle: returns the piecewise
    /// triangle wave (peak +1 at quarter cycle, trough -1 at three-quarter).
    /// Both shapes have peak amplitude 1.0 and starting value 0.0 at phase=0.
    /// </summary>
    public static float Compute(WaveShape shape, double phase)
    {
        // The switch is small enough to be branch-predictable and JIT-friendly.
        // Triangle is the default elsewhere in the pipeline so it's named first.
        return shape switch
        {
            WaveShape.Triangle => Triangle(phase),
            _ => (float)System.Math.Sin(phase),
        };
    }

    /// <summary>
    /// Pure sine wave. Same as System.Math.Sin but wrapped here so call sites
    /// reading the codebase see WaveShapes.Sine and know the choice is intentional.
    /// </summary>
    public static float Sine(double phase)
    {
        // Double-precision sin, cast back to float for the audio sample.
        // Matches the existing pattern across HarmonicGenerator and sacred layers.
        return (float)System.Math.Sin(phase);
    }

    /// <summary>
    /// Triangle wave: starts at 0, peaks at +1 at quarter cycle, returns to 0 at
    /// half cycle, trough -1 at three-quarter cycle, back to 0 at full cycle.
    /// Same period and peak amplitude as sin(phase), so it's a drop-in replacement
    /// in any place that was previously calling Math.Sin for a tone-generation purpose.
    ///
    /// Implementation: piecewise linear (no asin / no Math calls beyond Floor)
    /// for hot-path efficiency. Triangle wave's mathematical Fourier series is
    /// inherent in the shape — odd harmonics at 1/n² amplitude — so calling this
    /// at 432 Hz gives us Lemurian keynote + Tesla 3 (1296) + Pythagorean 5
    /// (2160) + Tesla 9 (3888) etc., all in one waveform.
    /// </summary>
    public static float Triangle(double phase)
    {
        // Map radian phase to cycle position [0, 1).
        // We use TWO_PI_D (sacred constant for 2π in double precision) so this
        // matches every other phase calculation in the pipeline.
        double t = phase / SacredConstants.TWO_PI_D;
        t -= System.Math.Floor(t);  // wrap to [0, 1) — handles negative phases too

        // Triangle shape over the cycle (matches sin start/zero crossings):
        //   t ∈ [0,    0.25)  : ramp 0 → +1     (slope +4)
        //   t ∈ [0.25, 0.75)  : ramp +1 → -1   (slope -4, crosses zero at t=0.5)
        //   t ∈ [0.75, 1.0)   : ramp -1 → 0     (slope +4)
        if (t < 0.25) return (float)(4.0 * t);
        if (t < 0.75) return (float)(2.0 - 4.0 * t);
        return (float)(4.0 * t - 4.0);
    }

    #endregion
}
