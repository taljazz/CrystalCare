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
/// The piecewise segments are NON-LINEAR. Each quarter-cycle climb or descent
/// follows <c>y = x^φ</c> rather than a straight line — the golden ratio (PHI)
/// shapes the curvature of the wave the same way it shapes her breath ladder,
/// her modulation, her fade timings, and her torus geometry. The curve of the
/// wave is the curve of everything else she does. Linear ramps belong to
/// mechanical synthesis; φ ramps belong to her.
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
    /// Triangle — piecewise wave with the same geometric structure as a triangle
    /// (zero at half cycles, peak at quarter, trough at three-quarter), but each
    /// quarter-cycle segment is shaped by the golden ratio: <c>y = x^φ</c> within
    /// the segment rather than a linear ramp. The default for the harmonic field
    /// and 6 of the 7 sacred layers.
    ///
    /// Odd-harmonic-rich (still odd-symmetric about every zero crossing) so at
    /// the Lemurian 432 Hz keynote she rings Tesla 3 (1296 Hz), Pythagorean 5
    /// (2160 Hz), sacred 7 (3024 Hz), and Tesla 9 (3888 Hz) inherent in the
    /// waveform. The φ-shaping additionally softens the zero crossings (slope
    /// approaches 0 at the bottom of each cycle) and sharpens the peaks
    /// (slope ±4φ ≈ ±6.47 at each extremum, larger discontinuity than a linear
    /// triangle's ±4) — brighter top, calmer bottom, like a struck bowl
    /// swelling to a focused peak then receding.
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
    /// Triangle wave with PHI-shaped segments: starts at 0, peaks at +1 at quarter
    /// cycle, returns to 0 at half cycle, trough -1 at three-quarter cycle, back
    /// to 0 at full cycle. Same period, peak amplitude, and zero-crossing geometry
    /// as a linear triangle (and as sin), so it's a drop-in replacement anywhere
    /// the codebase previously called Sin for tone generation.
    ///
    /// The piecewise segments are NOT linear. Each quarter-cycle climb or descent
    /// follows <c>y = x^φ</c> (PHI ≈ 1.618), where x is the linear position within
    /// the segment normalized to [0, 1]. The golden ratio shapes the curvature of
    /// the wave just as it shapes the breath ladder, the modulation, the fade
    /// timings, and the torus geometry — the curve of the wave is the curve of
    /// everything else she does. Linear ramps belong to mechanical instruments;
    /// φ ramps belong to her.
    ///
    /// Slope characteristics vs. a linear triangle (slope ±4 throughout):
    ///   - At zero crossings (t = 0, 0.5, 1): slope → 0 — smooth approach, no
    ///     buzz at the bottom of each cycle.
    ///   - At each extremum (t = 0.25 peak, 0.75 trough): slope = ±4·φ ≈ ±6.47,
    ///     jump of 12.94 across the corner (vs. linear triangle's 8) — sharper,
    ///     richer in upper odd harmonics.
    ///   - Net timbre: brighter at the top of each cycle, calmer at the bottom;
    ///     like a struck crystal bowl swelling to a focused peak then receding.
    ///
    /// Cost: one <c>Math.Pow</c> per sample per voice. Accepted in service of
    /// sacred shape over linear synthesis — performance loss is a fraction of a
    /// percent of the existing pipeline and the rest of the design assumes this
    /// trade.
    /// </summary>
    public static float Triangle(double phase)
    {
        // Map radian phase to cycle position [0, 1).
        // We use TWO_PI_D (sacred constant for 2π in double precision) so this
        // matches every other phase calculation in the pipeline.
        double t = phase / SacredConstants.TWO_PI_D;
        t -= System.Math.Floor(t);  // wrap to [0, 1) — handles negative phases too

        // Each quarter cycle is its own segment. We split the full cycle into four
        // equal segments (rather than the linear triangle's three) so that every
        // segment runs between an extremum and a zero crossing — that way the
        // PHI power can be applied to a quantity that always rises from 0 to 1
        // within the segment, then we negate as needed for the negative half-cycle.
        //
        //   t ∈ [0,    0.25)  :  +(4t)^φ          climb 0 → +1
        //   t ∈ [0.25, 0.5)   :  +(2 - 4t)^φ      descent +1 → 0
        //   t ∈ [0.5,  0.75)  :  -(4t - 2)^φ      climb (negated) 0 → -1
        //   t ∈ [0.75, 1.0)   :  -(4 - 4t)^φ      descent (negated) -1 → 0
        //
        // PHI is pulled from SacredConstants — same constant the breath ladder,
        // fade timings, torus radii, and reverb decay multiplier are built on.
        // Math.Pow takes double, so the implicit float→double cast on PHI is fine.
        if (t < 0.25) return (float)System.Math.Pow(4.0 * t, SacredConstants.PHI);
        if (t < 0.5)  return (float)System.Math.Pow(2.0 - 4.0 * t, SacredConstants.PHI);
        if (t < 0.75) return -(float)System.Math.Pow(4.0 * t - 2.0, SacredConstants.PHI);
        return -(float)System.Math.Pow(4.0 - 4.0 * t, SacredConstants.PHI);
    }

    #endregion
}
