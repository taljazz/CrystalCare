using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Math;
using CrystalCare.Core.Noise;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Toroidal panning with Rössler chaotic perturbation.
/// Creates immersive 3D-like spatial movement in stereo field.
///
/// The panning path traces a torus (donut shape) in parameter space,
/// with chaotic Rössler perturbation for organic, non-repeating movement.
///
/// Port of toroidal panning logic from SoundGenerator.py generate_dynamic_sound_stream().
/// </summary>
public static class ToroidalPanner
{
    // Pre-computed panning parameters — theta/phi frequencies and drift settings.
    // Drawn once at pipeline start for session-wide consistency.
    #region Parameters

    public sealed class PanParams
    {
        public float ThetaFreq { get; init; }
        public float PhiFreq { get; init; }
        public float PanDriftFreq { get; init; }
        public float PanDriftAmp { get; init; }
    }

    /// <summary>
    /// Draw random panning parameters.
    /// </summary>
    public static PanParams DrawParams(Random? rng = null)
    {
        rng ??= Random.Shared;
        return new PanParams
        {
            ThetaFreq = (float)(rng.NextDouble() * 0.04 + 0.01),   // [0.01, 0.05]
            PhiFreq = (float)(rng.NextDouble() * 0.04 + 0.01) * SacredConstants.PHI,
            PanDriftFreq = (float)(rng.NextDouble() * 0.005 + 0.002),
            PanDriftAmp = (float)(rng.NextDouble() * 0.1 + 0.05),
        };
    }

    #endregion

    // Computes stereo gain arrays from toroidal panning + optional Rössler chaos.
    // Maps torus surface point to horizontal pan and depth for stereo placement.
    // PHI-derived minor radius (0.382) creates golden-ratio torus geometry.
    #region Chunk Computation

    /// <summary>
    /// Compute stereo gains for a chunk using toroidal panning + Rössler chaos.
    /// Writes left and right gain arrays.
    /// </summary>
    public static void ComputeChunk(ReadOnlySpan<double> tChunk,
        PanParams panParams, RosslerAttractor.Trajectory? rossler,
        Simplex5D? panSimplex,
        Span<float> gainLeft, Span<float> gainRight)
    {
        const float R = 1.0f;      // Major torus radius
        const float r = 0.382f;   // 1/PHI² — sacred minor radius
        float piOver4 = MathF.PI / 4.0f;

        for (int i = 0; i < tChunk.Length; i++)
        {
            double time = tChunk[i];

            // Base toroidal angles — double precision phase for long-session stability
            double theta = SacredConstants.TWO_PI_D * panParams.ThetaFreq * time;
            double phi = SacredConstants.TWO_PI_D * panParams.PhiFreq * time;

            // Rössler chaotic perturbation (±0.08 radians) — Interpolate takes float time
            if (rossler != null)
            {
                theta += 0.08f * RosslerAttractor.Interpolate(rossler.X, rossler.T, (float)time);
                phi += 0.08f * RosslerAttractor.Interpolate(rossler.Y, rossler.T, (float)time);
            }

            // Toroidal to stereo: map to horizontal pan and depth
            float panH = (float)((R + r * System.Math.Cos(phi)) * System.Math.Cos(theta) / (R + r));
            float depth = (float)((R + r * System.Math.Cos(phi)) / (R + r));

            // Stereo gains from pan position
            float leftGain = MathF.Cos((panH + 1.0f) * piOver4) * depth;
            float rightGain = MathF.Sin((panH + 1.0f) * piOver4) * depth;

            gainLeft[i] = leftGain;
            gainRight[i] = rightGain;
        }
    }

    #endregion
}
