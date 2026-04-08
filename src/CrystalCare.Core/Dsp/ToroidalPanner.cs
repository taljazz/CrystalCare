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
    /// <summary>
    /// Pre-computed toroidal panning parameters.
    /// </summary>
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

    /// <summary>
    /// Compute stereo gains for a chunk using toroidal panning + Rössler chaos.
    /// Writes left and right gain arrays.
    /// </summary>
    public static void ComputeChunk(ReadOnlySpan<float> tChunk,
        PanParams panParams, RosslerAttractor.Trajectory? rossler,
        Simplex5D? panSimplex,
        Span<float> gainLeft, Span<float> gainRight)
    {
        const float R = 1.0f;      // Major torus radius
        const float r = 0.382f;   // 1/PHI² — sacred minor radius
        float piOver4 = MathF.PI / 4.0f;

        for (int i = 0; i < tChunk.Length; i++)
        {
            float time = tChunk[i];

            // Base toroidal angles
            float theta = SacredConstants.TWO_PI * panParams.ThetaFreq * time;
            float phi = SacredConstants.TWO_PI * panParams.PhiFreq * time;

            // Rössler chaotic perturbation (±0.08 radians)
            if (rossler != null)
            {
                theta += 0.08f * RosslerAttractor.Interpolate(rossler.X, rossler.T, time);
                phi += 0.08f * RosslerAttractor.Interpolate(rossler.Y, rossler.T, time);
            }

            // Toroidal to stereo: map to horizontal pan and depth
            float panH = (R + r * MathF.Cos(phi)) * MathF.Cos(theta) / (R + r);
            float depth = (R + r * MathF.Cos(phi)) / (R + r);

            // Stereo gains from pan position
            float leftGain = MathF.Cos((panH + 1.0f) * piOver4) * depth;
            float rightGain = MathF.Sin((panH + 1.0f) * piOver4) * depth;

            gainLeft[i] = leftGain;
            gainRight[i] = rightGain;
        }
    }
}
