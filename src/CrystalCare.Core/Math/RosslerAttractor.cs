namespace CrystalCare.Core.Math;

/// <summary>
/// Rössler attractor ODE solver for chaotic panning trajectories.
/// Uses 4th-order Runge-Kutta integration.
///
/// The Rössler system: dx/dt = -y - z, dy/dt = x + 0.2y, dz/dt = 0.2 + z(x - 5.7)
/// Produces deterministic chaotic trajectories from fixed initial conditions.
///
/// Port of AudioProcessor.compute_rossler_trajectory() from SoundGenerator.py.
/// </summary>
public static class RosslerAttractor
{
    // Result container for the Rössler trajectory. X and Y components drive
    // spatial panning perturbation (±0.08 radians). Z is computed but unused.
    #region Trajectory Data

    public sealed class Trajectory
    {
        public required float[] X { get; init; }
        public required float[] Y { get; init; }
        public required float[] Z { get; init; }
        public required float[] T { get; init; }
    }

    #endregion

    // 4th-order Runge-Kutta integration of the Rössler ODE system.
    // Classic parameters: a=0.2, b=0.2, c=5.7. PHI-derived initial conditions.
    // Time scaled by 0.1 for slow, organic chaos trajectories.
    // Output normalized to [-1, 1] range for direct use as panning perturbation.
    #region RK4 Integration

    public static Trajectory Compute(float duration, float rate = 10f)
    {
        int nSamples = (int)(duration * rate);
        if (nSamples < 2)
        {
            var z = new float[2];
            return new Trajectory { X = z, Y = new float[2], Z = new float[2], T = [0, duration] };
        }

        // RK4 integration
        const double a = 0.2, b = 0.2, c = 5.7;
        double dt = (duration * 0.1) / (nSamples - 1); // time scaled by 0.1

        var xArr = new double[nSamples];
        var yArr = new double[nSamples];
        var zArr = new double[nSamples];
        var tArr = new float[nSamples];

        // Initial conditions — PHI-derived sacred genesis
        xArr[0] = Frequencies.SacredConstants.PHI;               // φ
        yArr[0] = 1.0;                                           // Unity
        zArr[0] = Frequencies.SacredConstants.PHI_INVERSE;       // 1/φ
        tArr[0] = 0f;

        for (int i = 1; i < nSamples; i++)
        {
            double x0 = xArr[i - 1], y0 = yArr[i - 1], z0 = zArr[i - 1];

            // k1
            double k1x = -y0 - z0;
            double k1y = x0 + a * y0;
            double k1z = b + z0 * (x0 - c);

            // k2
            double x1 = x0 + 0.5 * dt * k1x;
            double y1 = y0 + 0.5 * dt * k1y;
            double z1 = z0 + 0.5 * dt * k1z;
            double k2x = -y1 - z1;
            double k2y = x1 + a * y1;
            double k2z = b + z1 * (x1 - c);

            // k3
            double x2 = x0 + 0.5 * dt * k2x;
            double y2 = y0 + 0.5 * dt * k2y;
            double z2 = z0 + 0.5 * dt * k2z;
            double k3x = -y2 - z2;
            double k3y = x2 + a * y2;
            double k3z = b + z2 * (x2 - c);

            // k4
            double x3 = x0 + dt * k3x;
            double y3 = y0 + dt * k3y;
            double z3 = z0 + dt * k3z;
            double k4x = -y3 - z3;
            double k4y = x3 + a * y3;
            double k4z = b + z3 * (x3 - c);

            xArr[i] = x0 + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x);
            yArr[i] = y0 + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y);
            zArr[i] = z0 + dt / 6.0 * (k1z + 2 * k2z + 2 * k3z + k4z);
            tArr[i] = duration * (float)i / (nSamples - 1);
        }

        // Normalize to [-1, 1]
        return new Trajectory
        {
            X = NormalizeToFloat32(xArr),
            Y = NormalizeToFloat32(yArr),
            Z = NormalizeToFloat32(zArr),
            T = tArr,
        };
    }

    #endregion

    // Cursor-based linear interpolation to look up trajectory values at
    // monotonically increasing time points. Used by the pipeline to get chaos
    // values at each sample.
    //
    // The pipeline queries time in strictly ascending order within a chunk, so
    // a walking cursor (`hint`) replaces the previous per-sample binary search:
    // the bracketing interval advances at most a step or two per sample, making
    // each lookup O(1) instead of O(log n). The bracket found is the same one
    // binary search found (last index with trajTimes[i] <= time), so the
    // interpolated output is bit-identical.
    #region Interpolation

    /// <summary>
    /// Interpolate the trajectory at the given time, using and updating a
    /// caller-held cursor. Initialize the cursor to 0 at the start of each
    /// chunk; pass the same variable for every sample of the chunk so the
    /// bracket walks forward instead of re-searching.
    /// </summary>
    /// <param name="hint">
    /// Bracketing-interval cursor. In: where to start walking. Out: the interval
    /// containing this time (reusable for the next, later, time value).
    /// </param>
    public static float Interpolate(float[] trajValues, float[] trajTimes, float time,
        ref int hint)
    {
        if (trajTimes.Length == 0) return 0f;
        if (time <= trajTimes[0]) { hint = 0; return trajValues[0]; }
        if (time >= trajTimes[^1]) { hint = trajTimes.Length - 2; return trajValues[^1]; }

        // Clamp the cursor into the valid interval range
        if (hint < 0) hint = 0;
        if (hint > trajTimes.Length - 2) hint = trajTimes.Length - 2;

        // Walk backward if the cursor overshot (defensive — time is monotonic
        // in the pipeline so this loop almost never runs)
        while (hint > 0 && trajTimes[hint] > time) hint--;

        // Advance forward to the bracketing interval: the last index with
        // trajTimes[hint] <= time — the same bracket binary search produced
        while (hint < trajTimes.Length - 2 && trajTimes[hint + 1] <= time) hint++;

        // Linear interpolation within the bracket
        float frac = (time - trajTimes[hint]) / (trajTimes[hint + 1] - trajTimes[hint]);
        return trajValues[hint] + frac * (trajValues[hint + 1] - trajValues[hint]);
    }

    #endregion

    // Normalizes double[] trajectory to float[] in [-1, 1] range.
    // Divides by peak absolute value for uniform amplitude.
    #region Normalization

    private static float[] NormalizeToFloat32(double[] arr)
    {
        double maxAbs = 0;
        for (int i = 0; i < arr.Length; i++)
            maxAbs = global::System.Math.Max(maxAbs, global::System.Math.Abs(arr[i]));
        if (maxAbs < 1e-10) maxAbs = 1e-10;

        var result = new float[arr.Length];
        for (int i = 0; i < arr.Length; i++)
            result[i] = (float)(arr[i] / maxAbs);
        return result;
    }

    #endregion
}
