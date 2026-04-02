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
    /// <summary>
    /// Result of a Rössler trajectory computation.
    /// X and Y are used for stereo pan perturbation (±0.08 radians).
    /// </summary>
    public sealed class Trajectory
    {
        public required float[] X { get; init; }
        public required float[] Y { get; init; }
        public required float[] Z { get; init; }
        public required float[] T { get; init; }
    }

    /// <summary>
    /// Compute a low-rate Rössler trajectory for chaotic panning.
    /// Parameters match Python: a=0.2, b=0.2, c=5.7, time scaled by 0.1.
    /// </summary>
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

        // Initial conditions
        xArr[0] = 1.0;
        yArr[0] = 1.0;
        zArr[0] = 1.0;
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

    /// <summary>
    /// Interpolate trajectory value at an arbitrary time point.
    /// </summary>
    public static float Interpolate(float[] trajValues, float[] trajTimes, float time)
    {
        if (trajTimes.Length == 0) return 0f;
        if (time <= trajTimes[0]) return trajValues[0];
        if (time >= trajTimes[^1]) return trajValues[^1];

        // Binary search for the bracketing interval
        int lo = 0, hi = trajTimes.Length - 1;
        while (hi - lo > 1)
        {
            int mid = (lo + hi) / 2;
            if (trajTimes[mid] <= time) lo = mid;
            else hi = mid;
        }

        // Linear interpolation
        float frac = (time - trajTimes[lo]) / (trajTimes[hi] - trajTimes[lo]);
        return trajValues[lo] + frac * (trajValues[hi] - trajValues[lo]);
    }

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
}
