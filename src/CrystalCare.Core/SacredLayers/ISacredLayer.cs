namespace CrystalCare.Core.SacredLayers;

/// <summary>
/// Interface for all sacred healing layers.
/// Each layer produces a mono signal that gets toroidally panned into stereo.
/// All chunk methods are stateless — they use absolute time values.
/// </summary>
public interface ISacredLayer
{
    /// <summary>
    /// Compute sacred layer audio for a time chunk.
    /// </summary>
    /// <param name="tChunk">Absolute time values for this chunk (seconds)</param>
    /// <param name="totalDuration">Total session duration in seconds</param>
    /// <returns>Mono float array of same length as tChunk</returns>
    float[] ComputeChunk(ReadOnlySpan<float> tChunk, float totalDuration);
}
