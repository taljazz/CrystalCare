namespace CrystalCare.Core.Generation;

/// <summary>
/// Pre-allocated buffer pool for the per-chunk pipeline loop.
/// Eliminates ~4 MB of float[] allocations per 3-second chunk,
/// saving ~2.4 GB of GC pressure over a 30-minute session.
///
/// Buffers are allocated once at the maximum chunk size and reused
/// each iteration. Call Clear() at the start of each chunk to zero
/// the active region before use.
/// </summary>
internal sealed class ChunkBufferPool
{
    // 7 pre-allocated float[] buffers reused each chunk iteration.
    // Allocated once at chunkSize; zeroed to activeLength each iteration via Clear().
    #region Pooled Buffers

    /// <summary>
    /// Time values for the current chunk, in seconds.
    /// Stored as double to maintain precision over long sessions (hours):
    /// float32 loses sub-millisecond resolution above ~1 hour of playback,
    /// causing staircase quantization in sine generation.
    /// </summary>
    public double[] TChunk { get; }

    /// <summary>ADSR envelope values.</summary>
    public float[] Envelope { get; }

    /// <summary>Delayed right channel for stereo widening.</summary>
    public float[] WaveRightDelayed { get; }

    /// <summary>Offset time array for right-channel noise (double for long-session precision).</summary>
    public double[] TChunkOffset { get; }

    /// <summary>Pan curve for toroidal panning.</summary>
    public float[] PanCurve { get; }

    /// <summary>Scaled time array 1 for panning simplex noise.</summary>
    public float[] PanTScaled1 { get; }

    /// <summary>Scaled time array 2 for panning simplex noise.</summary>
    public float[] PanTScaled2 { get; }

    #endregion

    // Allocates all 7 buffers at chunkSize. Clear() zeros each buffer up to
    // activeLength at the start of each chunk (last chunk may be shorter).
    #region Constructor and Clear

    public ChunkBufferPool(int chunkSize)
    {
        // Time arrays use double for long-session precision
        TChunk = new double[chunkSize];
        TChunkOffset = new double[chunkSize];

        // Signal/envelope arrays stay as float (no precision-over-time issue)
        Envelope = new float[chunkSize];
        WaveRightDelayed = new float[chunkSize];
        PanCurve = new float[chunkSize];
        PanTScaled1 = new float[chunkSize];
        PanTScaled2 = new float[chunkSize];
    }

    /// <summary>
    /// Clear all buffers up to the active length for this chunk.
    /// The last chunk may be shorter than the full buffer size.
    /// </summary>
    public void Clear(int activeLength)
    {
        Array.Clear(TChunk, 0, activeLength);
        Array.Clear(Envelope, 0, activeLength);
        Array.Clear(WaveRightDelayed, 0, activeLength);
        Array.Clear(TChunkOffset, 0, activeLength);
        Array.Clear(PanCurve, 0, activeLength);
        Array.Clear(PanTScaled1, 0, activeLength);
        Array.Clear(PanTScaled2, 0, activeLength);
    }

    #endregion
}
