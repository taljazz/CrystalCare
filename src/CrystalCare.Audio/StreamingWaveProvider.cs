using System.Collections.Concurrent;
using NAudio.Wave;

namespace CrystalCare.Audio;

/// <summary>
/// NAudio IWaveProvider that pulls stereo float chunks from a BlockingCollection.
/// Used as the consumer in the producer-consumer streaming playback pattern.
///
/// The generator thread pushes chunks; NAudio's playback thread calls Read().
/// </summary>
public sealed class StreamingWaveProvider : IWaveProvider
{
    private readonly BlockingCollection<float[,]> _chunkQueue;
    private readonly CancellationToken _ct;

    // Current chunk being consumed
    private float[,]? _currentChunk;
    private int _currentSample;

    public WaveFormat WaveFormat { get; }

    public StreamingWaveProvider(BlockingCollection<float[,]> chunkQueue,
        int sampleRate, CancellationToken ct)
    {
        _chunkQueue = chunkQueue;
        _ct = ct;
        WaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 2);
    }

    /// <summary>
    /// Called by NAudio's playback thread to fill the audio buffer.
    /// Returns 0 when stream is complete.
    /// </summary>
    public int Read(byte[] buffer, int offset, int count)
    {
        int bytesWritten = 0;
        int bytesPerSample = 4; // float32
        int bytesPerFrame = bytesPerSample * 2; // stereo

        while (bytesWritten < count)
        {
            if (_ct.IsCancellationRequested)
                return 0;

            // Need a new chunk?
            if (_currentChunk == null || _currentSample >= _currentChunk.GetLength(0))
            {
                try
                {
                    if (!_chunkQueue.TryTake(out _currentChunk, 500, _ct))
                    {
                        // Timeout — if queue is completed, signal end of stream
                        if (_chunkQueue.IsCompleted)
                            return bytesWritten; // Return what we have, 0 next time
                        continue; // Try again
                    }
                }
                catch (OperationCanceledException)
                {
                    return 0;
                }
                catch (InvalidOperationException)
                {
                    // Queue was completed
                    return bytesWritten;
                }

                if (_currentChunk == null)
                    return bytesWritten;

                _currentSample = 0;
            }

            // Copy samples from current chunk to buffer
            int samplesAvailable = _currentChunk.GetLength(0) - _currentSample;
            int bytesRemaining = count - bytesWritten;
            int framesToCopy = Math.Min(samplesAvailable, bytesRemaining / bytesPerFrame);

            for (int i = 0; i < framesToCopy; i++)
            {
                int sampleIdx = _currentSample + i;
                int bufferPos = offset + bytesWritten + i * bytesPerFrame;

                // Left channel
                float left = _currentChunk[sampleIdx, 0];
                BitConverter.TryWriteBytes(buffer.AsSpan(bufferPos), left);

                // Right channel
                float right = _currentChunk[sampleIdx, 1];
                BitConverter.TryWriteBytes(buffer.AsSpan(bufferPos + bytesPerSample), right);
            }

            _currentSample += framesToCopy;
            bytesWritten += framesToCopy * bytesPerFrame;
        }

        return bytesWritten;
    }
}
