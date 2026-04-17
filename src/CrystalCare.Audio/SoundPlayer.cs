using System.Collections.Concurrent;
using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Generation;
using NAudio.Wave;

namespace CrystalCare.Audio;

/// <summary>
/// Audio playback and WAV file saving.
/// Streaming playback uses producer-consumer pattern with NAudio's WasapiOut.
/// Port of SoundPlayer class from SoundManager.py.
/// </summary>
public sealed class SoundPlayer : IDisposable
{
    // Holds the sound generator (produces audio chunks) and the NAudio
    // wave output device (plays them through speakers).
    #region Fields and Constructor

    // The 16-stage pipeline sound generator — produces stereo float chunks
    private readonly SoundGenerator _generator;

    // NAudio wave output device — plays audio through the system's default output
    private IWavePlayer? _waveOut;

    public SoundPlayer(SoundGenerator generator)
    {
        _generator = generator;
    }

    #endregion

    // Real-time streaming playback using producer-consumer pattern.
    // A background thread generates audio chunks and pushes them to a BlockingCollection.
    // NAudio's playback thread pulls chunks via StreamingWaveProvider.
    // Audio starts playing within ~0.2 seconds of pressing Play.
    #region Streaming Playback

    /// <summary>
    /// Stream audio to speakers using producer-consumer pattern.
    /// Audio starts playing as soon as the first chunk is generated (~0.2s).
    /// </summary>
    public async Task PlayStreamAsync(float duration, float baseFreq,
        int sampleRate, CancellationToken ct,
        Action<string>? updateStatus = null,
        Action<float>? updateProgress = null,
        FrequencyMode freqMode = FrequencyMode.Standard,
        int deviceNumber = -1)
    {
        var chunkQueue = new BlockingCollection<float[,]>(boundedCapacity: 4);

        // Producer: generate audio chunks on background thread
        var producerTask = Task.Run(() =>
        {
            try
            {
                foreach (var chunk in _generator.GenerateStream(
                    duration, baseFreq, sampleRate,
                    ct: ct, updateProgress: updateProgress,
                    freqMode: freqMode))
                {
                    if (ct.IsCancellationRequested) break;
                    chunkQueue.Add(chunk, ct);
                }
            }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                updateStatus?.Invoke($"Generation error: {ex.Message}");
            }
            finally
            {
                chunkQueue.CompleteAdding();
            }
        }, ct);

        // Consumer: NAudio playback
        var provider = new StreamingWaveProvider(chunkQueue, sampleRate, ct);

        try
        {
            _waveOut = new WaveOutEvent
            {
                DeviceNumber = deviceNumber,
                DesiredLatency = 200,
                NumberOfBuffers = 3,
            };
            _waveOut.Init(provider);
            _waveOut.Play();

            updateStatus?.Invoke("Playing...");

            // Wait for producer to finish
            await producerTask;

            // Wait for audio buffer to drain
            while (_waveOut.PlaybackState == PlaybackState.Playing && !ct.IsCancellationRequested)
                await Task.Delay(50, ct);

            if (!ct.IsCancellationRequested)
                updateStatus?.Invoke("Playback stream finished.");
        }
        catch (OperationCanceledException) { }
        finally
        {
            _waveOut?.Stop();
            _waveOut?.Dispose();
            _waveOut = null;
        }
    }

    #endregion

    // WAV file saving — generates the full session in memory using batch mode,
    // finds the global peak for normalization, then writes 16-bit PCM WAV
    // in 10-second chunks via NAudio. Progress: 80% generation, 20% writing.
    #region WAV File Saving

    /// <summary>
    /// Save audio to WAV file using batch generation.
    /// Two-pass: generate full array, find peak, write normalized int16.
    /// </summary>
    public async Task SaveToWavAsync(string filename, float duration, float baseFreq,
        int sampleRate, CancellationToken ct,
        Action<string>? updateStatus = null,
        Action<float>? updateProgress = null,
        FrequencyMode freqMode = FrequencyMode.Standard)
    {
        await Task.Run(() =>
        {
            updateStatus?.Invoke("Generating audio...");

            var audioData = _generator.GenerateBatch(duration, baseFreq, sampleRate,
                ct: ct, updateProgress: p => updateProgress?.Invoke(p * 0.8f),
                freqMode: freqMode);

            if (ct.IsCancellationRequested) return;

            updateStatus?.Invoke("Saving WAV file...");

            // Find global peak for normalization
            int totalSamples = audioData.GetLength(0);
            float peak = 0.001f;
            for (int i = 0; i < totalSamples; i++)
            {
                peak = MathF.Max(peak, MathF.Abs(audioData[i, 0]));
                peak = MathF.Max(peak, MathF.Abs(audioData[i, 1]));
            }

            float scaleFactor = 32767.0f / peak;

            // Write WAV in chunks using NAudio
            var format = new WaveFormat(sampleRate, 16, 2);
            using var writer = new WaveFileWriter(filename, format);

            int chunkSize = sampleRate * 10; // 10-second chunks
            var int16Buffer = new short[chunkSize * 2]; // stereo

            for (int start = 0; start < totalSamples; start += chunkSize)
            {
                if (ct.IsCancellationRequested) break;

                int end = Math.Min(start + chunkSize, totalSamples);
                int count = end - start;

                for (int i = 0; i < count; i++)
                {
                    int16Buffer[i * 2] = (short)(audioData[start + i, 0] * scaleFactor);
                    int16Buffer[i * 2 + 1] = (short)(audioData[start + i, 1] * scaleFactor);
                }

                writer.WriteSamples(int16Buffer, 0, count * 2);
                updateProgress?.Invoke(0.8f + 0.2f * (float)end / totalSamples);
            }

            updateStatus?.Invoke($"Saved: {filename}");
        }, ct);
    }

    #endregion

    // Stop playback and dispose NAudio resources.
    #region Playback Control and Disposal

    /// <summary>
    /// Stop current playback immediately.
    /// </summary>
    public void StopPlayback()
    {
        try
        {
            _waveOut?.Stop();
        }
        catch { }
    }

    public void Dispose()
    {
        _waveOut?.Stop();
        _waveOut?.Dispose();
    }

    #endregion
}
