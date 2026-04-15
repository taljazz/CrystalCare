using CrystalCare.Core.Dsp;
using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Noise;
using CrystalCare.Core.SacredLayers;
using Math = CrystalCare.Core.Math;

namespace CrystalCare.Core.Generation;

/// <summary>
/// Main sound generator — orchestrates the 16-stage audio pipeline.
/// Two modes: streaming (GenerateStream) for playback, batch (GenerateBatch) for saves.
/// Port of SoundGenerator class from SoundGenerator.py.
/// </summary>
public sealed partial class SoundGenerator
{
    private readonly FrequencyManager _frequencyManager;
    private readonly CrystalProfileLibrary _crystalLibrary;
    private readonly ChaoticSelector _chaoticSelector = new();
    private readonly Random _rng = new();

    public float MasterVolume { get; set; } = 0.52f;

    public SoundGenerator(FrequencyManager frequencyManager)
    {
        _frequencyManager = frequencyManager;
        _crystalLibrary = new CrystalProfileLibrary();
    }

    /// <summary>
    /// Streaming generator: yields (chunkSamples, 2) float stereo chunks.
    /// Constant ~50 MB memory regardless of duration.
    /// This is the heart of CrystalCare — the 16-stage pipeline.
    /// </summary>
    public IEnumerable<float[,]> GenerateStream(float duration, float baseFreq,
        int sampleRate = 48000,
        int[]? intervalDurationList = null,
        CancellationToken ct = default,
        Action<float>? updateProgress = null,
        FrequencyMode freqMode = FrequencyMode.Standard)
    {
        intervalDurationList ??= [34, 55, 89, 144];
        bool dimensionalMode = freqMode == FrequencyMode.DimensionalShift;
        int totalSamples = (int)(sampleRate * duration);
        int chunkSize = 3 * sampleRate; // 3-second chunks

        // ========== PRE-COMPUTATION PHASE ==========

        // --- Modulation schedule ---
        var schedule = BuildModulationSchedule(duration, totalSamples, sampleRate,
            intervalDurationList, dimensionalMode, ct);
        if (ct.IsCancellationRequested) yield break;

        // --- Random parameters (drawn once) ---
        float chaoticFactor = _chaoticSelector.NextValue();
        float chaosVal = _chaoticSelector.NextValue();

        // ADSR parameters
        var adsrParams = OrganicAdsrEnvelope.ComputeParams(totalSamples, sampleRate,
            _chaoticSelector, _rng);

        // LFO parameters
        var lfoLeftParams = MicrotonalLfo.DrawParams(0.05f, _rng);
        var lfoRightParams = MicrotonalLfo.DrawParams(0.05f, _rng);

        // Toroidal panning parameters
        float torusThetaFreq = (float)(_rng.NextDouble() * 0.01 + 0.01);
        float torusPhiFreq = torusThetaFreq * SacredConstants.PHI;
        const float torusR = 0.618f;      // 1/PHI — golden ratio reciprocal
        const float torusRSmall = 0.382f;  // 1/PHI² — two halves sum to 1.0

        // Dedicated Simplex instances
        var simplexEnvelope = new Simplex5D(42);
        var simplexFractal = new Simplex5D(7);
        var simplexPan = new Simplex5D(13);

        // Taygetan binaural mode detection
        bool hasTaygetan = freqMode == FrequencyMode.TaygetanBinaural;
        var tayFreqResult = hasTaygetan ? _frequencyManager.GetFrequencies(FrequencyMode.TaygetanBinaural) : null;
        var tayPairs = tayFreqResult?.BinauralPairs;

        // Noise and fade parameters
        float noiseScaleLeft = (float)(_rng.NextDouble() * 0.02 + 0.04);
        float noiseScaleRight = (float)(_rng.NextDouble() * 0.02 + 0.04);
        float noiseOffsetRight = (float)(_rng.NextDouble() * 0.014 + 0.001);
        int fadeDuration = _rng.Next(2) == 0 ? 21 : 34; // Fibonacci pair
        int fadeSamples = fadeDuration * sampleRate;
        if (2 * fadeSamples > totalSamples) fadeSamples = totalSamples / 2;

        // Pan drift
        float driftFreq = (float)(_rng.NextDouble() * 0.0015 + 0.0005);
        float driftAmplitude = (float)(_rng.NextDouble() * 0.01 + 0.01);

        // Frequency sets: 6 PHI exponents + 3 ratio-1.3 exponents + 4 subharmonics = 13
        var allFrequencies = BuildFrequencySet(baseFreq);
        var modDepths = new float[allFrequencies.Length];
        for (int i = 0; i < modDepths.Length; i++)
            modDepths[i] = (float)(_rng.NextDouble() * 0.2 + 0.15);

        // --- IIR low-pass filter (order 2 Butterworth, ~5500 Hz cutoff) ---
        // Gentler than the original order 4 at 2200 Hz — lets upper harmonics through
        // for brighter, more open tones while still smoothing harsh digital artifacts.
        // Order 2 = 12 dB/octave rolloff (vs 24 dB/octave at order 4).
        float cutoffVariation = (float)(_rng.NextDouble() * 50 - 25);
        float lpCutoff = global::System.Math.Clamp(5500f + cutoffVariation, 2000f, 23000f);
        var filterLeft = BiquadFilter.CreateLowpass(2, lpCutoff, sampleRate);
        var filterRight = BiquadFilter.CreateLowpass(2, lpCutoff, sampleRate);

        // Pan curve filter — 0.002 Hz exponential smoother for ultra-slow panning
        // This makes the toroidal panning extremely smooth and organic — only the
        // slowest drift survives. Matches Python: butter(2, 0.002 / Nyquist).
        // ExponentialSmoother handles extreme cutoffs that Butterworth cannot.
        var panSmoother = new ExponentialSmoother(0.002f, sampleRate);

        // --- PHI-fractal feedback instances (carry echo tail between chunks) ---
        var phiFractalLeft = new PhiFractalFeedback(sampleRate);
        var phiFractalRight = new PhiFractalFeedback(sampleRate);

        // --- Reverb instances ---
        var reverbLeft = new StreamingReverb(sampleRate);
        var reverbRight = new StreamingReverb(sampleRate);

        // --- 12-sample right-channel delay buffer ---
        var delayBuffer = new float[12];

        // --- Rössler trajectory ---
        var rossler = Math.RosslerAttractor.Compute(duration);

        // --- Sacred layers ---
        // Crystal sequence: Lemurian Quartz always first
        int lemIdx = _crystalLibrary.LemurianIndex;
        var rest = Enumerable.Range(0, _crystalLibrary.Profiles.Length)
            .Where(i => i != lemIdx).OrderBy(_ => _rng.Next()).ToArray();
        var crystalSequence = new[] { lemIdx }.Concat(rest).ToArray();

        ISacredLayer[] sacredLayers =
        [
            new PleromaMercyLayer(),
            new SilentSolfeggioGrid(),
            new ArchonDissolutionLayer(),
            new CrystallineResonanceLayer(_crystalLibrary, crystalSequence, baseFreq),
            new LemurianMerkabaLayer(),
            new WaterElementLayer(),
        ];

        // Sacred layer toroidal panning frequencies (slow, independent per layer)
        float[] sacredThetaFreqs = [0.003f, 0.005f, 0.004f, 0.0035f, 0.0028f, 0.0045f];
        float[] sacredPhiFreqs = sacredThetaFreqs.Select(f => f * SacredConstants.PHI).ToArray();
        const float sacredR = 0.618f;      // 1/PHI — matches main torus
        const float sacredRSmall = 0.382f;  // 1/PHI² — two halves sum to 1.0

        // --- Normalization: estimate peak ---
        float normGain = EstimateNormGain(allFrequencies, modDepths, lfoLeftParams, sampleRate);

        // ========== PRE-ALLOCATE BUFFER POOL ==========
        var pool = new ChunkBufferPool(chunkSize);

        // ========== PER-CHUNK LOOP ==========
        int numChunks = (totalSamples + chunkSize - 1) / chunkSize;

        for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++)
        {
            if (ct.IsCancellationRequested) yield break;

            int chunkOffset = chunkIdx * chunkSize;
            int chunkSamples = global::System.Math.Min(chunkSize, totalSamples - chunkOffset);

            // Clear pooled buffers for this chunk
            pool.Clear(chunkSamples);

            // Time array for this chunk
            float tStart = (float)chunkOffset / sampleRate;
            var tChunk = pool.TChunk;
            for (int i = 0; i < chunkSamples; i++)
                tChunk[i] = tStart + (float)i / sampleRate;

            // --- Stage 1: Modulation ---
            var modulation = ComputeModulationChunk(tChunk, chunkOffset, chunkSamples, schedule);

            // --- Stage 2: Fractal variation ---
            var fractalVar = FractalVariation.ComputeChunkDual(tChunk, baseFreq, simplexFractal);

            // --- Stage 3: f_modulated ---
            float chaoticOffset = chaoticFactor * baseFreq * 0.25f;
            for (int i = 0; i < chunkSamples; i++)
                modulation[i] += baseFreq + chaoticOffset + fractalVar[i];

            // --- Stage 4: Envelope ---
            var envelope = pool.Envelope;
            OrganicAdsrEnvelope.ComputeChunk(envelope, chunkOffset, chunkSamples,
                adsrParams, simplexEnvelope, totalSamples);

            // --- Stage 5: LFOs ---
            var lfoLeft = MicrotonalLfo.Compute(tChunk, lfoLeftParams);
            var lfoRight = MicrotonalLfo.Compute(tChunk, lfoRightParams);

            // --- Stage 6: Harmonics ---
            var waveLeft = HarmonicGenerator.GenerateHarmonics(
                tChunk, allFrequencies, envelope, lfoLeft, modDepths);
            var waveRight = HarmonicGenerator.GenerateHarmonics(
                tChunk, allFrequencies, envelope, lfoRight, modDepths);

            // --- Stage 7: Taygetan binaural beats ---
            if (hasTaygetan && tayPairs != null)
            {
                var binauralWaves = BinauralOscillator.BatchGenerate(tChunk, tayPairs, ct);
                for (int p = 0; p < binauralWaves.Length; p++)
                {
                    for (int i = 0; i < chunkSamples; i++)
                    {
                        waveLeft[i] += binauralWaves[p][i, 0] * envelope[i] * 0.015f;
                        waveRight[i] += binauralWaves[p][i, 1] * envelope[i] * 0.015f;
                    }
                }
            }

            // --- Stage 8: Wave shaping ---
            WaveShaper.Process(waveLeft, 2.5f);
            WaveShaper.Process(waveRight, 2.5f);

            // --- Stage 8b: PHI-fractal feedback (stateful — carries echo tail between chunks) ---
            waveLeft = phiFractalLeft.ProcessChunk(waveLeft);
            waveRight = phiFractalRight.ProcessChunk(waveRight);

            // --- Stage 9: Low-pass filter with zi carry ---
            filterLeft.Process(waveLeft);
            filterRight.Process(waveRight);

            // --- Stage 10: Normalization ---
            for (int i = 0; i < chunkSamples; i++)
            {
                waveLeft[i] *= normGain;
                waveRight[i] *= normGain;
            }

            // --- Stage 11: Reverb with tail carry ---
            waveLeft = reverbLeft.ProcessChunk(waveLeft);

            // --- Stage 12: 12-sample right delay with carry ---
            var waveRightDelayed = pool.WaveRightDelayed;
            Array.Copy(delayBuffer, 0, waveRightDelayed, 0,
                global::System.Math.Min(12, chunkSamples));
            if (chunkSamples > 12)
                Array.Copy(waveRight, 0, waveRightDelayed, 12, chunkSamples - 12);
            Array.Copy(waveRight, global::System.Math.Max(0, chunkSamples - 12), delayBuffer, 0,
                global::System.Math.Min(12, chunkSamples));
            waveRight = reverbRight.ProcessChunk(waveRightDelayed);

            // --- Stage 13: Noise layers ---
            var noiseLeft = EvolvingNoiseLayer.Generate(tChunk, rng: _rng);
            var tChunkOffset = pool.TChunkOffset;
            for (int i = 0; i < chunkSamples; i++)
                tChunkOffset[i] = tChunk[i] + noiseOffsetRight;
            var noiseRight = EvolvingNoiseLayer.Generate(tChunkOffset, rng: _rng);

            for (int i = 0; i < chunkSamples; i++)
            {
                waveLeft[i] = (waveLeft[i] + noiseLeft[i]) * noiseScaleLeft;
                waveRight[i] = (waveRight[i] + noiseRight[i]) * noiseScaleRight;
            }

            // --- Stage 14: Fade in/out ---
            ApplyFade(waveLeft, waveRight, chunkOffset, chunkSamples, fadeSamples, totalSamples);

            // --- Stage 15: Toroidal pan + Rössler ---
            ApplyToroidalPanning(waveLeft, waveRight, tChunk, chunkSamples,
                torusThetaFreq, torusPhiFreq, torusR, torusRSmall,
                simplexPan, rossler, panSmoother, driftFreq, driftAmplitude, pool);

            // Apply master volume
            for (int i = 0; i < chunkSamples; i++)
            {
                waveLeft[i] *= MasterVolume;
                waveRight[i] *= MasterVolume;
            }

            // --- Stage 16: Sacred layers (parallel computation) ---
            if (duration > 60 || dimensionalMode)
            {
                // Launch all 6 sacred layers in parallel
                var sacredTasks = new Task<float[]>[sacredLayers.Length];
                for (int li = 0; li < sacredLayers.Length; li++)
                {
                    var layer = sacredLayers[li];
                    sacredTasks[li] = Task.Run(() => layer.ComputeChunk(tChunk, duration));
                }

                try
                {
                    Task.WaitAll(sacredTasks, TimeSpan.FromSeconds(60));
                }
                catch (AggregateException) { }

                // Apply independent toroidal panning per sacred layer
                for (int li = 0; li < sacredLayers.Length; li++)
                {
                    float[]? layerData = sacredTasks[li].IsCompletedSuccessfully
                        ? sacredTasks[li].Result
                        : null;
                    if (layerData == null) continue;

                    float stFreq = sacredThetaFreqs[li];
                    float spFreq = sacredPhiFreqs[li];

                    for (int i = 0; i < chunkSamples; i++)
                    {
                        float sTheta = SacredConstants.TWO_PI * stFreq * tChunk[i];
                        float sPhi = SacredConstants.TWO_PI * spFreq * tChunk[i];
                        float sacredPan = (sacredR + sacredRSmall * MathF.Cos(sPhi)) *
                            MathF.Cos(sTheta) / (sacredR + sacredRSmall);
                        float panScaled = sacredPan * 0.4f;
                        waveLeft[i] += layerData[i] * (1.0f - panScaled);
                        waveRight[i] += layerData[i] * (1.0f + panScaled);
                    }
                }
            }

            // Build stereo chunk and yield
            var stereoChunk = new float[chunkSamples, 2];
            for (int i = 0; i < chunkSamples; i++)
            {
                stereoChunk[i, 0] = waveLeft[i];
                stereoChunk[i, 1] = waveRight[i];
            }

            updateProgress?.Invoke((float)(chunkIdx + 1) / numChunks);
            yield return stereoChunk;
        }
    }

    /// <summary>
    /// Batch mode: collect all chunks into a single array (for WAV saves).
    /// </summary>
    public float[,] GenerateBatch(float duration, float baseFreq,
        int sampleRate = 48000,
        CancellationToken ct = default,
        Action<float>? updateProgress = null,
        FrequencyMode freqMode = FrequencyMode.Standard)
    {
        int totalSamples = (int)(sampleRate * duration);
        var result = new float[totalSamples, 2];
        int offset = 0;

        foreach (var chunk in GenerateStream(duration, baseFreq, sampleRate,
            ct: ct, updateProgress: updateProgress,
            freqMode: freqMode))
        {
            int samples = chunk.GetLength(0);
            int toCopy = global::System.Math.Min(samples, totalSamples - offset);
            for (int i = 0; i < toCopy; i++)
            {
                result[offset + i, 0] = chunk[i, 0];
                result[offset + i, 1] = chunk[i, 1];
            }
            offset += toCopy;
        }

        return result;
    }

}
