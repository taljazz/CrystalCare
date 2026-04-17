using CrystalCare.Core.Dsp;
using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Noise;
using CrystalCare.Core.SacredLayers;
using Math = CrystalCare.Core.Math;

namespace CrystalCare.Core.Generation;

/// <summary>
/// Main sound generator — orchestrates the 16-stage audio pipeline.
/// Two modes: streaming (GenerateStream) for real-time playback,
/// batch (GenerateBatch) for WAV file saves.
/// The streaming generator yields 3-second stereo chunks via producer-consumer pattern.
/// The batch generator wraps streaming and collects all chunks into one array.
/// </summary>
public sealed partial class SoundGenerator
{
    // Core services and state used across the entire pipeline.
    // FrequencyManager provides ratio sets and frequency lookups.
    // CrystalProfileLibrary holds the 9 crystal harmonic profiles (Raman spectroscopy data).
    // ChaoticSelector produces logistic-map chaos for organic frequency variation.
    // MasterVolume (0.52, digits sum to 7) is applied after all processing.
    #region Fields and Properties

    // Manages sacred geometry ratio sets and frequency mode lookups
    private readonly FrequencyManager _frequencyManager;

    // 9 crystal profiles with Raman harmonic ratios for the Crystalline Resonance Layer
    private readonly CrystalProfileLibrary _crystalLibrary;

    // Logistic map chaotic number generator — produces deterministic but chaotic values
    private readonly ChaoticSelector _chaoticSelector = new();

    // Random number generator for all stochastic parameters (drawn once at pipeline start)
    private readonly Random _rng = new();

    // Master output volume — 0.43 (digits sum to 7, sacred number).
    // Applied after all 16 pipeline stages and sacred layers.
    public float MasterVolume { get; set; } = 0.43f;

    #endregion

    // Creates the sound generator with a frequency manager for ratio lookups
    // and initializes the crystal profile library with all 9 crystal types.
    #region Constructor

    public SoundGenerator(FrequencyManager frequencyManager)
    {
        _frequencyManager = frequencyManager;
        _crystalLibrary = new CrystalProfileLibrary();
    }

    #endregion

    // The heart of CrystalCare — the 16-stage streaming audio pipeline.
    // Yields 3-second stereo float chunks for real-time playback.
    // Constant ~50 MB memory regardless of session duration.
    // All random parameters are drawn once before the loop for consistency.
    #region Streaming Generator

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
        // Default modulation intervals are Fibonacci numbers
        intervalDurationList ??= [34, 55, 89, 144];

        // Dimensional Shift mode activates sacred layers regardless of duration
        bool dimensionalMode = freqMode == FrequencyMode.DimensionalShift;
        int totalSamples = (int)(sampleRate * duration);
        int chunkSize = 3 * sampleRate; // 3-second chunks (144,000 samples at 48kHz)

        // All parameters in this region are computed once before the chunk loop begins.
        // This ensures consistency across the entire session — the same random seed,
        // the same filter coefficients, the same chaos trajectory from start to end.
        #region Pre-Computation Phase

        // Build the modulation schedule: which ratio sets play when, at what depth
        var schedule = BuildModulationSchedule(duration, totalSamples, sampleRate,
            intervalDurationList, dimensionalMode, ct);
        if (ct.IsCancellationRequested) yield break;

        // Draw chaotic values from the logistic map for organic frequency variation
        float chaoticFactor = _chaoticSelector.NextValue();
        float chaosVal = _chaoticSelector.NextValue();

        // Pre-compute ADSR envelope parameters (attack/decay/sustain/release durations)
        var adsrParams = OrganicAdsrEnvelope.ComputeParams(totalSamples, sampleRate,
            _chaoticSelector, _rng);

        // Draw LFO parameters for left and right channels (PHI-modulated microtonal drift).
        // Base frequency = BREATH_ROOT × PHI^4 — the breath ladder extended 4 PHI steps,
        // connecting LFO amplitude modulation to the same Schumann root as the breath.
        var lfoLeftParams = MicrotonalLfo.DrawParams(SacredConstants.LFO_BASE_FREQ, _rng);
        var lfoRightParams = MicrotonalLfo.DrawParams(SacredConstants.LFO_BASE_FREQ, _rng);

        // Toroidal panning — sound traces a donut-shaped path in stereo space.
        // Theta frequency is randomized; phi frequency is theta * PHI for golden relationship.
        // Radii are PHI-derived: 0.618 + 0.382 = 1.0 (two halves summing to unity).
        float torusThetaFreq = (float)(_rng.NextDouble() * 0.01 + 0.01);
        float torusPhiFreq = torusThetaFreq * SacredConstants.PHI;
        const float torusR = 0.618f;      // 1/PHI — golden ratio reciprocal (major radius)
        const float torusRSmall = 0.382f;  // 1/PHI² — golden ratio squared reciprocal (minor radius)

        // Dedicated Simplex noise instances for different pipeline stages.
        // Seeds are Fibonacci numbers for sacred consistency (21, 7, 13).
        var simplexEnvelope = new Simplex5D(21);   // ADSR envelope organic variation (Fibonacci)
        var simplexFractal = new Simplex5D(7);     // Fractal frequency variation (Fibonacci)
        var simplexPan = new Simplex5D(13);        // Toroidal panning perturbation (Fibonacci)

        // Check if Taygetan binaural mode is active — requires stereo frequency pairs
        bool hasTaygetan = freqMode == FrequencyMode.TaygetanBinaural;
        var tayFreqResult = hasTaygetan ? _frequencyManager.GetFrequencies(FrequencyMode.TaygetanBinaural) : null;
        var tayPairs = tayFreqResult?.BinauralPairs;

        // Randomized noise and fade parameters for organic uniqueness each session.
        // Noise scale bounded by Fibonacci reciprocals: 1/21 to 1/13.
        float noiseScaleLeft = SacredConstants.NOISE_SCALE_MIN +
            (float)_rng.NextDouble() * (SacredConstants.NOISE_SCALE_MAX - SacredConstants.NOISE_SCALE_MIN);
        float noiseScaleRight = SacredConstants.NOISE_SCALE_MIN +
            (float)_rng.NextDouble() * (SacredConstants.NOISE_SCALE_MAX - SacredConstants.NOISE_SCALE_MIN);
        float noiseOffsetRight = (float)(_rng.NextDouble() * 0.014 + 0.001); // Time offset for right noise
        int fadeDuration = _rng.Next(2) == 0 ? 21 : 34; // Fibonacci pair for master fade
        int fadeSamples = fadeDuration * sampleRate;
        if (2 * fadeSamples > totalSamples) fadeSamples = totalSamples / 2;

        // Slow pan drift — gentle stereo movement independent of the torus.
        // Drift frequency centered on BREATH_ROOT / PHI^3 (PHI sub-harmonic of Earth's breath).
        // Drift amplitude bounded by Fibonacci reciprocals (1/89 to 1/55).
        float driftCenter = SacredConstants.DRIFT_FREQ_CENTER;
        float driftFreq = driftCenter + (float)(_rng.NextDouble() - 0.5) * driftCenter * SacredConstants.PHI_SQ_INVERSE;
        float driftAmplitude = SacredConstants.DRIFT_AMP_MIN +
            (float)_rng.NextDouble() * (SacredConstants.DRIFT_AMP_MAX - SacredConstants.DRIFT_AMP_MIN);

        // Build the 13-frequency set: 6 PHI exponents + 3 ratio-1.3 exponents + 4 subharmonics
        // Each frequency gets slight random jitter (±2-5%) for organic uniqueness
        var allFrequencies = BuildFrequencySet(baseFreq);
        var modDepths = new float[allFrequencies.Length];
        for (int i = 0; i < modDepths.Length; i++)
            modDepths[i] = (float)(_rng.NextDouble() * 0.2 + 0.15);

        // Order 2 biquad lowpass (~5500 Hz) — 12 dB/octave rolloff.
        // Gentler than the original order 4 at 2200 Hz — lets upper harmonics through
        // for brighter tones while smoothing harsh digital artifacts.
        float cutoffVariation = (float)(_rng.NextDouble() * 50 - 25);
        float lpCutoff = global::System.Math.Clamp(5500f + cutoffVariation, 2000f, 23000f);
        var filterLeft = BiquadFilter.CreateLowpass(2, lpCutoff, sampleRate);
        var filterRight = BiquadFilter.CreateLowpass(2, lpCutoff, sampleRate);

        // Pan smoother cutoff = BREATH_ROOT / PHI^3 — same as drift center.
        // Connects pan smoothing to Earth's Schumann breath through PHI.
        // Extremely slow — only the slowest organic drift survives.
        var panSmoother = new ExponentialSmoother(SacredConstants.PAN_SMOOTHER_CUTOFF, sampleRate);

        // PHI-fractal feedback — golden-ratio-delayed micro-echoes for harmonic richness.
        // Stateful: carries echo tail between chunks for seamless continuity.
        var phiFractalLeft = new PhiFractalFeedback(sampleRate);
        var phiFractalRight = new PhiFractalFeedback(sampleRate);

        // FFT convolution reverb — exponential decay with PHI sinusoidal modulation.
        // Stateful: carries overlap-add tail between chunks.
        var reverbLeft = new StreamingReverb(sampleRate);
        var reverbRight = new StreamingReverb(sampleRate);

        // 12-sample right-channel delay buffer for subtle stereo widening
        var delayBuffer = new float[12];

        // Pre-compute the Rössler chaotic attractor trajectory for the full session.
        // X and Y components drive the spatial panning perturbation (±0.08 radians).
        var rossler = Math.RosslerAttractor.Compute(duration);

        #endregion

        // Set up the 6 sacred healing layers and their independent toroidal panning.
        // Layers activate for sessions >60s or in Dimensional Journey mode.
        // Crystal sequence always starts with Lemurian Quartz (divine feminine first).
        #region Sacred Layer Setup

        // Build crystal sequence: Lemurian Quartz always first, rest randomized
        int lemIdx = _crystalLibrary.LemurianIndex;
        var rest = Enumerable.Range(0, _crystalLibrary.Profiles.Length)
            .Where(i => i != lemIdx).OrderBy(_ => _rng.Next()).ToArray();
        var crystalSequence = new[] { lemIdx }.Concat(rest).ToArray();

        // Instantiate all 6 sacred layers — each implements ISacredLayer via SacredLayerBase
        ISacredLayer[] sacredLayers =
        [
            new PleromaMercyLayer(),        // 1st: Aeonic ladder + Ogdoad gateway + Archon mercy
            new SilentSolfeggioGrid(),      // 2nd: 12-tone Solfeggio + Tesla 3-6-9 vortex
            new ArchonDissolutionLayer(),   // 3rd: AEG mercy for 7 planetary Archons
            new CrystallineResonanceLayer(_crystalLibrary, crystalSequence, baseFreq), // 4th: 9 crystal profiles
            new LemurianMerkabaLayer(),     // 5th: Sonic Merkaba + heart coherence
            new WaterElementLayer(),        // 6th: Hexagonal ripple field + lemniscate observer
        ];

        // Each sacred layer gets its own slow toroidal panning frequency
        float[] sacredThetaFreqs = [0.003f, 0.005f, 0.004f, 0.0035f, 0.0028f, 0.0045f];
        float[] sacredPhiFreqs = sacredThetaFreqs.Select(f => f * SacredConstants.PHI).ToArray();

        // Golden angle phase offsets — each layer starts at 137.5° * n around the torus,
        // like sunflower seeds filling space with maximum coverage, no repeating patterns
        float[] sacredPhaseOffsets = new float[6];
        for (int i = 0; i < 6; i++)
            sacredPhaseOffsets[i] = i * SacredConstants.GOLDEN_ANGLE_RAD;

        // Sacred layer torus radii — PHI-derived, matching the main torus geometry
        const float sacredR = 0.618f;      // 1/PHI — major radius
        const float sacredRSmall = 0.382f;  // 1/PHI² — minor radius (sum to 1.0)

        #endregion

        // Estimate peak amplitude for normalization (prevents clipping)
        float normGain = EstimateNormGain(allFrequencies, modDepths, lfoLeftParams, sampleRate);

        // Pre-allocate reusable buffers — saves ~4 MB of GC allocations per chunk
        var pool = new ChunkBufferPool(chunkSize);

        // The main generation loop — processes one 3-second chunk per iteration.
        // Each chunk passes through all 16 stages sequentially, then sacred layers
        // are computed in parallel and mixed in with independent toroidal panning.
        #region Per-Chunk Pipeline Loop

        int numChunks = (totalSamples + chunkSize - 1) / chunkSize;

        for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++)
        {
            if (ct.IsCancellationRequested) yield break;

            int chunkOffset = chunkIdx * chunkSize;
            int chunkSamples = global::System.Math.Min(chunkSize, totalSamples - chunkOffset);

            // Zero the pooled buffers for this chunk (last chunk may be shorter)
            pool.Clear(chunkSamples);

            // Build the absolute time array for this chunk's sample positions
            float tStart = (float)chunkOffset / sampleRate;
            var tChunk = pool.TChunk;
            for (int i = 0; i < chunkSamples; i++)
                tChunk[i] = tStart + (float)i / sampleRate;

            // Stage 1: Geometric modulation — sum of sine waves at sacred ratio frequencies
            var modulation = ComputeModulationChunk(tChunk, chunkOffset, chunkSamples, schedule);

            // Stage 2: Fractal frequency variation — dual simplex noise for organic pitch drift
            var fractalVar = FractalVariation.ComputeChunkDual(tChunk, baseFreq, simplexFractal);

            // Stage 3: Combine base frequency + chaotic offset + fractal variation + modulation
            float chaoticOffset = chaoticFactor * baseFreq * 0.25f;
            for (int i = 0; i < chunkSamples; i++)
                modulation[i] += baseFreq + chaoticOffset + fractalVar[i];

            // Stage 4: Organic ADSR envelope — Fibonacci-timed attack/decay/sustain/release
            var envelope = pool.Envelope;
            OrganicAdsrEnvelope.ComputeChunk(envelope, chunkOffset, chunkSamples,
                adsrParams, simplexEnvelope, totalSamples);

            // Stage 5: Microtonal LFOs — PHI-modulated breathing amplitude variation
            var lfoLeft = MicrotonalLfo.Compute(tChunk, lfoLeftParams);
            var lfoRight = MicrotonalLfo.Compute(tChunk, lfoRightParams);

            // Stage 6: Harmonic generation — 13 frequencies with envelope, LFO, and cross-modulation
            var waveLeft = HarmonicGenerator.GenerateHarmonics(
                tChunk, allFrequencies, envelope, lfoLeft, modDepths);
            var waveRight = HarmonicGenerator.GenerateHarmonics(
                tChunk, allFrequencies, envelope, lfoRight, modDepths);

            // Stage 7: Taygetan binaural beats — stereo frequency pairs with 7.7 Hz sync beat
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

            // Stage 8: Tanh wave shaping — soft saturation for gentle harmonic warmth
            WaveShaper.Process(waveLeft, 2.5f);
            WaveShaper.Process(waveRight, 2.5f);

            // Stage 8b: PHI-fractal feedback — golden-ratio-delayed echoes for crystal-like decay
            waveLeft = phiFractalLeft.ProcessChunk(waveLeft);
            waveRight = phiFractalRight.ProcessChunk(waveRight);

            // Stage 9: Biquad lowpass filter — smooths harsh digital artifacts (stateful, carries zi)
            filterLeft.Process(waveLeft);
            filterRight.Process(waveRight);

            // Stage 10: Normalization — apply estimated gain to prevent clipping
            for (int i = 0; i < chunkSamples; i++)
            {
                waveLeft[i] *= normGain;
                waveRight[i] *= normGain;
            }

            // Stage 11: FFT convolution reverb on left channel (stateful overlap-add)
            waveLeft = reverbLeft.ProcessChunk(waveLeft);

            // Stage 12: 12-sample right-channel delay + reverb (creates stereo width)
            var waveRightDelayed = pool.WaveRightDelayed;
            Array.Copy(delayBuffer, 0, waveRightDelayed, 0,
                global::System.Math.Min(12, chunkSamples));
            if (chunkSamples > 12)
                Array.Copy(waveRight, 0, waveRightDelayed, 12, chunkSamples - 12);
            Array.Copy(waveRight, global::System.Math.Max(0, chunkSamples - 12), delayBuffer, 0,
                global::System.Math.Min(12, chunkSamples));
            waveRight = reverbRight.ProcessChunk(waveRightDelayed);

            // Stage 13: Evolving noise layers — low-frequency modulated Gaussian noise
            var noiseLeft = EvolvingNoiseLayer.Generate(tChunk, rng: _rng);
            var tChunkOffset = pool.TChunkOffset;
            for (int i = 0; i < chunkSamples; i++)
                tChunkOffset[i] = tChunk[i] + noiseOffsetRight;
            var noiseRight = EvolvingNoiseLayer.Generate(tChunkOffset, rng: _rng);

            // Mix noise into the signal and apply per-channel noise scaling
            for (int i = 0; i < chunkSamples; i++)
            {
                waveLeft[i] = (waveLeft[i] + noiseLeft[i]) * noiseScaleLeft;
                waveRight[i] = (waveRight[i] + noiseRight[i]) * noiseScaleRight;
            }

            // Stage 14: Master fade in/out — t^1.5 curve at session start and end
            ApplyFade(waveLeft, waveRight, chunkOffset, chunkSamples, fadeSamples, totalSamples);

            // Stage 15: Toroidal panning — PHI-derived torus + Rössler chaos + simplex perturbation
            ApplyToroidalPanning(waveLeft, waveRight, tChunk, chunkSamples,
                torusThetaFreq, torusPhiFreq, torusR, torusRSmall,
                simplexPan, rossler, panSmoother, driftFreq, driftAmplitude, pool);

            // Apply master volume (0.52 — digits sum to 7, sacred number)
            for (int i = 0; i < chunkSamples; i++)
            {
                waveLeft[i] *= MasterVolume;
                waveRight[i] *= MasterVolume;
            }

            // Stage 16: Sacred healing layers — 6 layers computed in parallel, each with
            // independent toroidal panning at golden angle phase offsets
            if (duration > 60 || dimensionalMode)
            {
                // Launch all 6 sacred layers on the thread pool
                var sacredTasks = new Task<float[]>[sacredLayers.Length];
                for (int li = 0; li < sacredLayers.Length; li++)
                {
                    var layer = sacredLayers[li];
                    sacredTasks[li] = Task.Run(() => layer.ComputeChunk(tChunk, duration));
                }

                // Wait for all layers to complete (60-second timeout safety)
                try
                {
                    Task.WaitAll(sacredTasks, TimeSpan.FromSeconds(60));
                }
                catch (AggregateException) { }

                // Mix each sacred layer into the stereo field with independent toroidal panning.
                // Each layer's panning starts at a golden angle offset from the previous,
                // ensuring maximum spatial separation like sunflower seeds on a torus.
                for (int li = 0; li < sacredLayers.Length; li++)
                {
                    float[]? layerData = sacredTasks[li].IsCompletedSuccessfully
                        ? sacredTasks[li].Result
                        : null;
                    if (layerData == null) continue;

                    float stFreq = sacredThetaFreqs[li];
                    float spFreq = sacredPhiFreqs[li];
                    float phaseOff = sacredPhaseOffsets[li];

                    for (int i = 0; i < chunkSamples; i++)
                    {
                        // Compute torus position with golden angle phase offset
                        float sTheta = SacredConstants.TWO_PI * stFreq * tChunk[i] + phaseOff;
                        float sPhi = SacredConstants.TWO_PI * spFreq * tChunk[i] + phaseOff;

                        // Map torus position to stereo pan value
                        float sacredPan = (sacredR + sacredRSmall * MathF.Cos(sPhi)) *
                            MathF.Cos(sTheta) / (sacredR + sacredRSmall);
                        float panScaled = sacredPan * 0.4f;

                        // Mix into left and right channels with pan weighting
                        waveLeft[i] += layerData[i] * (1.0f - panScaled);
                        waveRight[i] += layerData[i] * (1.0f + panScaled);
                    }
                }
            }

            // Build the final stereo output chunk (fresh allocation — yielded to consumer)
            var stereoChunk = new float[chunkSamples, 2];
            for (int i = 0; i < chunkSamples; i++)
            {
                stereoChunk[i, 0] = waveLeft[i];
                stereoChunk[i, 1] = waveRight[i];
            }

            // Report progress and yield the chunk to the consumer
            updateProgress?.Invoke((float)(chunkIdx + 1) / numChunks);
            yield return stereoChunk;
        }

        #endregion
    }

    #endregion

    // Batch mode wraps the streaming generator and collects all chunks
    // into a single float[totalSamples, 2] array for WAV file writing.
    // Uses the exact same pipeline — identical sound output.
    #region Batch Generator

    /// <summary>
    /// Batch mode: collect all streaming chunks into a single array.
    /// Used by SoundPlayer.SaveToWavAsync for WAV file generation.
    /// Output is identical to streaming — same pipeline, same sound.
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

        // Iterate over streaming chunks and copy each into the result array
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

    #endregion
}
