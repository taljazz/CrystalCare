using CrystalCare.Core.Dsp;
using CrystalCare.Core.Frequencies;

namespace CrystalCare.Core.Generation;

public sealed partial class SoundGenerator
{
    private float[] BuildFrequencySet(float baseFreq)
    {
        var freqs = new List<float>();

        // 6 PHI exponents
        for (int i = 0; i < 6; i++)
            freqs.Add(baseFreq * SacredConstants.PHI_EXPONENTS_6[i] *
                (float)(_rng.NextDouble() * 0.04 + 0.98));

        // 3 ratio-1.3 exponents
        for (int i = 0; i < 3; i++)
            freqs.Add(baseFreq * SacredConstants.RATIO_1_3_EXPONENTS_3[i] *
                (float)(_rng.NextDouble() * 0.04 + 0.98));

        // 4 subharmonics
        for (int i = 0; i < 4; i++)
            freqs.Add(baseFreq / SacredConstants.SUBHARMONIC_DIVISORS[i] *
                (float)(_rng.NextDouble() * 0.1 + 0.95));

        return freqs.ToArray();
    }

    private List<(int start, int end, float[] ratioValues, float modIndex)>
        BuildModulationSchedule(float duration, int totalSamples, int sampleRate,
            int[] intervalDurations, bool dimensionalMode, CancellationToken ct)
    {
        var schedule = new List<(int, int, float[], float)>();
        int current = 0;

        if (dimensionalMode)
        {
            int[] phases = [0, 2, 1, 3, 4, 3, 5];
            int phaseLen = totalSamples / (phases.Length + 1);
            foreach (int sel in phases)
            {
                if (ct.IsCancellationRequested) break;
                var ratioSet = _frequencyManager.SelectRandomRatioSet(ct);
                float modIndex = (float)(_rng.NextDouble() * 0.05 + 0.2);
                int end = global::System.Math.Min(current + phaseLen, totalSamples);
                schedule.Add((current, end, ratioSet.Values.ToArray(), modIndex));
                current = end;
            }
            // 'all' phase
            if (current < totalSamples)
            {
                var allRatios = FrequencyManager.RatioSets.Values
                    .SelectMany(d => d.Values).Distinct().ToArray();
                float modIndex = (float)(_rng.NextDouble() * 0.05 + 0.2);
                schedule.Add((current, totalSamples, allRatios, modIndex));
            }
        }
        else
        {
            float remaining = duration;
            int intervalCount = 0;
            while (remaining > 0 && !ct.IsCancellationRequested)
            {
                float interval = global::System.Math.Min(
                    intervalDurations[intervalCount % intervalDurations.Length], remaining);
                int segSamples = (int)(sampleRate * interval);
                var ratioSet = _frequencyManager.SelectRandomRatioSet(ct);
                float modIndex = (float)(_rng.NextDouble() * 0.05 + 0.2);
                int end = global::System.Math.Min(current + segSamples, totalSamples);
                schedule.Add((current, end, ratioSet.Values.ToArray(), modIndex));
                current = end;
                remaining -= interval;
                intervalCount++;
            }
        }

        return schedule;
    }

    private static float[] ComputeModulationChunk(ReadOnlySpan<float> tChunk,
        int chunkOffset, int chunkSamples,
        List<(int start, int end, float[] ratioValues, float modIndex)> schedule)
    {
        var result = new float[chunkSamples];
        int chunkEnd = chunkOffset + chunkSamples;

        foreach (var (start, end, ratioValues, modIndex) in schedule)
        {
            if (start >= chunkEnd || end <= chunkOffset) continue;

            int localStart = global::System.Math.Max(0, start - chunkOffset);
            int localEnd = global::System.Math.Min(chunkSamples, end - chunkOffset);

            for (int r = 0; r < ratioValues.Length; r++)
            {
                float ratio = ratioValues[r];
                for (int i = localStart; i < localEnd; i++)
                    result[i] += modIndex * MathF.Sin(SacredConstants.TWO_PI * ratio * tChunk[i]);
            }
        }

        return result;
    }

    private float EstimateNormGain(float[] frequencies, float[] modDepths,
        MicrotonalLfo.LfoParams lfoParams, int sampleRate)
    {
        int estSamples = global::System.Math.Min(sampleRate / 2, 24000);
        float quarterPeriod = 1.0f / (4.0f * MathF.Max(lfoParams.Lfo1Freq, 0.001f));

        var estT = new float[estSamples];
        for (int i = 0; i < estSamples; i++)
            estT[i] = quarterPeriod + (float)i / sampleRate;

        var estEnv = new float[estSamples];
        Array.Fill(estEnv, 1.0f);
        var estLfo = MicrotonalLfo.Compute(estT, lfoParams);
        var estWave = HarmonicGenerator.GenerateHarmonics(estT, frequencies, estEnv, estLfo, modDepths);
        WaveShaper.Process(estWave, 2.5f);

        float peak = 0.001f;
        for (int i = 0; i < estSamples; i++)
            peak = MathF.Max(peak, MathF.Abs(estWave[i]));

        return 1.0f / (peak * 1.3f);
    }
}
