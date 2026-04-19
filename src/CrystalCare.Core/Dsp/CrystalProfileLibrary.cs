using CrystalCare.Core.Frequencies;
using CrystalCare.Core.Noise;

namespace CrystalCare.Core.Dsp;

/// <summary>
/// Crystal acoustic profiles based on Raman spectroscopy data.
/// 9 crystal types, each with unique harmonic ratios, symmetry orders,
/// and optional shimmer/beating/piezo effects.
///
/// Port of AudioProcessor._build_crystal_profiles() and _generate_crystal_harmonics().
/// </summary>
public sealed class CrystalProfileLibrary
{
    // Data model for a single crystal profile — harmonic ratios from Raman spectroscopy,
    // symmetry order, optional shimmer/beating/piezo/detune effects.
    #region Crystal Profile Data Model

    public sealed class CrystalProfile
    {
        public required CrystalType Type { get; init; }
        public required string Name { get; init; }
        public required float[] HarmonicRatios { get; init; }
        public float[] AmplitudeWeights { get; set; } = [];
        public int SymmetryOrder { get; init; }
        public float DetuneFactor { get; init; }
        public float ShimmerRate { get; init; }
        public float ShimmerDepth { get; init; } = 0.15f;
        public (int, int)[] BeatingPairs { get; init; } = [];
        public float PiezoFactor { get; init; }
        public float BowlBeat { get; init; }
    }

    #endregion

    // Public access to the profile array and the Lemurian Quartz index.
    // Lemurian Quartz is always first in every session — divine feminine heart energy.
    #region Properties and Constructor

    public CrystalProfile[] Profiles { get; }
    public int LemurianIndex { get; }

    public CrystalProfileLibrary()
    {
        Profiles = BuildProfiles();
        LemurianIndex = Array.FindIndex(Profiles, p => p.Type == CrystalType.LemurianQuartz);
    }

    #endregion

    // Generates the harmonic signal for a single crystal profile at given time values.
    // Sums weighted sine waves at harmonic ratio frequencies, then applies optional
    // piezo phase wobble, organic detune drift, shimmer modulation, beating pairs,
    // and bowl beat modulation depending on the crystal's properties.
    #region Harmonic Generation

    /// <summary>
    /// Generate harmonics for a single crystal profile at given time values.
    /// </summary>
    public static float[] GenerateHarmonics(ReadOnlySpan<double> t, float baseFreq,
        CrystalProfile profile, Simplex5D simplex)
    {
        var ratios = profile.HarmonicRatios;
        var weights = profile.AmplitudeWeights;
        int nRatios = ratios.Length;
        var result = new float[t.Length];

        // Phase wobble from piezo effect — simplex takes float, scaled values stay small
        float[]? phaseWobble = null;
        if (profile.PiezoFactor > 0)
        {
            var tScaled = new float[t.Length];
            for (int i = 0; i < t.Length; i++)
                tScaled[i] = (float)(t[i] * 0.015);
            phaseWobble = simplex.GenerateNoise(tScaled, 2.0f);
            for (int i = 0; i < phaseWobble.Length; i++)
                phaseWobble[i] *= profile.PiezoFactor;
        }

        // Detune noise
        float[]? detuneNoise = null;
        if (profile.DetuneFactor > 0)
        {
            var tScaled = new float[t.Length];
            for (int i = 0; i < t.Length; i++)
                tScaled[i] = (float)(t[i] * 0.008);
            detuneNoise = simplex.GenerateNoise(tScaled, 3.0f);
        }

        // Sum weighted harmonics — double precision phase for long-session stability
        for (int r = 0; r < nRatios; r++)
        {
            double freq = baseFreq * ratios[r];
            float weight = weights[r];

            for (int i = 0; i < t.Length; i++)
            {
                double phase;
                if (detuneNoise != null)
                {
                    double detuneScale = (r + 1) * profile.DetuneFactor;
                    phase = SacredConstants.TWO_PI_D * freq * (1.0 + detuneScale * detuneNoise[i]) * t[i];
                }
                else
                {
                    phase = SacredConstants.TWO_PI_D * freq * t[i];
                }

                if (phaseWobble != null)
                    phase += phaseWobble[i];

                result[i] += weight * (float)System.Math.Sin(phase);
            }
        }

        // Shimmer modulation — low frequency, use double for consistency
        if (profile.ShimmerRate > 0)
        {
            for (int i = 0; i < t.Length; i++)
            {
                float shimmer = 1.0f + profile.ShimmerDepth *
                    (float)System.Math.Sin(SacredConstants.TWO_PI_D * profile.ShimmerRate * t[i]);
                result[i] *= shimmer;
            }
        }

        // Beating pairs — double precision phase
        foreach (var pair in profile.BeatingPairs)
        {
            double h1Freq = baseFreq * ratios[pair.Item1];
            double h2Freq = baseFreq * ratios[pair.Item2];
            for (int i = 0; i < t.Length; i++)
            {
                float beat = (float)System.Math.Sin(SacredConstants.TWO_PI_D * h1Freq * t[i]) *
                             (float)System.Math.Sin(SacredConstants.TWO_PI_D * h2Freq * t[i]);
                result[i] += 0.05f * beat;
            }
        }

        // Bowl beat modulation — slow rate, use double
        if (profile.BowlBeat > 0)
        {
            for (int i = 0; i < t.Length; i++)
            {
                float beatMod = 1.0f + 0.08f *
                    (float)System.Math.Sin(SacredConstants.TWO_PI_D * profile.BowlBeat * t[i]);
                result[i] *= beatMod;
            }
        }

        return result;
    }

    #endregion

    // Defines all 9 crystal profiles with harmonic ratios from Raman spectroscopy.
    // Lemurian Quartz: 12 harmonics with heart chakra bridges and divine feminine warmth.
    // Clear Quartz: 9 harmonics, master healer. Amethyst: meditative drift.
    // Also computes PHI-weighted amplitude weights and mineral-specific warmth weights.
    #region Profile Definitions

    private static CrystalProfile[] BuildProfiles()
    {
        float[] quartzRatios = [0.276f, 0.444f, 0.571f, 0.765f, 1.0f, 1.500f, 1.741f, 2.339f, 2.504f];
        float[] tourmalineRatios = [0.375f, 0.583f, 1.0f, 1.061f, 1.098f, 2.047f];
        float[] seleniteRatios = [0.412f, 0.491f, 0.617f, 0.669f, 1.0f, 1.126f];
        float[] bowlRatios = [1.0f, 2.828f, 5.424f, 9.130f];
        float[] roseRatios = [.. quartzRatios, 2.509f, 4.300f];
        float[] lapisRatios = [0.472f, 1.0f, 1.066f, 0.144f, 0.260f, 0.656f, 1.0f, 0.905f, 1.0f, 1.135f];
        float[] lapisMineralWeights = [0.5f, 0.5f, 0.5f, 0.3f, 0.3f, 0.3f, 0.3f, 0.2f, 0.2f, 0.2f];

        float[] lemurianRatios =
        [
            0.276f, 0.444f, 0.571f, 0.765f,    // Sub-fundamental quartz
            1.0f,                                 // Fundamental
            528f / SacredConstants.MERKABA_KEYNOTE, // 528/432 Love Frequency ratio (exact)
            1.375f,                               // Heart resonance (594/432 exact)
            1.500f,                               // Quartz harmonic
            SacredConstants.PHI,                  // Golden heart
            1.741f,                               // Quartz harmonic
            2.339f, 2.504f,                       // Upper quartz
        ];
        float[] lemurianWarmthWeights =
        [
            0.6f, 0.7f, 0.8f, 0.9f,
            1.0f, 1.2f, 1.1f, 0.9f,
            1.0f, 0.5f, 0.3f, 0.2f,
        ];

        var profiles = new[]
        {
            new CrystalProfile { Type = CrystalType.ClearQuartz, Name = "Clear Quartz",
                HarmonicRatios = quartzRatios, SymmetryOrder = 3, PiezoFactor = 0.02f },
            new CrystalProfile { Type = CrystalType.Amethyst, Name = "Amethyst",
                HarmonicRatios = (float[])quartzRatios.Clone(),
                SymmetryOrder = 3, DetuneFactor = 0.008f, PiezoFactor = 0.02f },
            new CrystalProfile { Type = CrystalType.RoseQuartz, Name = "Rose Quartz",
                HarmonicRatios = roseRatios, SymmetryOrder = 3, PiezoFactor = 0.02f },
            new CrystalProfile { Type = CrystalType.Citrine, Name = "Citrine",
                HarmonicRatios = (float[])quartzRatios.Clone(),
                SymmetryOrder = 3, DetuneFactor = 0.006f, PiezoFactor = 0.02f },
            new CrystalProfile { Type = CrystalType.BlackTourmaline, Name = "Black Tourmaline",
                HarmonicRatios = tourmalineRatios, SymmetryOrder = 3,
                BeatingPairs = [(2, 3), (3, 4), (2, 4)], PiezoFactor = 0.025f },
            new CrystalProfile { Type = CrystalType.Selenite, Name = "Selenite",
                HarmonicRatios = seleniteRatios, SymmetryOrder = 2, BeatingPairs = [(4, 5)] },
            new CrystalProfile { Type = CrystalType.LapisLazuli, Name = "Lapis Lazuli",
                HarmonicRatios = lapisRatios, SymmetryOrder = 4 },
            new CrystalProfile { Type = CrystalType.CrystalSingingBowl, Name = "Crystal Singing Bowl",
                HarmonicRatios = bowlRatios, SymmetryOrder = 1, BowlBeat = 1.8f },
            new CrystalProfile { Type = CrystalType.LemurianQuartz, Name = "Lemurian Quartz",
                HarmonicRatios = lemurianRatios, SymmetryOrder = 3, DetuneFactor = 0.003f,
                ShimmerRate = 1.8f, ShimmerDepth = 0.08f, PiezoFactor = 0.02f },
        };

        // Compute amplitude weights for each profile
        foreach (var profile in profiles)
        {
            var ratios = profile.HarmonicRatios;
            int n = ratios.Length;
            var weights = new float[n];
            Array.Fill(weights, 1.0f);

            int sym = profile.SymmetryOrder;
            if (sym > 0)
            {
                for (int j = 0; j < n; j++)
                {
                    if ((j + 1) % sym == 0)
                        weights[j] *= SacredConstants.PHI;
                }
            }

            // Apply mineral weights for lapis and lemurian
            float[]? mineralWeights = profile.Type switch
            {
                CrystalType.LapisLazuli => lapisMineralWeights,
                CrystalType.LemurianQuartz => lemurianWarmthWeights,
                _ => null
            };

            if (mineralWeights != null)
            {
                for (int j = 0; j < n && j < mineralWeights.Length; j++)
                    weights[j] *= mineralWeights[j];
            }

            // Normalize
            float sum = weights.Sum();
            for (int j = 0; j < n; j++)
                weights[j] /= sum;

            profile.AmplitudeWeights = weights;
        }

        return profiles;
    }

    #endregion
}
