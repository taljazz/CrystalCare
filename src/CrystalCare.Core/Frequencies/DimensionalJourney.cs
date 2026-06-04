namespace CrystalCare.Core.Frequencies;

/// <summary>
/// Definitions for the 9-dimension journey (Mode 7 — DimensionalShift /
/// "Dimensional Journey 1D–9D Realignment").
///
/// Each dimension has its OWN sacred-mathematics signature, expressed three ways:
///
///   1. Ratio set — which set of sacred-geometry ratios drives Stage 1 modulation.
///      Each of the 9 ratio sets in FrequencyManager maps to one dimension, so a
///      dimensional session walks through the full Lemurian / Atlantean /
///      Pythagorean / Pleiadian palette in coherent ascending order.
///
///   2. Modulation intensity — gentle in low dimensions (foundation, body),
///      stronger in high dimensions (soul, source). Reflects how the higher
///      dimensions are felt as more vivid in channeled tradition.
///
///   3. Per-voice amp emphasis — the 13-voice harmonic field's "spectral center"
///      moves with the dimension. Low dimensions emphasize subharmonics (body
///      grounding); high dimensions emphasize upper PHI exponents (crown,
///      ascension). The CARRIER FREQUENCY stays at 432 Hz (Lemurian keynote
///      anchor), but the FELT center of the field journeys upward.
///
/// This replaces the prior broken implementation where the dimension index was
/// iterated but never used — every phase just picked a random ratio set, and
/// 3D was missing entirely.
/// </summary>
public sealed record Dimension(
    int Number,            // 1..9
    string Label,          // e.g., "3D — Heart / Body"
    string RatioSetKey,    // key into FrequencyManager.RatioSets, or "all_blended" for 9D
    float ModIndex,        // Stage 1 modulation intensity (0.18 gentle, 0.30 strong)
    float[] AmpScales,     // per-voice amp emphasis (length 13, multipliers on 0.015/(f+1) decay)
    float[] LayerEmphasis); // per-sacred-layer amplitude emphasis (length 7, multipliers on layer output)

/// <summary>
/// The 9-dimension journey definitions. Used by SoundGenerator when freqMode
/// is DimensionalShift to build the modulation schedule and apply per-chunk
/// harmonic-field amp emphasis.
/// </summary>
public static class DimensionalJourney
{
    // Per-dimension amp emphasis arrays for the 13-voice harmonic field.
    // Each array has 13 entries multiplying the natural 0.015/(f+1) decay scale.
    // Voice indices: 0-5 = PHI^0..5, 6-8 = 1.3^0..2, 9-12 = subharmonics /2,/4,/8,/16.
    // Higher dimensions emphasize upper PHI exponents (crown-band frequencies);
    // lower dimensions emphasize subharmonics (body-grounding frequencies).
    // The "spectral center of mass" of the field journeys upward as d ascends.
    //
    // Values are hand-tuned around 1.0 (no emphasis = baseline); >1.0 boosts,
    // <1.0 quiets. All multiply on top of the natural harmonic decay so the
    // sacred-mathematics shape is preserved — these only shift emphasis, not
    // structure.
    //
    // DECLARATION ORDER MATTERS: these must be declared BEFORE the DIMENSIONS
    // array below because C# static-field initialization runs in textual order
    // (DIMENSIONS' initializer references these arrays — if they came after,
    // they'd still be null at the moment DIMENSIONS initializes).
    #region Per-Dimension Amp Scales

    /// <summary>1D Foundation: emphasize subharmonics (body grounding, Earth contact).</summary>
    private static readonly float[] AmpScales1D = [
        0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.2f,   // PHI^0..5 (lower presence)
        0.7f, 0.6f, 0.5f,                       // 1.3^0..2
        1.6f, 1.7f, 1.6f, 1.5f                  // subharms — strong (body grounding)
    ];

    /// <summary>2D Etheric: subharmonics still emphasized but lifting upward slightly.</summary>
    private static readonly float[] AmpScales2D = [
        0.8f, 0.7f, 0.5f, 0.4f, 0.3f, 0.2f,
        0.8f, 0.7f, 0.6f,
        1.4f, 1.5f, 1.4f, 1.3f
    ];

    /// <summary>3D Heart / Body: balanced — full spectrum, the body's frequency.</summary>
    private static readonly float[] AmpScales3D = [
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f
    ];

    /// <summary>4D Emotional: shift upward — slight PHI emphasis, subharmonics quiet a touch.</summary>
    private static readonly float[] AmpScales4D = [
        1.1f, 1.2f, 1.2f, 1.1f, 1.0f, 0.9f,
        1.0f, 1.0f, 0.9f,
        0.9f, 0.8f, 0.8f, 0.8f
    ];

    /// <summary>5D Light Body: full geometric — PHI exponents foreground, body voices quiet.</summary>
    private static readonly float[] AmpScales5D = [
        1.2f, 1.3f, 1.4f, 1.3f, 1.2f, 1.0f,
        1.0f, 1.0f, 0.9f,
        0.7f, 0.6f, 0.6f, 0.5f
    ];

    /// <summary>6D Higher Mind: upper PHI exponents emphasized — structural ascent.</summary>
    private static readonly float[] AmpScales6D = [
        1.1f, 1.3f, 1.4f, 1.5f, 1.4f, 1.2f,
        0.9f, 0.9f, 0.8f,
        0.6f, 0.5f, 0.5f, 0.4f
    ];

    /// <summary>7D Soul / Causal: transcendental — high voices stronger.</summary>
    private static readonly float[] AmpScales7D = [
        1.0f, 1.2f, 1.4f, 1.5f, 1.6f, 1.4f,
        0.8f, 0.8f, 0.7f,
        0.5f, 0.4f, 0.4f, 0.3f
    ];

    /// <summary>8D Galactic: upper PHI dominant — cosmic frequencies.</summary>
    private static readonly float[] AmpScales8D = [
        0.9f, 1.1f, 1.3f, 1.5f, 1.7f, 1.6f,
        0.7f, 0.7f, 0.6f,
        0.4f, 0.3f, 0.3f, 0.2f
    ];

    /// <summary>9D Source / Crown: highest PHI exponents fully foreground — crown band.</summary>
    private static readonly float[] AmpScales9D = [
        0.8f, 1.0f, 1.2f, 1.5f, 1.8f, 1.9f,
        0.7f, 0.6f, 0.5f,
        0.3f, 0.3f, 0.2f, 0.2f
    ];

    #endregion

    // Per-dimension sacred-layer emphasis arrays. Each array has 7 entries
    // multiplying each layer's output before Stage 16 mixing. Layer indices
    // match the sacredLayers array in SoundGenerator:
    //
    //   [0] PleromaMercyLayer     — aeonic ladder, divine fullness
    //   [1] SilentSolfeggioGrid   — 12-Solfeggio + Tesla 3-6-9
    //   [2] ArchonDissolutionLayer — AEG mercy for 7 planetary spheres
    //   [3] CrystallineResonanceLayer — 9 crystal Raman profiles
    //   [4] LemurianMerkabaLayer  — Sonic Merkaba (star tetrahedron)
    //   [5] WaterElementLayer     — hexagonal ripple field, lemniscate path
    //   [6] BlueRayResonanceLayer — Arcturian zero-point still axis
    //
    // Each dimension foregrounds the layer(s) most aligned with its meaning:
    //   1D Earth          → Water dominant (Earth element)
    //   2D Etheric        → Water still strong + Pleroma rises
    //   3D Heart/Body     → Crystalline dominant (crystals heal the body)
    //   4D Emotional      → Solfeggio Grid dominant (12-tone emotional balance)
    //   5D Light Body     → Merkaba dominant (light body in geometric form)
    //   6D Higher Mind    → Archon dominant (structure for mind transformation)
    //   7D Soul/Causal    → Pleroma dominant (highest channeled descent)
    //   8D Galactic       → Pleroma + Blue Ray (cosmic ascent)
    //   9D Source/Crown   → Blue Ray DOMINANT (the zero point IS the crown)
    //
    // Values around 1.0 = baseline. Above 1.0 = emphasized. Below 1.0 = quieted.
    // Range chosen to avoid extreme attenuation; every layer always present.
    #region Per-Dimension Sacred-Layer Emphasis

    /// <summary>1D Foundation/Earth: Water dominant (Earth element grounding).</summary>
    private static readonly float[] LayersEmph1D = [
        0.5f,  // Pleroma — quiet, far from Earth body
        0.8f,  // Solfeggio Grid
        0.7f,  // Archon
        1.0f,  // Crystalline — physical baseline
        0.8f,  // Merkaba — geometric not yet active
        1.5f,  // Water — dominant (Earth element)
        0.5f,  // Blue Ray — quiet (crown far from foundation)
    ];

    /// <summary>2D Etheric/Atlantean: Water + Pleroma rising from Earth toward etheric.</summary>
    private static readonly float[] LayersEmph2D = [
        0.8f,  // Pleroma rises (etheric body activated)
        0.9f,  // Solfeggio
        0.9f,  // Archon
        0.9f,  // Crystalline
        0.7f,  // Merkaba — not yet
        1.4f,  // Water — still strong (etheric is watery)
        0.5f,  // Blue Ray
    ];

    /// <summary>3D Heart/Body: Crystalline dominant (crystals heal the physical body).</summary>
    private static readonly float[] LayersEmph3D = [
        0.9f,  // Pleroma
        1.0f,  // Solfeggio
        1.0f,  // Archon
        1.4f,  // Crystalline — DOMINANT (physical body, crystals)
        1.0f,  // Merkaba
        1.0f,  // Water — full presence
        0.5f,  // Blue Ray
    ];

    /// <summary>4D Emotional/Astral: Solfeggio Grid dominant (12-tone emotional field).</summary>
    private static readonly float[] LayersEmph4D = [
        1.0f,  // Pleroma
        1.4f,  // Solfeggio — DOMINANT (emotional/astral balance)
        1.0f,  // Archon
        1.0f,  // Crystalline
        1.0f,  // Merkaba — beginning to activate
        0.9f,  // Water
        0.7f,  // Blue Ray
    ];

    /// <summary>5D Light Body: Merkaba dominant (Sonic Merkaba in the light-body field).</summary>
    private static readonly float[] LayersEmph5D = [
        1.0f,  // Pleroma
        1.0f,  // Solfeggio
        1.0f,  // Archon
        1.0f,  // Crystalline
        1.5f,  // Merkaba — DOMINANT (light body, star tetrahedron)
        0.8f,  // Water — receding
        0.8f,  // Blue Ray
    ];

    /// <summary>6D Higher Mind: Archon dominant (7 planetary spheres = mind structure).</summary>
    private static readonly float[] LayersEmph6D = [
        1.1f,  // Pleroma rising
        1.0f,  // Solfeggio
        1.4f,  // Archon — DOMINANT (mind/structure transformation)
        0.9f,  // Crystalline
        1.2f,  // Merkaba — still strong
        0.7f,  // Water
        0.9f,  // Blue Ray rising
    ];

    /// <summary>7D Soul/Causal: Pleroma dominant (highest channeled descent — divine fullness).</summary>
    private static readonly float[] LayersEmph7D = [
        1.4f,  // Pleroma — DOMINANT (soul/causal — channeling from the Pleroma)
        1.0f,  // Solfeggio
        1.0f,  // Archon
        0.8f,  // Crystalline
        1.0f,  // Merkaba
        0.6f,  // Water
        1.0f,  // Blue Ray
    ];

    /// <summary>8D Galactic/Taygetan: Pleroma + Blue Ray (cosmic — both divine fullness and zero point).</summary>
    private static readonly float[] LayersEmph8D = [
        1.2f,  // Pleroma
        1.0f,  // Solfeggio
        1.0f,  // Archon
        0.7f,  // Crystalline — receding
        0.9f,  // Merkaba
        0.5f,  // Water — quiet (far from Earth)
        1.1f,  // Blue Ray — Arcturian / Pleiadian alignment
    ];

    /// <summary>9D Source/Crown: Blue Ray DOMINANT (the zero point IS the crown / source).</summary>
    private static readonly float[] LayersEmph9D = [
        1.3f,  // Pleroma
        1.1f,  // Solfeggio
        1.1f,  // Archon
        0.7f,  // Crystalline
        1.2f,  // Merkaba
        0.5f,  // Water — minimal
        1.5f,  // Blue Ray — DOMINANT (Source, crown, zero point)
    ];

    #endregion

    // The 9 dimensions in ascending order — each with its own sacred mathematics
    // identity. Mapping draws from the Python original's intent + filling in 3D
    // (which the prior implementation skipped) + completing the 1-to-1 pairing
    // with the 9 ratio sets in FrequencyManager so every set has its dimension.
    #region Dimensions

    /// <summary>
    /// The 9 dimensions in order. Each dimension occupies 1/9 of the session
    /// duration when Mode 7 is active.
    ///
    /// Mapping rationale:
    ///   1D Foundation       → minimal           (just √2 — Metatron's Cube, simplest geometric anchor)
    ///   2D Etheric/Astral   → fibonacci_set     (PHI-based — Atlantean cosmic harmonics)
    ///   3D Heart/Body       → triple_helix      (DNA ratios 1.0/1.2/1.4 — the physical body)
    ///   4D Emotional        → flower_of_life    (soft 1.3/1.5/2.5 — astral flow)
    ///   5D Light Body       → sacred_geometry   (full Pythagorean — Metatron/Vesica/hex/PHI/circle)
    ///   6D Higher Mind      → combined          (sacred + flower — bridging structure and flow)
    ///   7D Soul / Causal    → fractal_set       (transcendental π and e — beyond rational form)
    ///   8D Galactic         → taygetan          (Pleiadian / Taygetan ratios)
    ///   9D Source / Crown   → all_blended       (every ratio across every set sounds together)
    /// </summary>
    public static readonly Dimension[] DIMENSIONS =
    [
        new(1, "1D — Foundation / Earth",      "minimal",          0.18f, AmpScales1D, LayersEmph1D),
        new(2, "2D — Etheric / Atlantean",     "fibonacci_set",    0.20f, AmpScales2D, LayersEmph2D),
        new(3, "3D — Heart / Body",            "triple_helix",     0.21f, AmpScales3D, LayersEmph3D),
        new(4, "4D — Emotional / Astral",      "flower_of_life",   0.22f, AmpScales4D, LayersEmph4D),
        new(5, "5D — Light Body / Geometry",   "sacred_geometry",  0.23f, AmpScales5D, LayersEmph5D),
        new(6, "6D — Higher Mind",             "combined",         0.24f, AmpScales6D, LayersEmph6D),
        new(7, "7D — Soul / Causal",           "fractal_set",      0.25f, AmpScales7D, LayersEmph7D),
        new(8, "8D — Galactic / Taygetan",     "taygetan",         0.27f, AmpScales8D, LayersEmph8D),
        new(9, "9D — Source / Crown / All",    "all_blended",      0.30f, AmpScales9D, LayersEmph9D),
    ];

    #endregion

    // Helper to resolve a dimension's ratio set into actual ratio values.
    // For 9D ("all_blended") we union the ratio values across every set in
    // FrequencyManager so 9D culminates with every sacred ratio sounding at once.
    #region Ratio Resolution

    /// <summary>
    /// Resolve a dimension's RatioSetKey into the actual array of ratio values
    /// that Stage 1 modulation will multiply against the carrier. For named
    /// keys (e.g., "sacred_geometry"), looks up FrequencyManager.RatioSets.
    /// For the special "all_blended" key (9D), returns every distinct ratio
    /// across every set — the Source / Crown culmination.
    /// </summary>
    public static float[] ResolveRatios(Dimension dim)
    {
        // 9D Source: every ratio across every sacred set, deduplicated.
        // This is the culmination — at the end of the journey, the listener
        // hears every sacred-mathematics ratio sounding together as one.
        if (dim.RatioSetKey == "all_blended")
        {
            return FrequencyManager.RatioSets.Values
                .SelectMany(d => d.Values)
                .Distinct()
                .ToArray();
        }

        // Named ratio set — look up by key from FrequencyManager.
        // Returns the set's ratio values (already a float[] essentially).
        if (FrequencyManager.RatioSets.TryGetValue(dim.RatioSetKey, out var set))
        {
            return set.Values.ToArray();
        }

        // Defensive fallback — should never trip since we own the keys above.
        // Return a single 1.0 ratio so the modulation isn't silent if a key
        // is ever mistyped during refactoring.
        return [1.0f];
    }

    /// <summary>
    /// Compute which dimension is active at a given time within the session.
    /// Phases are equal-duration: each dimension occupies totalDuration / 9.
    /// Returns the Dimension record + its index (0..8). Beyond session end, clamps to 9D.
    /// </summary>
    public static (Dimension dim, int index) AtTime(float timeSec, float totalDuration)
    {
        // Guard against zero/negative durations
        if (totalDuration <= 0f) return (DIMENSIONS[0], 0);

        // Phase length = total / 9 — each dimension gets equal airtime
        float phaseLen = totalDuration / DIMENSIONS.Length;
        int idx = (int)(timeSec / phaseLen);

        // Clamp to valid index range (session-end edge case)
        idx = global::System.Math.Clamp(idx, 0, DIMENSIONS.Length - 1);
        return (DIMENSIONS[idx], idx);
    }

    #endregion

    // Smootherstep crossfade between adjacent dimensions. The last
    // CROSSFADE_FRACTION of each phase blends smoothly into the next dimension's
    // amp scales / layer emphasis, so the carrier-journey transitions feel
    // gentle organic shifts instead of hard switches. Perlin smootherstep
    // (6t^5 - 15t^4 + 10t^3) is used because it's the same curve every other
    // crossfade in the codebase uses (Crystalline, Taygetan ratio bias).
    #region Smootherstep Crossfade

    /// <summary>
    /// Fraction of each dimension's window devoted to smootherstep crossfade
    /// INTO the next dimension. 0.15 = the last 15% of each phase blends with
    /// the next. Felt as gentle transitions; mid-90% of each phase still
    /// expresses the dimension's pure character.
    /// </summary>
    public const float CROSSFADE_FRACTION = 0.15f;

    /// <summary>
    /// Perlin smootherstep: 6t^5 - 15t^4 + 10t^3. Smooth at both endpoints
    /// (zero first and second derivatives), no overshoot. Used everywhere
    /// in the codebase for organic-feeling transitions.
    /// </summary>
    private static float Smootherstep(float t)
    {
        t = global::System.Math.Clamp(t, 0f, 1f);
        return t * t * t * (t * (t * 6f - 15f) + 10f);
    }

    /// <summary>
    /// Compute per-voice amp scales at a given session time, smoothly
    /// interpolating between adjacent dimensions during the crossfade window
    /// at each phase boundary. Returns a fresh float[13] each call.
    ///
    /// Inside a dimension's main window (mid 85%), returns that dimension's
    /// pure AmpScales. In the last 15% (the crossfade tail), blends via
    /// smootherstep into the next dimension's AmpScales. The very last
    /// dimension (9D) has no "next" so its tail stays pure.
    /// </summary>
    public static float[] ComputeAmpScalesAt(float timeSec, float totalDuration)
    {
        // Guard
        if (totalDuration <= 0f) return DIMENSIONS[0].AmpScales;

        float phaseLen = totalDuration / DIMENSIONS.Length;
        int idx = global::System.Math.Clamp(
            (int)(timeSec / phaseLen), 0, DIMENSIONS.Length - 1);
        var cur = DIMENSIONS[idx];

        // Position within current phase: 0 at phase start, 1 at phase end
        float posInPhase = (timeSec - idx * phaseLen) / phaseLen;
        posInPhase = global::System.Math.Clamp(posInPhase, 0f, 1f);

        // Within main window — pure dimension AmpScales (most of the phase)
        if (posInPhase < 1f - CROSSFADE_FRACTION || idx == DIMENSIONS.Length - 1)
        {
            // Return a copy so the caller can't mutate the shared array
            return (float[])cur.AmpScales.Clone();
        }

        // In the crossfade tail — blend toward next dimension via smootherstep
        var next = DIMENSIONS[idx + 1];
        float x = (posInPhase - (1f - CROSSFADE_FRACTION)) / CROSSFADE_FRACTION;
        float blend = Smootherstep(x);  // 0..1 across the crossfade window

        var result = new float[cur.AmpScales.Length];
        for (int v = 0; v < result.Length; v++)
        {
            // Linear interpolation between cur and next, weighted by smootherstep
            result[v] = cur.AmpScales[v] * (1f - blend) + next.AmpScales[v] * blend;
        }
        return result;
    }

    /// <summary>
    /// Compute per-sacred-layer emphasis multipliers at a given session time,
    /// smoothly interpolating between adjacent dimensions during the crossfade
    /// window. Same pattern as ComputeAmpScalesAt — pure inside the main
    /// window, smootherstep blend in the last CROSSFADE_FRACTION of each phase.
    /// Returns a fresh float[7] each call (one per sacred layer).
    /// </summary>
    public static float[] ComputeLayerEmphasisAt(float timeSec, float totalDuration)
    {
        if (totalDuration <= 0f) return DIMENSIONS[0].LayerEmphasis;

        float phaseLen = totalDuration / DIMENSIONS.Length;
        int idx = global::System.Math.Clamp(
            (int)(timeSec / phaseLen), 0, DIMENSIONS.Length - 1);
        var cur = DIMENSIONS[idx];

        float posInPhase = (timeSec - idx * phaseLen) / phaseLen;
        posInPhase = global::System.Math.Clamp(posInPhase, 0f, 1f);

        if (posInPhase < 1f - CROSSFADE_FRACTION || idx == DIMENSIONS.Length - 1)
        {
            return (float[])cur.LayerEmphasis.Clone();
        }

        var next = DIMENSIONS[idx + 1];
        float x = (posInPhase - (1f - CROSSFADE_FRACTION)) / CROSSFADE_FRACTION;
        float blend = Smootherstep(x);

        var result = new float[cur.LayerEmphasis.Length];
        for (int v = 0; v < result.Length; v++)
        {
            result[v] = cur.LayerEmphasis[v] * (1f - blend) + next.LayerEmphasis[v] * blend;
        }
        return result;
    }

    #endregion
}
