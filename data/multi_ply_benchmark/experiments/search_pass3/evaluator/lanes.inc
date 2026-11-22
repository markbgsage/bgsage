#if defined(__AVX2__)
// Eight independent 24-point target masks, one per dice roll.
struct RollMasks {
    __m256i bits;
    explicit RollMasks(unsigned value) : bits(_mm256_set1_epi32(value)) {}
    RollMasks(__m256i value) : bits(value) {}
    friend RollMasks operator&(RollMasks a, RollMasks b) { return _mm256_and_si256(a.bits, b.bits); }
    friend RollMasks operator|(RollMasks a, RollMasks b) { return _mm256_or_si256(a.bits, b.bits); }
    friend RollMasks operator~(RollMasks a) { return _mm256_xor_si256(a.bits, _mm256_set1_epi32(-1)); }
    friend RollMasks operator+(RollMasks a, RollMasks b) { return _mm256_add_epi32(a.bits, b.bits); }
    friend RollMasks operator-(RollMasks a, RollMasks b) { return _mm256_sub_epi32(a.bits, b.bits); }
    friend RollMasks operator>>(RollMasks a, RollMasks b) { return _mm256_srlv_epi32(a.bits, b.bits); }
    friend RollMasks operator<<(RollMasks a, RollMasks b) { return _mm256_sllv_epi32(a.bits, b.bits); }
    RollMasks first() const { return *this & (RollMasks(0u) - *this); }
    RollMasks nonempty() const { return _mm256_cmpgt_epi32(bits, _mm256_setzero_si256()); }
    int hit_rolls() const {
        return _mm_popcnt_u32(_mm256_movemask_ps(_mm256_castsi256_ps(nonempty().bits)));
    }
    RollMasks highest_points() const {
        // A 24-bit mask converts exactly to binary32. Its unbiased exponent
        // plus one is the point index. Clamp the empty mask's result to zero.
        const __m256i exp = _mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(bits)), 23);
        return _mm256_max_epi32(_mm256_setzero_si256(), _mm256_sub_epi32(exp, _mm256_set1_epi32(126)));
    }
    int sum() const {
        int lanes[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), bits);
        return std::accumulate(lanes, lanes + 8, 0);
    }
};

static ShotFeatures off_bar_shots(const ContactPointMasks& masks, unsigned reachable_blots) {
    ShotFeatures counts;
    const RollMasks occupied(masks.occupied), made(masks.made), blocks(masks.enemy_blocks);
    const RollMasks pip_sources(masks.pip_loss_sources), pip_targets(masks.pip_loss_targets);
    const RollMasks blots(reachable_blots);
    const bool multiple = (reachable_blots & (reachable_blots - 1)) != 0;

    // Six doubles. Shifts of 32 make the spare lanes empty.
    const RollMasks die = _mm256_setr_epi32(1,2,3,4,5,6,32,32);
    const RollMasks twice = die + die, three = twice + die, four = twice + twice;
    const RollMasks b1 = blocks >> die, b2 = blocks >> twice, b3 = blocks >> three;
    const auto double_paths = [&](RollMasks sources) {
        return (sources >> die) | (~b1 & ((sources >> twice) |
               (~b2 & ((sources >> three) | (~b3 & (sources >> four))))));
    };
    counts.single_hits = (blots & double_paths(occupied)).hit_rolls();
    if (multiple) {
        const RollMasks direct = blots & (occupied >> die);
        const RollMasks from2 = occupied >> twice, from3 = occupied >> three, from4 = occupied >> four;
        const RollMasks all = direct | (blots & ~b1 & (from2 | (~b2 & from3)));
        const RollMasks distinct = direct.nonempty() & all & (all - RollMasks(1u));
        const RollMasks lower = (blots << die) | (~(blocks << die) &
            ((blots << twice) | (~(blocks << twice) & (blots << three))));
        const RollMasks higher = blots >> die;
        const RollMasks via_blot = (higher & from2) | (~b2 & (higher | (blots >> twice)) & (from3 | (~b3 & from4)));
        counts.double_hits = (distinct | (direct & lower) | (blots & ~b1 & via_blot)).hit_rolls();
    }
    RollMasks pip_total = (pip_targets & double_paths(pip_sources)).highest_points();

    // Fifteen unlike rolls, each with two equally likely dice orders.
    const RollMasks high[] = {_mm256_setr_epi32(2,3,3,4,4,4,5,5), _mm256_setr_epi32(5,5,6,6,6,6,6,32)};
    const RollMasks low[] = {_mm256_setr_epi32(1,1,2,1,2,3,1,2), _mm256_setr_epi32(3,4,1,2,3,4,5,32)};
    for (int group = 0; group < 2; ++group) {
        const RollMasks h = high[group], l = low[group], sum = h + l;
        const RollMasks open = ~((blocks >> h) & (blocks >> l));
        const RollMasks hd = blots & (occupied >> h), ld = blots & (occupied >> l);
        counts.single_hits += 2 * (hd | ld | (blots & open & (occupied >> sum))).hit_rolls();
        if (multiple) {
            const RollMasks through = (hd & (blots << l)) | (ld & (blots << h));
            // A pair of direct hits can fail only on the same target or from
            // the same lone checker. If the two lowest targets coincide, any
            // higher h target works. If they share a source, any higher l target
            // works. Testing each lowest target against the other set suffices.
            const RollMasks first_h = hd.first(), first_l = ld.first();
            const RollMasks lone_h = ~made & (first_h << h), lone_l = ~made & (first_l << l);
            const RollMasks other_l = ld & ~(first_h | (lone_h >> l));
            const RollMasks other_h = hd & ~(first_l | (lone_l >> h));
            counts.double_hits += 2 * (through | (hd.nonempty() & other_l) |
                                       (ld.nonempty() & other_h)).hit_rolls();
        }
        const RollMasks hits = pip_targets &
            ((pip_sources >> h) | (pip_sources >> l) | (open & (pip_sources >> sum)));
        const RollMasks points = hits.highest_points();
        pip_total = pip_total + points + points;
    }
    counts.pip_loss = pip_total.sum();
    return counts;
}
#endif
