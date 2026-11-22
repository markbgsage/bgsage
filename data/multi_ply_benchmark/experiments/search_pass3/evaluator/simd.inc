        // Each lane is one non-double roll. The last lane has no sources.
        const __m256i high_dice[] = {
            _mm256_setr_epi32(2,3,3,4,4,4,5,5), _mm256_setr_epi32(5,5,6,6,6,6,6,32)};
        const __m256i low_dice[] = {
            _mm256_setr_epi32(1,1,2,1,2,3,1,2), _mm256_setr_epi32(3,4,1,2,3,4,5,32)};
        const __m256i zeros = _mm256_setzero_si256();
        const __m256i blockers = _mm256_set1_epi32(masks.enemy_blocks);
        const __m256i blot = _mm256_set1_epi32(blots);
        const __m256i occupied = _mm256_set1_epi32(masks.occupied);
        const __m256i made = _mm256_set1_epi32(masks.made);
        const __m256i pip_sources = _mm256_set1_epi32(masks.pip_loss_sources);
        const __m256i pip_targets = _mm256_set1_epi32(masks.pip_loss_targets);
        __m256i pip_total = zeros;
        for (int group = 0; group < 2; ++group) {
            const __m256i h = high_dice[group], l = low_dice[group], sum = _mm256_add_epi32(h,l);
            const __m256i blocked = _mm256_and_si256(_mm256_srlv_epi32(blockers, h),
                                                     _mm256_srlv_epi32(blockers, l));
            const __m256i high_direct = _mm256_and_si256(blot, _mm256_srlv_epi32(occupied, h));
            const __m256i low_direct = _mm256_and_si256(blot, _mm256_srlv_epi32(occupied, l));
            const __m256i indirect = _mm256_andnot_si256(blocked, _mm256_srlv_epi32(occupied, sum));
            const __m256i single = _mm256_or_si256(_mm256_or_si256(high_direct, low_direct),
                                                    _mm256_and_si256(blot, indirect));
            auto face_count = [&](const __m256i hits) {
                const int bits = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(hits, zeros)));
                return 2 * _mm_popcnt_u32(bits);
            };
            counts.single_hits += face_count(single);
            if (multiple_blots) {
                const __m256i through = _mm256_or_si256(
                    _mm256_and_si256(high_direct, _mm256_sllv_epi32(blot, l)),
                    _mm256_and_si256(low_direct, _mm256_sllv_epi32(blot, h)));
                // A direct pair fails only if it uses the same target or the same lone
                // source. Testing the lowest target of each set against the other set
                // suffices: two failing partners for one target must straddle it by h-l.
                const __m256i first_h = _mm256_and_si256(high_direct, _mm256_sub_epi32(zeros, high_direct));
                const __m256i first_l = _mm256_and_si256(low_direct, _mm256_sub_epi32(zeros, low_direct));
                const __m256i lone_h = _mm256_andnot_si256(made, _mm256_sllv_epi32(first_h, h));
                const __m256i lone_l = _mm256_andnot_si256(made, _mm256_sllv_epi32(first_l, l));
                const __m256i other_l = _mm256_andnot_si256(_mm256_or_si256(first_h, _mm256_srlv_epi32(lone_h, l)), low_direct);
                const __m256i other_h = _mm256_andnot_si256(_mm256_or_si256(first_l, _mm256_srlv_epi32(lone_l, h)), high_direct);
                const __m256i separate = _mm256_or_si256(
                    _mm256_and_si256(_mm256_cmpgt_epi32(first_h, zeros), other_l),
                    _mm256_and_si256(_mm256_cmpgt_epi32(first_l, zeros), other_h));
                counts.double_hits += face_count(_mm256_or_si256(through, separate));
            }
            const __m256i pip_direct_hits = _mm256_or_si256(_mm256_srlv_epi32(pip_sources, h),
                                                             _mm256_srlv_epi32(pip_sources, l));
            const __m256i pip_indirect = _mm256_andnot_si256(blocked, _mm256_srlv_epi32(pip_sources, sum));
            const __m256i hits = _mm256_and_si256(pip_targets, _mm256_or_si256(pip_direct_hits, pip_indirect));
            // All 24-bit masks convert exactly to binary32. Extract floor(log2(mask))+1,
            // clamping the zero exponent to zero for rolls with no qualifying blot.
            const __m256i exponent = _mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(hits)), 23);
            pip_total = _mm256_add_epi32(pip_total, _mm256_max_epi32(zeros,
                _mm256_sub_epi32(exponent, _mm256_set1_epi32(126))));
        }
        int lanes[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), pip_total);
        for (int lane : lanes) counts.pip_loss += 2 * lane;
