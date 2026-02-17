#include "bgbot/board.h"

namespace bgbot {

Board flip(const Board& b) {
    Board f;
    f[0] = b[25];   // opponent's bar becomes index 0
    f[25] = b[0];   // player's bar becomes index 25
    for (int i = 1; i <= 24; ++i) {
        f[i] = -b[25 - i];
    }
    return f;
}

GameResult check_game_over(const Board& b) {
    // Count checkers on board for each player
    int sum1 = 0; // player 1 checkers still on board (including bar)
    int sum2 = 0; // player 2 checkers still on board (including bar)

    for (int i = 1; i <= 24; ++i) {
        if (b[i] > 0) sum1 += b[i];
        if (b[i] < 0) sum2 -= b[i];
    }
    sum1 += b[25]; // player 1 on bar
    sum2 += b[0];  // player 2 on bar

    if (sum1 == 0 && sum2 > 0) {
        // Player 1 has borne off all checkers
        if (sum2 < 15) {
            return GameResult::WIN_SINGLE;
        }
        // Player 2 has no checkers off. Check for backgammon.
        // Backgammon: opponent has checker in winner's home board or on bar
        int home_sum = b[0]; // p2 on bar
        for (int i = 1; i <= 6; ++i) {
            if (b[i] < 0) home_sum -= b[i];
        }
        if (home_sum > 0) {
            return GameResult::WIN_BACKGAMMON;
        }
        return GameResult::WIN_GAMMON;
    }

    if (sum2 == 0 && sum1 > 0) {
        // Player 2 has borne off all checkers
        if (sum1 < 15) {
            return GameResult::LOSS_SINGLE;
        }
        // Player 1 has no checkers off. Check for backgammon.
        int home_sum = b[25]; // p1 on bar
        for (int i = 19; i <= 24; ++i) {
            if (b[i] > 0) home_sum += b[i];
        }
        if (home_sum > 0) {
            return GameResult::LOSS_BACKGAMMON;
        }
        return GameResult::LOSS_GAMMON;
    }

    return GameResult::NOT_OVER;
}

bool is_race(const Board& b) {
    // If either player has a checker on the bar, not a race
    if (b[0] > 0 || b[25] > 0) return false;

    // Find the furthest-forward player 1 checker
    int max1 = 0;
    for (int i = 24; i >= 1; --i) {
        if (b[i] > 0) { max1 = i; break; }
    }
    // Find the furthest-back player 2 checker (from p2's perspective, closest to p1's home)
    int min2 = 25;
    for (int i = 1; i <= 24; ++i) {
        if (b[i] < 0) { min2 = i; break; }
    }

    // It's a race if player 1's furthest checker is not past player 2's nearest
    return max1 <= min2;
}

bool is_crashed(const Board& b) {
    // A position is "crashed" if either side's home board wall is collapsing:
    // - No outfield checkers (points 7-18)
    // - Has checkers on opponent's side (points 19-24 or bar for player; 1-6 or bar for opp)
    // - No checkers on their own 6-point

    // Player 1: outfield = points 7-18, opponent side = points 19-24 + bar(25)
    int outfield_p1 = 0;
    for (int i = 7; i <= 18; ++i) {
        if (b[i] > 0) outfield_p1 += b[i];
    }
    int opp_side_p1 = b[25]; // bar counts as being on opponent's side
    for (int i = 19; i <= 24; ++i) {
        if (b[i] > 0) opp_side_p1 += b[i];
    }
    // Reference: checkers[6] <= 0 means p1 has no checkers on point 6
    bool p1_crashed = (outfield_p1 == 0 && opp_side_p1 > 0 && b[6] <= 0);

    // Player 2: outfield = points 7-18 (negative values), opponent side = points 1-6 + bar(0)
    int outfield_p2 = 0;
    for (int i = 7; i <= 18; ++i) {
        if (b[i] < 0) outfield_p2 -= b[i];
    }
    int opp_side_p2 = b[0]; // bar
    for (int i = 1; i <= 6; ++i) {
        if (b[i] < 0) opp_side_p2 -= b[i];
    }
    // Reference: checkers[19] >= 0 means p2 has no checkers on point 19 (p2's 6-point)
    bool p2_crashed = (outfield_p2 == 0 && opp_side_p2 > 0 && b[19] >= 0);

    return p1_crashed || p2_crashed;
}

int player_checkers_on_board(const Board& b) {
    int n = b[25]; // bar
    for (int i = 1; i <= 24; ++i) {
        if (b[i] > 0) n += b[i];
    }
    return n;
}

int player_borne_off(const Board& b) {
    return 15 - player_checkers_on_board(b);
}

int opponent_borne_off(const Board& b) {
    int n = b[0]; // bar
    for (int i = 1; i <= 24; ++i) {
        if (b[i] < 0) n -= b[i];
    }
    return 15 - n;
}

std::pair<int, int> pip_counts(const Board& b) {
    int pip1 = 0;
    int pip2 = 0;
    for (int i = 1; i <= 24; ++i) {
        if (b[i] > 0) pip1 += i * b[i];
        if (b[i] < 0) pip2 -= (25 - i) * b[i];
    }
    pip1 += 25 * b[25];  // player on bar
    pip2 += 25 * b[0];   // opponent on bar
    return {pip1, pip2};
}

} // namespace bgbot
