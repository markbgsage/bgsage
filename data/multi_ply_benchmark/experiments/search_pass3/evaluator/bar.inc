// The trained pip-loss feature takes the largest hit target for each roll.
// With one on the bar it also allows a direct hit by an on-board checker
// when the other die can enter. The legacy feature visits the smaller direct
// die first and uses at most one on-board hit; on doubles it checks the
// two-die entry point for that direct hit.
static int entry_pip_loss(const Board& board, const ContactPointMasks& masks) {
    constexpr unsigned bar = 1u << 24;
    const unsigned targets = masks.pip_loss_targets;
    if (!targets) return 0;
    const unsigned blocks = masks.enemy_blocks;
    unsigned direct[7], entered[7];
    int total = 0;
    const bool one_on_bar = board[25] == 1;
    for (int die = 1; die <= 6; ++die) {
        entered[die] = (bar >> die) & targets;
        direct[die] = (masks.pip_loss_sources >> die) & targets;
        unsigned hits = entered[die];
        if (one_on_bar) {
            hits = double_roll_targets(bar, targets, die, blocks >> die, blocks >> (2*die),
                                      blocks >> (3*die));
            if (!(blocks & (bar >> (2 * die)))) hits |= direct[die];
        }
        total += highest_set_point(hits);
    }
    for (int high = 2; high <= 6; ++high) {
        for (int low = 1; low < high; ++low) {
            unsigned hits = entered[low] | entered[high];
            if (one_on_bar) {
                const bool enter_high = !(blocks & (bar >> high));
                const bool enter_low = !(blocks & (bar >> low));
                if (enter_high || enter_low) hits |= (bar >> (high + low)) & targets;
                if (enter_high && direct[low]) hits |= direct[low];
                else if (enter_low) hits |= direct[high];
            }
            total += 2 * highest_set_point(hits);
        }
    }
    return total;
}
