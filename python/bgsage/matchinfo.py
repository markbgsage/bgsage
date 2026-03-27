"""Match play take points and gammon prices.

Live cube take points are empirically-derived lookup tables.
Dead cube take points and gammon prices are calculated exactly from the match
equity table. All take points are gammonless — they assume zero probability of
gammon/backgammon wins and losses.

For money games (no match context):
- Gammonless take point: 0.22 (both players)
- Gammonless dead cube take point: 0.25 (both players)
- Gammon price: 0.5 (both players)
"""


def _lookup(table, default, away1, away2):
    """Look up values from a symmetric table keyed by (min_away, max_away).

    Returns (value_for_player1, value_for_player2).
    The table stores (leader_value, trailer_value) keyed by (leader_away, trailer_away).
    """
    key = (min(away1, away2), max(away1, away2))
    if key not in table:
        return (default, default)
    leader_val, trailer_val = table[key]
    if away1 <= away2:
        return (leader_val, trailer_val)
    else:
        return (trailer_val, leader_val)


# Take points keyed by (leader_away, trailer_away)
# Values: (leader_take_point, trailer_take_point)
_TAKE_POINTS_CV1 = {
    (2, 2): (0.32, 0.32),
    (2, 3): (0.26, 0.25),
    (3, 3): (0.25, 0.25),
    (2, 4): (0.20, 0.19),
    (3, 4): (0.21, 0.25),
    (4, 4): (0.22, 0.22),
    (2, 5): (0.17, 0.23),
    (3, 5): (0.19, 0.22),
    (4, 5): (0.18, 0.24),
    (5, 5): (0.20, 0.20),
}

_TAKE_POINTS_CV2 = {
    (3, 5): (0.33, 0.16),
    (4, 5): (0.30, 0.26),
    (3, 6): (0.28, 0.11),
}

_TAKE_POINTS = {1: _TAKE_POINTS_CV1, 2: _TAKE_POINTS_CV2}


def _auto_redouble_take_point(away_taker, away_doubler, cube_value):
    """Gammonless dead cube take point when the taker has an automatic redouble.

    The taker has an automatic redouble when the doubler will win the match on a
    single win at the new cube level (away_doubler - 2*C <= 0). In that case, it
    never hurts the taker to redouble (if they lose they still lose the match),
    so the effective dead cube take point uses 4C instead of 2C:

    T = (MWC(a1, a2-C) - MWC(a1, a2-4C)) / (MWC(a1-4C, a2) - MWC(a1, a2-4C))
    """
    c = cube_value
    num = _mwc(away_taker, away_doubler - c) - _mwc(away_taker, away_doubler - 4 * c)
    den = _mwc(away_taker - 4 * c, away_doubler) - _mwc(away_taker, away_doubler - 4 * c)
    if den == 0:
        return 0.25
    return num / den


def take_points(away1, away2, cube_value=1):
    """Return gammonless (take_point_player1, take_point_player2) for the given match score.

    Gammonless means assuming zero probability of gammon/backgammon wins and losses.

    Handles three cases for each player's take point:
    1. Automatic redouble: if the doubler will win the match on a single win at the
       new cube level (their away - 2C <= 0), the taker has a free redouble, so a
       special dead cube formula at 4C is used.
    2. Lookup table: empirically-derived values for common scores at cube_value=1 and 2.
    3. Janowski approximation: fallback with x=0.68 for scores not in the table.

    Args:
        away1: Points player 1 needs to win the match.
        away2: Points player 2 needs to win the match.
        cube_value: Current cube value.

    Returns:
        Tuple of (take_point_for_player1, take_point_for_player2).
    """
    c = cube_value
    # Check for automatic redouble cases
    p1_auto = away2 - 2 * c <= 0  # player 2 doubles, player 1 has auto-redouble
    p2_auto = away1 - 2 * c <= 0  # player 1 doubles, player 2 has auto-redouble

    if p1_auto and p2_auto:
        tp1 = _auto_redouble_take_point(away1, away2, c)
        tp2 = _auto_redouble_take_point(away2, away1, c)
        return (tp1, tp2)

    if p1_auto:
        tp1 = _auto_redouble_take_point(away1, away2, c)
        # Player 2's take point: lookup or Janowski
        tp2 = _get_single_take_point(away2, away1, c)
        return (tp1, tp2)

    if p2_auto:
        tp2 = _auto_redouble_take_point(away2, away1, c)
        # Player 1's take point: lookup or Janowski
        tp1 = _get_single_take_point(away1, away2, c)
        return (tp1, tp2)

    # No automatic redoubles — use lookup or Janowski
    if cube_value in _TAKE_POINTS:
        table = _TAKE_POINTS[cube_value]
        key = (min(away1, away2), max(away1, away2))
        if key in table:
            return _lookup(table, 0.22, away1, away2)
    return take_points_janowski(away1, away2, cube_value, x=0.68)


def _get_single_take_point(away_taker, away_doubler, cube_value):
    """Get a single player's take point via lookup table or Janowski fallback."""
    if cube_value in _TAKE_POINTS:
        table = _TAKE_POINTS[cube_value]
        key = (min(away_taker, away_doubler), max(away_taker, away_doubler))
        if key in table:
            leader_val, trailer_val = table[key]
            if away_taker <= away_doubler:
                return leader_val
            else:
                return trailer_val
    return _janowski_take_point(away_taker, away_doubler, cube_value, x=0.68)


def _mwc(away1, away2):
    """MWC from the match equity table for player 1 needing away1, player 2 needing away2.

    If either away is 1, uses Crawford MWC. If away1 <= 0, returns 1.0 (player 1 has
    already won). If away2 <= 0, returns 0.0 (player 2 has already won).
    """
    if away1 <= 0:
        return 1.0
    if away2 <= 0:
        return 0.0
    import bgbot_cpp
    return bgbot_cpp.get_met(away1, away2, False)


def _dead_cube_take_point(away1, away2, cube_value):
    """Gammonless dead cube take point for player 1 at the given score and cube value.

    T = (MWC(a1, a2-C) - MWC(a1, a2-2C)) / (MWC(a1-2C, a2) - MWC(a1, a2-2C))
    where a1=away1, a2=away2, C=cube_value.
    """
    c = cube_value
    num = _mwc(away1, away2 - c) - _mwc(away1, away2 - 2 * c)
    den = _mwc(away1 - 2 * c, away2) - _mwc(away1, away2 - 2 * c)
    if den == 0:
        return 0.25
    return num / den


def _janowski_take_point(away1, away2, cube_value, x):
    """Gammonless Janowski live cube take point for player 1.

    T_live = T_dead * CP / (CP + x * (1 - CP))

    where T_dead is the dead cube take point and CP is the dead cube cash point
    at cube 2C (= 1 minus opponent's dead cube take point at 4C).

    If the taker (player 1) can't survive a redouble (away1 - 2*cube_value <= 0),
    the cube is dead after taking and the dead cube take point is returned directly.
    """
    c = cube_value
    # If the taker can't survive a redouble, cube is dead after taking
    if away1 - 2 * c <= 0:
        return _dead_cube_take_point(away1, away2, c)
    t_dead = _dead_cube_take_point(away1, away2, c)
    # Cash point: player 1's P(win) above which they'd redouble at cube 2C
    # CP = (MWC(a1-2C, a2) - MWC(a1, a2-4C)) / (MWC(a1-4C, a2) - MWC(a1, a2-4C))
    cp_num = _mwc(away1 - 2 * c, away2) - _mwc(away1, away2 - 4 * c)
    cp_den = _mwc(away1 - 4 * c, away2) - _mwc(away1, away2 - 4 * c)
    if cp_den == 0:
        return t_dead
    cp = cp_num / cp_den
    den = cp + x * (1 - cp)
    if den == 0:
        return t_dead
    return t_dead * cp / den


def take_points_janowski(away1, away2, cube_value=1, x=0.68):
    """Return gammonless Janowski live cube take points for both players.

    Uses the Janowski approximation with cube life index x:
    T_live = T_dead * CP / (CP + x * (1 - CP))

    Args:
        away1: Points player 1 needs to win the match.
        away2: Points player 2 needs to win the match.
        cube_value: Current cube value.
        x: Cube life index (cube efficiency). 0 = dead cube, 1 = fully live.

    Returns:
        Tuple of (take_point_for_player1, take_point_for_player2).
    """
    tp1 = _janowski_take_point(away1, away2, cube_value, x)
    tp2 = _janowski_take_point(away2, away1, cube_value, x)
    return (tp1, tp2)


def take_points_dead_cube(away1, away2, cube_value=1):
    """Return gammonless (dead_cube_take_point_player1, dead_cube_take_point_player2).

    Calculated from the match equity table. Gammonless means assuming zero
    probability of gammon/backgammon wins and losses.

    Args:
        away1: Points player 1 needs to win the match.
        away2: Points player 2 needs to win the match.
        cube_value: Current cube value.

    Returns:
        Tuple of (take_point_for_player1, take_point_for_player2).
    """
    tp1 = _dead_cube_take_point(away1, away2, cube_value)
    tp2 = _dead_cube_take_point(away2, away1, cube_value)
    return (tp1, tp2)


def gammon_prices(away1, away2, cube_value=1):
    """Return (gammon_price_player1, gammon_price_player2) for the given match score.

    Calculated from the match equity table.

    Args:
        away1: Points player 1 needs to win the match.
        away2: Points player 2 needs to win the match.
        cube_value: Current cube value.

    Returns:
        Tuple of (gammon_price_for_player1, gammon_price_for_player2).
    """
    c = cube_value
    den = _mwc(away1 - 2 * c, away2) - _mwc(away1, away2 - 2 * c)
    if den == 0:
        return (0.5, 0.5)
    g1 = (_mwc(away1, away2 - 2 * c) - _mwc(away1, away2 - 4 * c)) / den
    g2 = (_mwc(away1 - 4 * c, away2) - _mwc(away1 - 2 * c, away2)) / den
    return (g1, g2)
