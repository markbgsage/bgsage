"""Cube action points (Take / Double / Cash / Too Good).

For money games this uses the standard closed-form Janowski formulas in
``(W, L, x)``. For match play the LIVE cube thresholds come from the recursive
GetPoints algorithm in bgbot_cpp (``match_live_cash_points``) and the DEAD cube
thresholds come from exact MET arithmetic; each of the four action points is
then a linear interpolation of the two in the cube life index ``x``.

Money-game reference:
    Rick Janowski, "Take-Points in Money Games" — reproduced in
    ``docs/Janowski.docx`` and re-implemented in ``bgsage/cpp/src/cube.cpp``.

Match-play reference:
    GNUbg's GetPoints, ``get_match_points()`` in ``bgsage/cpp/src/cube.cpp``.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Sequence

from .matchinfo import _mwc


# Outcome probability vector convention (matches CubeAnalysisResponse):
#   probs[0] = P(win)                  P(single win)     = probs[0] - probs[1]
#   probs[1] = P(gammon OR bg win)     P(gammon only)    = probs[1] - probs[2]
#   probs[2] = P(backgammon win)
#   probs[3] = P(gammon OR bg loss)    P(gammon only L)  = probs[3] - probs[4]
#   probs[4] = P(backgammon loss)


@dataclass
class CubePointTriple:
    """A single cube action point at three values of the cube life index."""
    dead: float       # x = 0
    live: float       # x = 1
    janowski: float   # the user's x


@dataclass
class PlayerCubePoints:
    """Cube action points from a single player's perspective."""
    W: float
    L: float
    double_label: Optional[str]   # "Double" (centered) / "Redouble" (owns) / None
    cannot_take: bool              # True when this player owns the cube
    cannot_double: bool            # True when the opponent owns or cube is dead
    cannot_cash: bool              # == cannot_double
    cannot_too_good: bool          # == cannot_double
    take: CubePointTriple
    double: CubePointTriple        # all-NaN when cannot_double
    cash: CubePointTriple          # all-NaN when cannot_cash
    too_good: CubePointTriple      # all-NaN when cannot_too_good

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Shared helpers ─────────────────────────────────────────────────


def _compute_wl(probs: Sequence[float]) -> tuple[float, float]:
    """Raw W and L from a 5-probability vector."""
    p_win = probs[0]
    p_gw = probs[1]
    p_bw = probs[2]
    p_gl = probs[3]
    p_bl = probs[4]

    W = 1.0 + (p_gw + p_bw) / p_win if p_win > 1e-7 else 1.0
    p_lose = 1.0 - p_win
    L = 1.0 + (p_gl + p_bl) / p_lose if p_lose > 1e-7 else 1.0
    return W, L


def _gammon_ratios(probs: Sequence[float]) -> tuple[float, float, float, float]:
    """(rG0, rBG0, rG1, rBG1) — gammon / backgammon ratios per win/loss side."""
    p_win = probs[0]
    if p_win > 1e-7:
        rG0 = (probs[1] - probs[2]) / p_win
        rBG0 = probs[2] / p_win
    else:
        rG0 = 0.0
        rBG0 = 0.0
    p_lose = 1.0 - p_win
    if p_lose > 1e-7:
        rG1 = (probs[3] - probs[4]) / p_lose
        rBG1 = probs[4] / p_lose
    else:
        rG1 = 0.0
        rBG1 = 0.0
    return rG0, rBG0, rG1, rBG1


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


NAN = float("nan")
_NA_TRIPLE = CubePointTriple(dead=NAN, live=NAN, janowski=NAN)


# ─── Money-game Janowski formulas ───────────────────────────────────
#
# The closed-form formulas below are linear in the reciprocal of the cube
# life index; here we split them into dead (x=0) and live (x=1) endpoints
# and expose the user's x value via the same interpolation the formulas bake
# in. For completeness we keep the match-play API symmetric by doing the same
# split here, but for money games all three values come from the same formula.


def _money_closed_form(W: float, L: float, x: float, numerator: float) -> float:
    denom = W + L + 0.5 * x
    if abs(denom) < 1e-12:
        return NAN
    return numerator / denom


def _money_take(W: float, L: float, x: float) -> float:
    return _money_closed_form(W, L, x, L - 0.5)


def _money_cash(W: float, L: float, x: float) -> float:
    return _money_closed_form(W, L, x, L + 0.5 + 0.5 * x)


def _money_too_good(W: float, L: float, x: float) -> float:
    return _money_closed_form(W, L, x, L + 1.0)


def _money_redouble(W: float, L: float, x: float) -> float:
    return _money_closed_form(W, L, x, L + x)


def _money_initial_double(W: float, L: float, x: float) -> float:
    # Janowski cube-action formulae table, basic general model.
    g = (x * (3.0 - x)) / (2.0 * (2.0 - x)) if (2.0 - x) > 1e-12 else 1.0
    return _money_closed_form(W, L, x, L + g)


def _jacoby_k1(W: float, L: float, x: float) -> float:
    denom = L * (W + L - (1.0 - x))
    if abs(denom) < 1e-9:
        return 1.0
    return ((W + L) * (L - 0.5 * (1.0 - x))) / denom


def _jacoby_k2(W: float, L: float, x: float) -> float:
    denom = L * (W + L - 0.5 * (1.0 - x))
    if abs(denom) < 1e-9:
        return 1.0
    k2 = ((W + L) * (L - 0.25 * (1.0 - x))) / denom
    return max(k2, _jacoby_k1(W, L, x))


def _money_points(
    probs: Sequence[float],
    cube_owner: str,
    cube_life_index: float,
    jacoby: bool,
    beaver: bool,
    no_gammons: bool,
) -> PlayerCubePoints:
    """Cube action points for a money / unlimited game."""
    W_raw, L_raw = _compute_wl(probs)
    W, L = (1.0, 1.0) if no_gammons else (W_raw, L_raw)
    x = _clamp01(cube_life_index)

    cannot_take = cube_owner == "player"
    cannot_decide = cube_owner == "opponent"

    def triple(fn):
        return CubePointTriple(dead=fn(W, L, 0.0),
                               live=fn(W, L, 1.0),
                               janowski=fn(W, L, x))

    take = _NA_TRIPLE if cannot_take else triple(_money_take)
    cash = _NA_TRIPLE if cannot_decide else triple(_money_cash)
    too_good = _NA_TRIPLE if cannot_decide else triple(_money_too_good)

    # Double row: Redouble (owns) / Double (centered, Jacoby-aware) / N/A (opp owns).
    if cannot_decide:
        double = _NA_TRIPLE
        double_label = None
    elif cube_owner == "player":
        double = triple(_money_redouble)
        double_label = "Redouble"
    else:
        # Centered. Apply Jacoby / beaver factors when active.
        use_jacoby = jacoby
        def id_at(wv: float, lv: float, xv: float) -> float:
            base = _money_initial_double(wv, lv, xv)
            if not use_jacoby:
                return base
            k = _jacoby_k2(wv, lv, xv) if beaver else _jacoby_k1(wv, lv, xv)
            return base * k
        double = triple(id_at)
        double_label = "Double"

    return PlayerCubePoints(
        W=W, L=L,
        double_label=double_label,
        cannot_take=cannot_take,
        cannot_double=cannot_decide,
        cannot_cash=cannot_decide,
        cannot_too_good=cannot_decide,
        take=take, double=double, cash=cash, too_good=too_good,
    )


# ─── Match-play dead-cube (MET-exact) thresholds ────────────────────


def _dead_take(away1: int, away2: int, cv: int,
               rG0: float, rBG0: float, rG1: float, rBG1: float) -> float:
    """Dead-cube take point for player 1.

    Opp doubles at cv; player decides take/pass. Take ⇒ cube becomes 2*cv, dead
    (no more cube action). Solve ``mwc_pass == mwc_take`` for p = P(player wins).

        mwc_pass  = MWC(a1, a2 - cv)
        mwc_take(p) = p * ( (1-rG0-rBG0)*MWC(a1-2cv, a2)
                          + rG0*MWC(a1-4cv, a2)
                          + rBG0*MWC(a1-6cv, a2) )
                   + (1-p) * ( (1-rG1-rBG1)*MWC(a1, a2-2cv)
                             + rG1*MWC(a1, a2-4cv)
                             + rBG1*MWC(a1, a2-6cv) )
    """
    c = cv
    pass_mwc = _mwc(away1, away2 - c)
    win_mwc = ((1.0 - rG0 - rBG0) * _mwc(away1 - 2 * c, away2)
               + rG0 * _mwc(away1 - 4 * c, away2)
               + rBG0 * _mwc(away1 - 6 * c, away2))
    lose_mwc = ((1.0 - rG1 - rBG1) * _mwc(away1, away2 - 2 * c)
                + rG1 * _mwc(away1, away2 - 4 * c)
                + rBG1 * _mwc(away1, away2 - 6 * c))
    denom = win_mwc - lose_mwc
    if abs(denom) < 1e-12:
        return 0.25
    return (pass_mwc - lose_mwc) / denom


def _dead_cash(away1: int, away2: int, cv: int,
               rG0: float, rBG0: float, rG1: float, rBG1: float) -> float:
    """Dead-cube cash point (player doubles ⇒ opp passes threshold).

    Equivalent to ``1 − opp's dead take point`` — symmetric across players.
    """
    opp_tp = _dead_take(away2, away1, cv, rG1, rBG1, rG0, rBG0)
    return _clamp01(1.0 - opp_tp)


def _dead_too_good(away1: int, away2: int, cv: int,
                   rG0: float, rBG0: float, rG1: float, rBG1: float) -> float:
    """Dead-cube too-good point (player plays on vs cashing).

    The player is "too good" when the MWC from playing on (at the current cube)
    exceeds the MWC from cashing (= mwc(a1-cv, a2)). With gammon-weighted
    outcomes:

        ND(p)   = p * win_mwc_cv + (1-p) * lose_mwc_cv
        Cash    = mwc(a1-cv, a2)
        TG_p    = (Cash - lose_mwc_cv) / (win_mwc_cv - lose_mwc_cv)

    In the gammonless / money-like case ``win_mwc_cv == Cash`` so TG = 1.0.
    """
    c = cv
    cash = _mwc(away1 - c, away2)
    win_mwc = ((1.0 - rG0 - rBG0) * _mwc(away1 - c, away2)
               + rG0 * _mwc(away1 - 2 * c, away2)
               + rBG0 * _mwc(away1 - 3 * c, away2))
    lose_mwc = ((1.0 - rG1 - rBG1) * _mwc(away1, away2 - c)
                + rG1 * _mwc(away1, away2 - 2 * c)
                + rBG1 * _mwc(away1, away2 - 3 * c))
    denom = win_mwc - lose_mwc
    if abs(denom) < 1e-12:
        return 1.0
    return _clamp01((cash - lose_mwc) / denom)


def _dead_double(away1: int, away2: int, cv: int,
                 rG0: float, rBG0: float, rG1: float, rBG1: float,
                 centered: bool) -> float:
    """Dead-cube initial / redouble point — the P(win) threshold below which
    the player doesn't want to (re)double (because ND-equity ≥ double-equity).

    Solves ``ND(p) = Redouble_take(p)`` in MWC space:

        ND(p) = p * win_mwc_cv + (1-p) * lose_mwc_cv
        RT(p) = p * win_mwc_2cv + (1-p) * lose_mwc_2cv   (opponent takes)

    where both win/lose MWCs are gammon-weighted.
    """
    c = cv
    win_cv = ((1.0 - rG0 - rBG0) * _mwc(away1 - c, away2)
              + rG0 * _mwc(away1 - 2 * c, away2)
              + rBG0 * _mwc(away1 - 3 * c, away2))
    win_2cv = ((1.0 - rG0 - rBG0) * _mwc(away1 - 2 * c, away2)
               + rG0 * _mwc(away1 - 4 * c, away2)
               + rBG0 * _mwc(away1 - 6 * c, away2))
    lose_cv = ((1.0 - rG1 - rBG1) * _mwc(away1, away2 - c)
               + rG1 * _mwc(away1, away2 - 2 * c)
               + rBG1 * _mwc(away1, away2 - 3 * c))
    lose_2cv = ((1.0 - rG1 - rBG1) * _mwc(away1, away2 - 2 * c)
                + rG1 * _mwc(away1, away2 - 4 * c)
                + rBG1 * _mwc(away1, away2 - 6 * c))

    denom = (win_2cv - lose_2cv) - (win_cv - lose_cv)
    if abs(denom) < 1e-12:
        return NAN
    p = (lose_cv - lose_2cv) / denom
    # ``centered`` vs ``player-owns`` doesn't change the dead-cube math:
    # in both cases the double/pass decision reduces to identical arithmetic.
    # The flag is accepted for signature symmetry with the live form.
    _ = centered
    return _clamp01(p)


# ─── Match-play live cube (recursive GetPoints) ─────────────────────


def _live_cash_points(away1: int, away2: int, cv: int, is_crawford: bool,
                      rG0: float, rBG0: float,
                      rG1: float, rBG1: float) -> tuple[float, float]:
    """(player_cp, opp_cp) from the recursive C++ implementation."""
    import bgbot_cpp
    return bgbot_cpp.match_live_cash_points(
        away1, away2, cv, is_crawford, rG0, rBG0, rG1, rBG1
    )


# ─── Janowski-exact thresholds via binary search on cube_decision_1ply ──
#
# For the user's cube life index x, thresholds come from solving indifference
# equations in equity space — ``E_cf(p, x) = (1-x)*E_dead(p) + x*E_live(p)``
# matches whatever cl2cf_match produces internally, so a binary search over
# cube_decision_1ply at exactly this x yields the exact Janowski threshold
# (the equivalent of money-game closed-form formulas like
# ``TP(x) = (L-0.5)/(W+L+0.5x)`` but for the match-play piecewise model).


def _probs_from_ratios(p_win: float,
                       rG0: float, rBG0: float,
                       rG1: float, rBG1: float) -> list[float]:
    """Build a 5-vector at the given p_win, keeping gammon/BG ratios fixed."""
    p = max(0.0, min(1.0, p_win))
    return [
        p,
        p * (rG0 + rBG0),
        p * rBG0,
        (1.0 - p) * (rG1 + rBG1),
        (1.0 - p) * rBG1,
    ]


def _owner_enum(owner: str):
    import bgbot_cpp
    return {
        "centered": bgbot_cpp.CubeOwner.CENTERED,
        "player": bgbot_cpp.CubeOwner.PLAYER,
        "opponent": bgbot_cpp.CubeOwner.OPPONENT,
    }[owner]


def _flip_owner_str(owner: str) -> str:
    if owner == "player":
        return "opponent"
    if owner == "opponent":
        return "player"
    return "centered"


def _binary_transition(predicate, lo: float = 0.0, hi: float = 1.0,
                        iters: int = 50) -> float:
    """Find the p ∈ [lo, hi] where predicate(p) transitions from False to True.

    Assumes predicate is monotonic on [lo, hi]. If predicate(lo) is already
    True, returns lo; if predicate(hi) is still False, returns hi.
    """
    if predicate(lo):
        return lo
    if not predicate(hi):
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if predicate(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _janowski_thresholds(
    cube_owner: str,
    cube_value: int, away1: int, away2: int, is_crawford: bool,
    cube_life_index: float,
    rG0: float, rBG0: float, rG1: float, rBG1: float,
) -> dict:
    """Return {tp, cp, dp, tg} at the user's cube_life_index via binary search.

    Each threshold comes from binary-searching ``cube_decision_1ply`` at the
    user's x. For cube_life_index=0 this matches the dead-cube MET formulas;
    for cube_life_index=1 it matches the recursive live cash points. In
    between it is the exact Janowski interpolation of the piecewise-linear
    cl2cf_match equity curves.

    Returns NaN for action rows that are not applicable under the current
    ownership (e.g. TP when the player owns the cube).
    """
    import bgbot_cpp
    x = _clamp01(cube_life_index)
    player_owner = _owner_enum(cube_owner)
    opp_owner_str = _flip_owner_str(cube_owner)
    opp_owner = _owner_enum(opp_owner_str)

    def player_cd(p: float):
        probs = _probs_from_ratios(p, rG0, rBG0, rG1, rBG1)
        return bgbot_cpp.cube_decision_1ply(
            probs, cube_value, player_owner, x,
            away1, away2, is_crawford, False,
        )

    def opp_cd(p_player: float):
        # opp's perspective: p_opp = 1 - p_player; opp's gammon ratios = player's loss ratios.
        opp_probs = _probs_from_ratios(1.0 - p_player, rG1, rBG1, rG0, rBG0)
        return bgbot_cpp.cube_decision_1ply(
            opp_probs, cube_value, opp_owner, x,
            away2, away1, is_crawford, False,
        )

    # ── Take point (player's own P(win) below which they'd pass) ──
    # From opp's cube_decision, should_take transitions False → True as p_player
    # increases (opp's DT decreases faster than their DP).
    tp = NAN
    if cube_owner != "player":
        tp = _binary_transition(lambda p: opp_cd(p).should_take)

    # ── Cash point: opp would pass when should_take flips True → False ──
    cp = NAN
    if cube_owner != "opponent":
        cp = _binary_transition(lambda p: not player_cd(p).should_take)

    # ── Double point: should_double False → True inside [0, cp] ──
    #
    # should_double is not globally monotonic — it flips False → True at DP
    # and back True → False at TG — so searching [0, cp] keeps the monotonic
    # section.  In the degenerate gammonless live-cube case at x=1, where
    # the cube is dead after a redouble, should_double is nowhere True
    # strictly and the binary search returns cp (DP collapses to CP, same as
    # money-game Janowski at x=1).
    dp_p = NAN
    if cube_owner != "opponent":
        dp_p = _binary_transition(
            lambda p: player_cd(p).should_double,
            lo=0.0, hi=max(cp, 1e-6) if cp == cp else 1.0,
        )

    # ── Too-good point: above this P(win) the player plays on for gammon ──
    # Defined by equity_nd > equity_dp; search [cp, 1]. If the player is
    # never "too good" (gammonless money-style), return 1.0.
    tg = NAN
    if cube_owner != "opponent":
        def too_good_at(p: float) -> bool:
            cd = player_cd(p)
            return cd.equity_nd > cd.equity_dp
        lo = cp if cp == cp else 0.0
        if not too_good_at(1.0):
            tg = 1.0
        else:
            tg = _binary_transition(too_good_at, lo=lo, hi=1.0)

    return {"tp": tp, "cp": cp, "dp": dp_p, "tg": tg}


# ─── Public entry point ─────────────────────────────────────────────


def cube_action_points(
    probs: Sequence[float],
    cube_owner: str,
    cube_value: int = 1,
    away1: int = 0,
    away2: int = 0,
    is_crawford: bool = False,
    cube_life_index: float = 0.68,
    jacoby: bool = True,
    beaver: bool = False,
    no_gammons: bool = False,
) -> PlayerCubePoints:
    """Cube action points (Take / Double / Cash / Too Good) from this player's
    perspective.

    Arguments:
        probs: 5-probability vector (P(win), P(gw+bgw), P(bgw), P(gl+bgl), P(bgl)).
        cube_owner: 'centered', 'player' (this player owns), or 'opponent'.
        cube_value: Current cube value (1, 2, 4, ...).
        away1: This player's away-score. 0 (together with ``away2=0``) means a
            money / unlimited game.
        away2: Opponent's away-score.
        is_crawford: Crawford game flag (match play only). When true, all
            four rows are N/A because the cube can't be doubled.
        cube_life_index: ``x`` ∈ [0, 1]. 0 = dead cube, 1 = live cube,
            ~0.68 typical contact.
        jacoby, beaver: Money-game rules (ignored for match play).
        no_gammons: If true, force ``W = L = 1`` and zero all gammon ratios
            (money game). For match play this still strips gammons from the
            probs before computing the match W/L.

    Returns a :class:`PlayerCubePoints`.
    """
    if cube_owner not in ("centered", "player", "opponent"):
        raise ValueError(f"invalid cube_owner: {cube_owner!r}")

    is_money = (away1 <= 0 or away2 <= 0)

    if no_gammons:
        # Zero gammon/backgammon components. Recall the cumulative convention:
        #   probs[1] = P(gammon OR bg win)   probs[2] = P(bg win)
        #   probs[3] = P(gammon OR bg loss)  probs[4] = P(bg loss)
        # Gammonless means every gammon/bg probability is zero.
        p_win = probs[0]
        probs = (p_win, 0.0, 0.0, 0.0, 0.0)

    if is_money:
        return _money_points(
            probs=probs,
            cube_owner=cube_owner,
            cube_life_index=cube_life_index,
            jacoby=jacoby,
            beaver=beaver,
            no_gammons=no_gammons,
        )

    # ── Match play ────────────────────────────────────────────────
    W, L = _compute_wl(probs)
    rG0, rBG0, rG1, rBG1 = _gammon_ratios(probs)

    # Cube is dead during Crawford; report all rows as N/A.
    if is_crawford:
        return PlayerCubePoints(
            W=W, L=L, double_label=None,
            cannot_take=True, cannot_double=True,
            cannot_cash=True, cannot_too_good=True,
            take=_NA_TRIPLE, double=_NA_TRIPLE,
            cash=_NA_TRIPLE, too_good=_NA_TRIPLE,
        )

    cv = cube_value
    x = _clamp01(cube_life_index)

    # Cash points from recursive GetPoints. Passing gammon ratios from this
    # player's perspective; the C++ helper interprets them matching
    # cl2cf_match_centered.
    player_cp, opp_cp = _live_cash_points(
        away1, away2, cv, is_crawford, rG0, rBG0, rG1, rBG1
    )

    # In the recursive linear-equity model the LIVE thresholds collapse:
    #
    #   live TP = 1 - opp_cp                        (opp_tg in player's space)
    #   live CP = live TG = live DP = player_cp
    #
    # The collapse of CP/TG/DP mirrors the money-game Janowski formulas at
    # x=1: ``(L+1)/(W+L+0.5) = ID = CP = TG``. At x=0 (dead cube) these
    # three are distinct. Janowski interpolation recovers intermediate values.
    live_tp = _clamp01(1.0 - opp_cp)
    live_cp = _clamp01(player_cp)
    live_tg = live_cp
    live_dp = live_cp

    # Dead-cube values from exact MET arithmetic.
    dead_tp = _clamp01(_dead_take(away1, away2, cv, rG0, rBG0, rG1, rBG1))
    dead_cp = _clamp01(_dead_cash(away1, away2, cv, rG0, rBG0, rG1, rBG1))
    dead_tg = _clamp01(_dead_too_good(away1, away2, cv, rG0, rBG0, rG1, rBG1))
    dead_dp = _clamp01(_dead_double(away1, away2, cv, rG0, rBG0, rG1, rBG1,
                                     centered=(cube_owner == "centered")))

    # Janowski column (user's x) uses the exact equity-space interpolation
    # that cl2cf_match performs internally — binary-searched via
    # cube_decision_1ply so the threshold is the equivalent of the money-game
    # closed-form ``(L-0.5)/(W+L+0.5x)`` shape but for match play.
    jan = _janowski_thresholds(
        cube_owner=cube_owner, cube_value=cv,
        away1=away1, away2=away2, is_crawford=is_crawford,
        cube_life_index=x,
        rG0=rG0, rBG0=rBG0, rG1=rG1, rBG1=rBG1,
    )

    def triple(dead: float, live: float, jan_val: float) -> CubePointTriple:
        return CubePointTriple(
            dead=dead,
            live=live,
            janowski=_clamp01(jan_val) if jan_val == jan_val else NAN,
        )

    cannot_take = cube_owner == "player"
    cannot_decide = cube_owner == "opponent"

    take = _NA_TRIPLE if cannot_take else triple(dead_tp, live_tp, jan["tp"])
    cash = _NA_TRIPLE if cannot_decide else triple(dead_cp, live_cp, jan["cp"])
    too_good = (_NA_TRIPLE if cannot_decide
                else triple(dead_tg, live_tg, jan["tg"]))
    if cannot_decide:
        double = _NA_TRIPLE
        double_label = None
    else:
        double = triple(dead_dp, live_dp, jan["dp"])
        double_label = "Redouble" if cube_owner == "player" else "Double"

    return PlayerCubePoints(
        W=W, L=L,
        double_label=double_label,
        cannot_take=cannot_take,
        cannot_double=cannot_decide,
        cannot_cash=cannot_decide,
        cannot_too_good=cannot_decide,
        take=take, double=double, cash=cash, too_good=too_good,
    )
