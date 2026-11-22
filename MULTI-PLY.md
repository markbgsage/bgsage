# Multi-ply evaluation

The public analyzer evaluates checker play, pre-roll cube actions and positions
immediately after a move. This guide describes how the CPU engine computes
their probabilities and equities, including the searches that run inside
truncated and full-game rollouts. The [analyzer](python/bgsage/analyzer.py)
defines the public results. The [rollout specification](ROLLOUT.md) describes
trial simulation, dice generation and uncertainty estimates.

| To understand | Start here |
| --- | --- |
| A returned value or board perspective | [Analytics calls](#1-analytics-calls-and-search-depth) and [probability conventions](#2-probabilities-perspective-and-board-constraints) |
| A cube branch, move filter or selected network | [Cube recursion](#4-n-ply-cubeful-recursion) and [checker selection](#6-checker-selection-within-the-cube-tree) |
| A 1T, 2T or 3T result | [Trial settings](#9-multi-ply-search-inside-rollouts) and [reference measurements](#11-accuracy-and-measurements) |

## 1. Analytics calls and search depth

A deterministic 1-ply evaluation (1P) uses an endpoint probability evaluator.
A 2P pre-roll search enumerates one roll and evaluates its continuation at
1P. A 3P search enumerates the roller's roll and the opponent's reply, up to
21 × 21 roll pairs. At 4P it enumerates one more half-move. Terminal positions
can finish earlier. Within each roll only the selected checker play continues.
The public deterministic levels are `1ply`, `2ply`, `3ply` and `4ply`.

| Public call at 2P–4P | Position and perspective | Returned probabilities | Reported equity |
| --- | --- | --- | --- |
| `checker_play`, cubeful | Pre-move board and known dice. One result per legal endpoint, in the mover's perspective. | One cube-aware tree per endpoint, beginning just before the opponent rolls. | Cubeless equity from the five probabilities and cubeful equity directly from the same tree. |
| `cube_action` | Pre-roll board, in the player-on-roll perspective. | An independent cubeless search from the same physical position. | Cubeless equity from that probability vector. No-double, double/take, double/pass and optimal equity from the cube decision. |
| `post_move_analytics` | Post-move board, in the perspective of the player who just moved. | A cube-aware tree from the opponent's pre-roll position when an owner is supplied. | Cubeless equity from those probabilities. The public cubeful scalar applies Janowski to the resulting mover probabilities on the supplied board. |

All three calls take a 26-element board and the cube context. Ownership is
relative to the board perspective, with `centered`, `player` and `opponent`
as the public values. Away scores count points needed to win. Both are zero
for unlimited play. Checker play and cube action also accept the beaver flag
and a maximum cube value (zero for unlimited). Post-move accepts cube value,
owner, away scores, Crawford and Jacoby. Match play disables Jacoby and beavers.

To evaluate the position immediately preceding a cube decision from the
previous mover's perspective, flip the cube decision's board, swap away scores
and flip ownership. A complement or sign change alone cannot convert the
headline values of the two APIs. The searches use different checker policies,
and the public post-move scalar applies a nonlinear conversion after averaging.

## 2. Probabilities, perspective and board constraints

A post-move evaluator produces five cumulative probabilities in the mover's
perspective. With `p = [W, GW, BW, GL, BL]`, `W` includes all wins, `GW`
includes backgammon wins, and `GL` includes backgammon losses. The single-win
probability is `W − GW`, gammon-only wins are `GW − BW`, and similarly for
losses. Use these disjoint outcomes when weighting a match equity table.

```text
invert(p) = [1 − W, GL, BL, GW, BW]
cubeless_equity(p) = 2W − 1 + GW − GL + BW − BL
pre_roll_1P(board) = invert(post_move_1P(flip(board)))
```

The first two formulas change perspective and convert points to current-cube
units. The third accounts for *whose turn is next*. Looking up the same
orientation as both a pre-roll and a post-move board changes the tempo.

The board clamp zeros gammon and backgammon losses once the mover has borne
off a checker, and zeros gammon and backgammon wins once the opponent has
borne one off. Once contact breaks, a player without a checker on the bar
or in the opponent's home board cannot lose a backgammon. The analogous
condition on the opponent rules out a backgammon win.

In the main cube tree, batched interior picks use the evaluator's candidate
outputs directly. The leaf inverts the chosen post-move probabilities, clamps
against the pre-roll board and applies the cube calculation. Internal nodes
weight and invert their children's probabilities. The standalone cubeless
post-move entry also clamps its final result. Terminals supply their exact
result vector. These details matter when reproducing a selected line near an
equity tie. [Board and probability helpers](cpp/include/bgbot/strategy.h)

## 3. Leaf equity and cube rules

The leaf combines a dead-cube estimate and a live-cube estimate using Janowski's
cube efficiency `x`. Contact positions use 0.68. Race positions use
`clamp(0.55 + 0.00125 × average_pips, 0.60, 0.70)`, with both players'
pip counts in the average. Native callers can supply a cube-efficiency
override. [Equity functions](cpp/src/cube.cpp)

### Unlimited play

The expected size of a win is `w = 1 + (GW + BW) / W`. The expected size of
a loss is `l = 1 + (GL + BL) / (1 − W)`. A denominator at or below
`1e-7` gives the default size 1 for that side. The live take point is
`TP = (l − 0.5) / (w + l + 0.5)` and the cash point is
`CP = (l + 1) / (w + l + 0.5)`. The live equity linearly interpolates
between the following points as `W` increases.

| Owner | Knots (win probability, live equity) |
| --- | --- |
| Centered | (0, −l), (TP, −1), (CP, +1), (1, +w) |
| Player | (0, −l), (CP, +1), (1, +w) |
| Opponent | (0, −l), (TP, −1), (1, +w) |

With Jacoby active on a centered cube, the two outer live segments are
constant at −1 and +1. The dead term is `2W − 1`. With Jacoby inactive the
dead term is ordinary cubeless equity including gammons. The blended estimate
is `(1 − x) × dead + x × live`. Turning the cube gives it an owner and activates
gammon scoring. With an active centered Jacoby cube, terminal awards also
use single-game points.

At 1P, no-double uses the current cube conversion. Double/take uses twice
the conversion with the opponent owning the doubled cube. Double/pass is +1
current-cube unit. When beavers are enabled, a negative double/take value
from the doubler's perspective doubles again and carries the beaver flag.
Cube availability and the configured cap govern which actions the roller can
take. A structurally capped cube follows the engine's cubeless-value path.

### Match play

The recursive unit is match winning chance (MWC). For each of the six
disjoint cubeless outcomes, award one, two or three times the cube value to
the winner and look up the resulting match equity in the Kazaross-XG2 table.
Their weighted mean is dead-cube MWC. Passing awards the current cube value
to the doubler. For a given score, cube and Crawford state, let `Mwin` and
`Mlose` be the match equities after a single win or loss. Then

```text
MWC(e) = ((Mwin − Mlose) × e + Mwin + Mlose) / 2
e(MWC) = (2 × MWC − Mwin − Mlose) / (Mwin − Mlose)
```

The live term interpolates in MWC space using score-dependent cash points
for both players. `get_match_points` computes them from the highest relevant
cube level down to the current level, accounting for gammon rates and
automatic redoubles. Centered cubes have three interpolation regions and
owned cubes two. The leaf blends dead and live MWC at the same `x` as above.
The public equity conversion uses the score and cube appropriate to the
reported value. [Cube conversion](cpp/src/cube.cpp) and [match equity table](cpp/src/match_equity.cpp)

Crawford games have a dead cube. For an owned cube, match equity is dead
when the owner's away score is at most the current cube value. For a centered
cube both players' away scores must be covered. These match rules live in
the Janowski conversion and doubling-permission checks. The lower-level
`cube_is_dead` predicate used for cache sharing and trial dispatch tests the
structural cube cap. A match position can consequently retain match-aware
trial handling while its conversion has no live-cube premium.

## 4. N-ply cubeful recursion

`cubeful_equity_nply_multi` evaluates an array of cube states on one pre-roll
board. Each state includes value, owner, away scores, Crawford, rule options,
cap and optional efficiency override. The core returns one equity per state
and a single vector of cubeless outcome probabilities along the shared
checker line. A cube action starts with separate no-double and double/take
states. An evaluation of an endpoint's continuation starts with its single
current state and allows the next roller to decide whether to double.
[Native search](cpp/src/cube_eval.cpp)

1. Look up the internal node using the board, remaining ply and complete
   evaluation context. A terminal board supplies its known outcome to each
   valid cube state. At 1P, evaluate the flipped post-move board once, invert
   and clamp the probabilities, then compute all leaf equities.
2. Expand each state into a no-double branch and, when permitted, a
   double/take branch at twice the cube with the opponent owning it. An
   unavailable state has `cube_value = −1`. At the root of a cube decision
   the ND/DT pair already exists, so the root suppresses another double.
3. For each of the 21 unordered rolls, generate legal endpoints and choose
   one move by the rule in section 6. All active cube states share that
   endpoint. The cube states swap owners and away scores when the turn changes.
   If a roll cannot be played the unchanged board passes to the opponent.
4. If the move ends the game, score it immediately for every state. If it
   leaves a 1P leaf, use the selected candidate's post-move vector. A forced
   single endpoint gets its own leaf evaluation. Otherwise recurse from the
   flipped board at one less ply.
5. Accumulate in a fixed roll order. Doubles have weight 1, the other 15
   rolls have weight 2, and the divisor is 36. Convert the opponent's mean to
   the roller's value by negating unlimited equity or complementing MWC.
   Invert the averaged probability vector and restore each cube's perspective.
6. Collapse each pair to `max(ND, min(DT, DP))` when doubling is available,
   or ND otherwise. In unlimited play scale DT by two relative to ND, apply
   the beaver rule if applicable, and use DP = +1. In match play compare
   raw MWC with the MWC of passing. Store the resulting state values and
   shared probabilities.

For the single-state endpoint entry, there is one primary state for checker
selection. For the multi-state entry the first supplied state is primary.
Thus ND and DT may have different equity along the *same* board sequence.
Solving a separate best checker line under every alternative cube would be
a different policy. The probability vector describes game outcomes along
the selected line. A cube pass has a point payoff and does not become a
terminal checker-game outcome in that probability vector.

The native work arrays use `EVAL_MAX_CCI = 64` and space for a two-way
expansion. Each half-move doubles the state-slot count, including unavailable
slots. The actual public 1P–4P searches use small starting arrays.

## 5. Cube action and optional roll details

At the top-level cube decision, ND uses the current cube and DT uses the
opponent-owned doubled cube. Each is evaluated at the requested depth. The
roller doubles when `min(DT, DP) > ND`, and the opponent takes when
`DT ≤ DP`. A roller unable to double keeps ND. A post-Crawford trailer
against a one-away opponent automatically doubles when entitled to do so.
The ordinary match decision compares raw MWC first and presents all three
headline equities in the initial cube's normalized units.

The public cube call loads the model and bearoff database on the analyzer and
passes the prepared strategy into the native binding. Standalone binding
callers can also supply a model description. The public cubeless probabilities
come from a second traversal that plays both sides under a dead unlimited
cube. Native per-cube values and the headline probability vector therefore
have different selection objectives. [Cube binding](cpp/pybind/bindings.cpp)

### Reusing candidate evaluations at 3P and 4P

An ordinary 3P or 4P cube call can reuse the live traversal's candidate
probabilities in the subsequent cubeless traversal. The table belongs to
one cube call. Each sequence of rolls identifies a pre-roll node. At 3P the
table covers the root and its 21 children. At 4P it also covers their 441
children. Each node has a slot for each of the 21 rolls. The live traversal
allocates the table only when it searches the root. A root transposition
cache hit skips the allocation. On x64 the allocated table occupies about
181 KiB at 3P or 3.72 MiB at 4P.

Each slot holds raw probabilities for up to 16 candidates. A complete
unfiltered list of 2–16 candidates can be copied directly. For a larger
list filtered to 15 by the live search, a built-in pair evaluator using the
pre-roll neural base can also save the surviving candidates' original move
indices. Its unfiltered cubeless reader uses those probabilities and evaluates
every missing candidate. Reuse requires the same dice path, full pre-roll board, roll,
evaluator and pinned network. The board and roll reproduce the entire
ordered batch through deterministic move generation. The live traversal
finishes before the cubeless traversal reads the table. Each traversal
scores and selects its own moves. If the boards diverge, the reader filters
its own list, or a partial batch uses an ineligible evaluator, the reader
evaluates its own candidates.
Root bearoffs and terminals, full-depth opponent search, a separate filter
strategy and the per-roll detail path use their own evaluation routines.
[Batch identity and reuse](cpp/src/cube_eval.cpp)

### Roll details

`incl_2ply_details=True` on a deterministic cube call requires 2P or above.
The dedicated detail traversal records 21 initial rolls under each of ND and
DT. Each roll gives dice, selected post-move board, probabilities, cubeful
equity, and terminal/pass flags. At 2P the next node is a leaf. At 3P and
above, a surviving line has 21 opponent-roll entries, with boards and values
in the initial roller's perspective. A terminal or opponent double/pass
cuts off the reply list. Details use cubeless selection for the explicitly
recorded rolls. The ordinary cube tree uses cube-aware selection. The two
modes can choose different moves and give different equities.

Unlimited detail equities scale the applicable branch to the initial cube.
The headline DT value includes any beaver adjustment. Match detail rows use
the cube value of the displayed branch for their MWC-to-equity conversion.
For a match DT line, convert row values back to MWC before comparing a
weighted mean to the headline DT equity on the initial cube.
[Detail construction and conversions](cpp/src/cube.cpp)

## 6. Checker selection within the cube tree

**Legal moves.** The hot path uses `possible_boards_unsorted`. It tries both
orders of unlike dice and plays the higher die if only one can be used.
For doubles it tries four, then three, then two, then one usable die, stopping
at the first legal length. A terminal win can finish before all dice are used.
A move sequence updates one board in place, carries a mask of occupied
origins and restores the board after each branch. For doubles, the generator
emits the earliest legal ordering of each set of checker departures. Once an
earlier sibling has played a legal origin, later siblings can skip sequences
that would play that same origin. For unlike dice, the reverse order explores
second steps from newly available origins. A growing thread-local table
checks the incremental hash and full board to deduplicate the endpoints.
The search consumes first-seen order. Callers requiring lexicographic order
sort the output of the same generator. [Move generator](cpp/src/moves.cpp)

**Filtering and ranking.** A caller-supplied cheap evaluator, normally PubEval,
can rank rolls with more than 16 endpoints and keep the best 15. When
`deep_prefilter` is enabled, contact nodes strictly below the entry depth
use PubEval at more than 16 endpoints and keep 14 by default. The environment
variables `BGBOT_CUBEFUL_DEEP_THRESHOLD` and `BGBOT_CUBEFUL_DEEP_KEEP`
set these process-level deep limits. Race nodes skip the extra deep rule.
Terminal endpoints receive priority during prefiltering.

| Tree entry | Entry filter and ordinary interior nodes | Extra deep filter |
| --- | --- | --- |
| Public cubeful checker and post-move continuations | No PubEval supplied | Off |
| Public cube decision, live tree | PubEval, above 16 keep 15 | Off |
| Public cube headline cubeless tree | No PubEval supplied by this binding | Off |
| Standalone cubeless `MultiPlyStrategy` from `create_multipy` | Factory supplies PubEval, above 16 keep 15 | Off |
| Native checker survivor rescoring and eligible in-trial cube or truncation calls | Supplied move prefilter if present | On where requested, contact descendants above 16 keep 14 |

After prefiltering, each survivor gets a 1P probability vector. The main tree
computes `cl2cf` relative to the primary cube, using the candidate's race and
pip counts. Find the maximum across the complete evaluated set, then consider
every candidate within `1e-6` of that maximum. Select the one with the highest
unlimited cubeless equity. A remaining exact tie retains the first such
candidate. Only the winner continues to deeper ply. A match score can make
several moves equivalent in MWC while their gammon probabilities differ.
[Selection rule](cpp/include/bgbot/cube.h)

**Routing and batching.** Each terminal candidate supplies exact terminal
probabilities. An applicable bearoff table supplies its stored probabilities.
The kernel examines each layer of a bearoff-wrapped strategy and passes
uncovered boards to its base evaluator. Pair-network strategies group live
candidates by network index. By default they classify each candidate board.
A configured `root_pin_for` can fix the contact network for the entire search.
The public cubeful checker call passes the decision's pre-move board. The live
cube decision uses the roller's pre-roll board. Its independent cubeless pass
uses the flipped board as its root. The public post-move probability entry
does not supply a root pin. Race positions retain race routing, and bearoff
lookups precede neural routing.

The production `stage11` configuration has a 100-hidden-unit race net
(index 0) and 400-hidden-unit contact nets. For a contact board, routing
checks for a snake (22), containment game (21) and massive backgame (23),
in that order. Otherwise it classifies both players' game plans, reusing
one pair of pip counts. An eligible deep, middle or double-anchor backgame
uses 17, 18 or 19 respectively. Early containment uses 20. The remaining
positions use the plan-pair networks 1–16. By default, only the snake
specialist is pinned when a supplied root selects it. Other model layouts
can offer different specialists. A strategy with blended networks evaluates
a blended candidate individually. [Model registry](python/bgsage/weights.py)
and [routing implementation](cpp/src/neural_net.cpp)

**CPU evaluation.** In standalone trees and searches inside rollout trials, a worker encodes the pre-roll board
and saves its pre-activation hidden sums once for each network it uses at
that node. All 21 dice rolls have that same starting board. Each worker keeps
separate scratch space by depth and network and tags it with the node's
unique identity. The deep rollout prefilter selects which candidates receive
a neural evaluation. For each candidate
evaluated by differences, the kernel compares the 24 board points to its base and encodes point features
only where the checker count changed. It recomputes each side's auxiliary
features (two per side for race, 26 for contact) and compares them to the
saved inputs. The evaluator gathers nonzero differences in ascending input
order and applies their weight columns to a copy of the saved hidden sums.
AVX2 applies four successive columns per hidden-vector load in that order.
At 64 changed inputs the candidate gets a full forward pass.
[Neural kernel](cpp/src/neural_net.cpp)

Pure race uses 196 Tesauro inputs. Contact uses 244 inputs. The contact encoder
forms masks of occupied points, made points, enemy blots and blocks. On AVX2
it extracts both sides' masks in one scan and reverses the opponent's point
order. The scalar path constructs each side's masks in its own perspective.
Off the bar, only blots in front of the rearmost checker can be hit. The AVX2
kernel evaluates the six doubles and fifteen mixed rolls in three groups of
lanes. The highest eligible target for each roll determines its pip loss.
The portable path enumerates the rolls in scalar code. With checkers on the
bar, both paths account for entry, blocked intermediate points and direct
hits. Escape, containment,
mobility, anchor and entry features also use the point masks where applicable.
The separate `double_hitting_shots` helper has a compile-time table indexed by
two hit distances. It stores the double and number of steps needed on each leg.
These operations implement the feature definitions consumed by the trained
networks. [Feature encoding](cpp/src/encoding.cpp)

## 7. Public checker analysis and cubeless search

The default cubeful checker analysis enumerates **every legal endpoint** at
2P–4P in one native call. The engine shares the loaded evaluator and bearoff
database across the candidates. For each endpoint it flips the board, owner
and away scores. The native batch computes both probabilities and equity in
one tree traversal per endpoint. Parallel 4P checker searches can converge
after two turns. A table scoped to this checker decision lets workers reuse
the complete 2P result for the same board and four cube states. Returning to the mover complements
the five probabilities and negates the opponent's normalized cubeful equity.
Each row gets the requested evaluation level. Sorting uses descending cubeful
equity, descending cubeless equity and then the board as a deterministic final
key. The reported equity difference is the row's equity minus the best equity.
A legal forced board is already in this set.

At 1P the analyzer evaluates each legal move and converts its probabilities
to cubeful equity on the calling thread. A pre-roll 1P cube decision uses the
analyzer's loaded network weights and bearoff database.

The Python checker analyzer without the cubeful wrapper scores the endpoints
at 1P, takes up to five within a strict 0.08 equity gap, and includes any legal
forced boards. The initial screening score uses a cube conversion if an owner
is supplied. Survivors receive deep cubeless values. Other moves initially
retain their 1P probabilities. The analyzer repeatedly evaluates a shallow
move appearing in the top two, re-sorting after each promotion. Read each
row's `eval_level` when using these results.

For a standalone cubeless evaluation, `MultiPlyStrategy::cubeless_tree_probs`
represents cubeless play as one unlimited cube with value and cap both set to
1. The pre-roll tree selects on cubeless equity throughout. The public
post-move strategy flips the board into that tree and inverts its returned
probabilities. A covered root can read post-move bearoff probabilities directly.
An explicitly configured separate filter model or full-depth-opponent mode
uses the dedicated cubeless recursion described below.
[Checker strategy](cpp/src/multipy.cpp)

## 8. Native best-move search and optional search modes

Native `best_move_index` selects a single checker play, including when a
rollout needs a move. It has a different work budget from a public list of
fully scored cubeful alternatives. With 2P or 3P, the filter chain first
scores at 1P and keeps at most `max_moves` within the filter's equity
threshold. A single survivor wins immediately. Several survivors receive the
target-ply evaluation. At 4P and higher there is also an intermediate
`N−1` step with `max(2, floor(2 × max_moves / 5))` moves and a gap of
`max(0.01, threshold / 4)`.

| Native preset | Maximum moves | Equity gap |
| --- | ---: | ---: |
| TINY | 5 | 0.08 |
| NARROW | 8 | 0.12 |
| NORMAL | 8 | 0.16 |
| LARGE | 16 | 0.32 |
| HUGE | 20 | 0.44 |

For cubeful selection over several trial cubes, a shared 1P batch supplies
all candidate probabilities. Each cube filters on its own cubeful ranking,
breaking exact score ties on cubeless equity. The union advances to deeper
search. The optional 4P intermediate step ranks this union on cubeless
values. The final rescore carries the opponent's flipped cube states for
each remaining board. Each cube independently picks the lowest opponent
equity, using the mover's 1P cubeless equity for ties within `1e-6`.
Trial-internal multi-ply checker strategies use a narrower first filter of
two moves and 0.03 equity. [Filter definitions](cpp/include/bgbot/multipy.h)
and [multi-cube move selection](cpp/src/multipy.cpp)

The dedicated cubeless recursion supports an optional inexpensive filter
model and a full-depth-opponent mode. It works on post-move boards. For all
21 opponent rolls it chooses a reply, evaluates it with the base model at
the remaining depth, inverts the reply probabilities and averages them.
In fast hybrid mode the filter model handles the 1P selection and the base
model handles the leaf, each using pre-move routing context. The full-depth
option ranks the surviving opponent choices at `plies − 1` when more than two
plies remain. A configured cheap move prefilter in this recursion defaults
to more than 20 candidates, keeping 15. With a single evaluator a 2P node
can reuse the selected reply's batched probabilities at the leaf.

## 9. Multi-ply search inside rollouts

The named T-levels are truncated rollouts. The following are public preset
values. Half-move numbers start at zero. Cube strength is the configured
strength of the decision after its inexpensive screen.

| Level | Trials | Truncate after | Configured checker depth | Cube decision throughout | Truncation evaluation |
| --- | ---: | ---: | --- | --- | --- |
| 1T (`truncated1`) | 72 | 5 | 1P throughout | 1P | 1P |
| 2T (`truncated2`) | 360 | 7 | 2P on move 0, then 1P | 2P | 2P |
| 3T (`truncated3`) | 360 | 7 | 3P on moves 0 and 1, then 2P | 3P | 3P |

The second checker play (half-move 1) is a shared exception. All three
presets use the first roll's precomputed reply table, which selects each of
the 21 replies at **1P**. In a normal 3T trial the effective checker depths
are therefore 3P on move 0, 1P on move 1 and 2P on subsequent moves.
Full-game rollouts use the same reply-table convention.
[Move1 cache construction](cpp/src/rollout.cpp)

The early cube configuration controls take/pass for the entire trial.
The late checker threshold changes checker strength. The public default
enables cube-aware trial moves. At 1T the ultra-late threshold of 2 switches
the move selector to the cubeless base from half-move 2. At 2T and 3T the
ultra-late threshold is 9999. A full-game `rollout` with truncation depth
zero plays to completion subject to the engine's stopping rules. Its
configurable defaults include 1,296 trials and 1P decision strength. The
accuracy corpus specifically uses 3P checker and cube decisions.
[Preset construction](python/bgsage/analyzer.py) and [trial loop](cpp/src/rollout.cpp)

At the public checker root, 1T runs one 1P shortlist of up to five within
0.08. At 2T/3T the 1P pass first keeps all moves within 0.15. If several
survive, a 2P rescore keeps up to five within 0.08. The runner includes a
second comparison move when available and all legal forced boards. Unrolled
rows retain their fallback probabilities and level. The cubeful wrapper assigns
a fallback cubeful equity at decision depth and rolls out an unrolled leader,
repeating until the displayed leader has a rollout. With cube-aware trial
moves enabled, post-move probabilities and cubeful equity are both trial
outputs, and can include standard errors.

Each preset uses stratified dice and 1P variance reduction. For the mean,
the engine selects a 1P move for each of the 21 rolls under the applicable
cube-aware or cubeless policy and weights the results. For the realized
roll it takes the 1P value of the **move actually played**, which can be
different when the trial uses a deeper checker strategy. It tracks the
difference in a consistent player perspective and subtracts accumulated
luck from the trial result, separately for probabilities and branch
equities. The first roll's correction can be skipped when the trial count is
a multiple of 36. At 1,296 games the first two ordered
rolls are jointly stratified. `set_seed` resets the rollout dice sequence and
retains reusable position evaluations.

During a trial, a cheap 1P screen checks available doubles and escalates flagged
cube states together at the configured decision strength. The endgame screen
also checks `x = 0` when each side has at most two checkers and at least one is
within 24 pips. Cubeful multi-ply truncation evaluates live branch states
together and obtains continuation probabilities in the same walk. Cubeless
truncation uses a dead cube or the direct 1P path as appropriate. Multiple
trials share the limited set of early roll positions, with cache entries
specific to their applicable board and cube context. The parallel, single-batch
cubeful path reserves all 21 first-roll cache slots and any scheduled
second-move slots before dispatch. Each first-roll index has one worker
that computes the chosen endpoint, publishes it, and fills that roll's
second-move entry when used. Readers wait for their own slot's release/acquire
publication. Trials can start while workers finish other roll indices.
[Rollout specification](ROLLOUT.md) and [cache publication](cpp/src/rollout.cpp)

## 10. Caching and threading

| Cache | Scope and contents | Reuse conditions |
| --- | --- | --- |
| Cubeful recursion | Per native thread, 32,768 slots, up to four probes. An entry contains the full board, five probabilities and up to 16 state values. | Internal nodes with at least two plies and at most 16 cube states. Match the full board, ply, state count, top flag and context fingerprint. |
| Cubeless `PosCache` | Per thread, 262,144 slots allocated on first insertion, up to eight probes, clears at 75% occupancy. Used by the dedicated hybrid/full-depth recursion. | Board equality and a key salted with the base strategy, ply and full-depth-opponent setting. |
| Rollout `SharedPosCache` | 2,097,152 slots, eight probes. The first search that uses the table allocates it once for all participating trial workers. | A matching key and full board. Dedicated cubeless and structurally dead cube-tree computations use distinct key spaces. |
| Rollout VR move selections | 2,048 slots per position, shared by its trials. Stores the 21 selected indices and their five probabilities. | VR enabled, at least 288 trials and a truncation depth above two. A hit matches the full board, ordered active selection cubes and cube efficiency. |
| 3P and 4P candidate batches | Per cube-action request, 22 nodes at 3P or 463 at 4P, each with 21 roll slots. Allocated on a live root search. | Complete batches of 2–16 candidates, or an eligible filtered subset with its generator indices. The reader applies its own policy. [Reuse conditions](#reusing-candidate-evaluations-at-3p-and-4p) |
| 4P checker subtrees | Per parallel checker decision, 64 independently locked shards of at most 128 completed entries each. | At the 2P layer, the full board and cube fingerprint must match. Each entry stores four equities and the five probabilities, with the model and root pin fixed for the request. |

The main cube cache fingerprints each ordered cube state's value, owner, away
scores, Crawford flag, Jacoby and beaver settings, cap and the exact float bits
of its efficiency override. It also includes evaluator identity, pinned
network, move filter identity, deep-pruning mode and, when deep pruning is on,
whether the node is the entry or a descendant. The model identity is process
specific. Each loaded or generated bearoff database has its own data-version
identifier. Wrappers combine it with the base identity, so equivalent
wrappers around the same database and model can share work.
An available matching/empty slot is used within the four probes. Otherwise
the first slot is replaced. `clear_cache` on a multi-ply strategy clears the
calling thread's local tables. Rollout dispatch clears the relevant worker
tables at operation boundaries. A clear advances the cubeful table's
generation. Lookups stop at the first slot from an earlier generation.
The hybrid table clears populated storage only. [Cube cache](cpp/src/cube_eval.cpp) and
[bearoff identity](cpp/src/bearoff.cpp)

The shared cache claims an empty slot atomically, publishes a board/key as
computing, and releases the result as ready. A worker finding the same pending
board yields at most 256 times, then can compute locally. For internal nodes
whose valid cube states are all structurally dead, the cube tree can recover
each state's money equity or match MWC directly from the cached probabilities.
Live-cube branches still require per-state values. [Shared cache](cpp/include/bgbot/multipy.h)

Standalone cube and post-move trees can distribute the 21 top-level rolls when
the requested thread count is greater than one and the depth is at least 2P.
Linux evaluates a 2P pure-race root on its calling thread. Its race network
and bearoff lookups make the individual roll jobs short enough that waking
helpers costs more than the work they save. Windows distributes those rolls.
With 4–21 requested threads, workers claim the next available roll, starting
with the six doubles. At other parallel widths, each worker processes a fixed
stride through the roll list. Each roll writes a separate slot and a fixed-order
reduction produces the same floating-point accumulation sequence. Child roll parallelism additionally
requires more than 21 requested threads and more than two plies remaining in
the child. The persistent native pool is sized from hardware concurrency and
uses 8 MiB stacks on Windows. Native `MultiPlyStrategy` requests at most the
hardware count, configured count and `4 × (plies − 1)` workers for plies above
one. [Worker scheduling](cpp/src/multipy.cpp)

Public cubeful checker analysis assigns endpoints to workers as they become
available. A single endpoint can use up to 21 workers across its dice rolls.
For a 4P call with multiple endpoints and a budget of 8–21 threads, the
number of endpoint groups is `min(endpoint_count, floor(threads / 2))`.
Each group can use `floor(threads / groups)` workers for its 21 root rolls,
including its caller. The caller can finish any unclaimed rolls itself and
waits only for rolls another worker has claimed. This also permits multiple
simultaneous checker calls to share the pool. Other multi-endpoint calls
evaluate each endpoint tree serially.

Rollouts parallelize across trials and use serial
embedded multi-ply searches, with early cache preparation also able to use
workers. The Python analyzer accepts `parallel_threads` and obtains its
automatic setting from `BGBOT_MULTIPLY_THREADS` or the CPU count. Explicit
positive thread counts on the Python deterministic multi-ply analyzer have
a minimum of two. Native trees can run with one thread. The
separate 2P checker shortlist strategy used by the Python rollout analyzer
takes its own automatic worker setting. These layers matter when measuring
single-call latency or dividing a fixed CPU budget among processes.

## 11. Accuracy and measurements

Dice averaging is exhaustive for the explicit search horizon. The endpoint
evaluator, Janowski interpolation, selective move filters, 1P interior picks,
shared checker line across cube states, and truncation policy all affect
accuracy. Reordering floating-point operations, changing candidate order
for sibling evaluations, or changing a filter can alter individual values
and actions. At close decisions, inspect the equity gap as well as the move
or cube action selected.

The fixed timing corpus has 100 checker boards and 100 cube boards. Each set
has 50 unlimited and 50 match positions, evenly divided among attacking,
priming, anchoring, racing with contact and pure race. The cube boards also
supply 100 post-move calls in the previous mover's perspective. Use the
stored API arguments and save the complete outputs, including all per-move
levels, for timing and numerical comparison. [Benchmark positions and runner](data/multi_ply_benchmark/README.md)

The position-level reference uses full-game 3P checker/cube rollouts with
variance reduction. Its metadata distinguishes fixed 1,296-game references
from archived decisions with an adaptive number of 1,296-game batches.
Match checker alternatives by endpoint board and score only reference moves
actually labeled rollout. For each scalar compare the change in absolute
error to that API's rollout. A deterioration greater than 0.0025 in a
probability (0.25 percentage points) or 0.005 in an equity is material.
Report standard errors where supplied and keep direct output differences
separate from errors relative to rollout. [Reference corpus and coverage](data/multi_ply_benchmark/rollout_positions/README.md)
