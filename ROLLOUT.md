# Rollout and truncated rollout evaluation

A rollout plays a set of seeded trial games from a backgammon position. It estimates five cubeless outcome probabilities and their standard errors. A cubeful rollout also estimates the value of playing on, doubling and taking, or playing a checker move that gives the opponent the opportunity to double. A full rollout ordinarily ends at a terminal result. A truncated rollout evaluates the board after a configured number of half-moves.

## 1. Goal

The three public calls answer different questions about the same game. Their board perspectives determine who acts first and whose equity appears in the result.

| Public analytics | Input perspective | Calculation |
| --- | --- | --- |
| `checker_play` | Player with known dice | Generate the legal post-move boards, screen them, and roll out the selected moves. Report a ranked list from this player's perspective. |
| `cube_action` | Player about to roll | Roll out no double (ND) and double/take (DT) on common dice. Compare those estimates with double/pass (DP) to select a legal cube action. |
| `post_move_analytics` | Player who just moved | Evaluate the opponent's pre-roll position with board, owner and match score flipped. Return probabilities and equity to the original player's perspective. |

The five probabilities are wins, gammon wins, backgammon wins, gammon losses and backgammon losses, in that order. Gammon probabilities include backgammons. With probabilities `(w, gw, bw, gl, bl)`, cubeless equity is `2w − 1 + gw + bw − gl − bl`. Reversing perspective gives `(1 − w, gl, bl, gw, bw)`. A cubeful equity includes the value of the cube under the supplied rules. [Public analyzer](python/bgsage/analyzer.py) · [Native result types](cpp/include/bgbot/rollout.h)

## 2. Full and truncated rollouts

One half-move is one player's roll and checker play. Half-move 0 is the first roll in a trial. A positive `truncation_depth` counts how many half-moves to play before evaluating the board. At depth zero the engine plays toward the end of the game with a safety limit of 200 half-moves. A trial reaching that limit uses the same board evaluation as a truncated trial. A completed game or a branch's pass settles the corresponding value sooner.

### Public presets

| Analyzer level | Trials | Half-moves | Checker at move 0 | Checker at move 1 | Checker from move 2 | Trial cube | Horizon evaluator |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| `truncated1` (1T) | 72 | 5 | 1P | 1P | 1P | 1P | 1P |
| `truncated2` (2T) | 360 | 7 | 2P | 1P | 1P | 2P | 2P |
| `truncated3` (3T) | 360 | 7 | 3P | 1P | 2P | 3P | 3P |
| `rollout`, public defaults | 1,296 | To completion | 1P | 1P | 1P | 1P | 1P |

Here 1P denotes a static evaluation and higher ply counts denote lookahead. The shared opening-reply table selects the second checker move at 1P in all four presets. The 2T late checker threshold is 1 and the 3T threshold is 2. The 1T ultra-late threshold is 2. The 2T, 3T and default full rollouts set it to 9,999. With default cube-aware checker play, that makes 1T choose by cubeless equity from move 2 and leaves 2T and 3T cube-aware at all seven moves. The cube decision itself uses the configured cube strength throughout. A full rollout configured with 3P checker and cube evaluators is distinct from the default full preset. [Preset construction](python/bgsage/analyzer.py) · [Configuration reference](#17-configuration-reference)

## 3. Stratified dice generation

The engine prepares an ordered dice pair for each trial index `t` and half-move `m`. A seeded `std::mt19937` shuffles permutations of the integers 0 through 35. The permutation table has six levels and 128 turn indices. For each level `i`, the engine initializes turns `i` through 127. It advances through each permutation from the first entry, swapping with a pseudo-random entry among the remaining entries.

For `m < 128`, start with `j = 0`. For each level `i` from 0 through `min(5, m)`, set `j = permutation[i][m][(floor(t / 36^i) + j) mod 36]`. The dice are `(floor(j / 6) + 1, (j mod 6) + 1)`. Every block of 36 opening rolls covers the ordered pairs once. With 1,296 trials the first two rolls are jointly stratified over all 36 × 36 ordered-pair combinations. A 360-trial batch has ten complete opening strata.

At half-move 128 and later a per-trial generator draws each die uniformly from 1 through 6. Its seed is `seed + 1000003 × t + 7` in 32-bit unsigned arithmetic. The strategy can reuse its prepared dice across endpoint calls, giving them common random numbers. `set_seed` invalidates the stored sequences. Each trial receives its dice by trial index, independently of which worker claims it. [Dice generation and dispatch](cpp/src/rollout.cpp)

## 4. Variance reduction (VR)

The engine estimates the luck of a sampled roll by comparing a static evaluation of the move played with a weighted static baseline over the possible rolls. Each of the six doubles has weight 1 and each of the fifteen non-doubles has weight 2. The weights sum to 36.

For each roll `r`, generate its legal candidates and select a 1P winner. Let `p[r]` be that winner's five probabilities in mover perspective. A forced pass uses the unchanged board. An exact terminal outcome takes precedence over a network evaluation. The baseline is `mean_p = Σ weight[r] × p[r] / 36`. Its ranking objective follows the trial's cubeful or cubeless selection mode, including the driving branch. A deeper checker policy can select a different move on the actual dice. In that case the engine evaluates the played move at 1P for the correction. The static baseline is an approximation to the deeper policy's expected play.

Let `d = played_p − mean_p`. On the starting player's turn add `d` to their accumulated probability luck. On the opponent's turn add `(−d.win, d.gammon_loss, d.backgammon_loss, d.gammon_win, d.backgammon_win)`. The scalar equity correction uses the difference of the two cubeless equities, positive on the starting player's turn and negated on the opponent's turn. At settlement subtract accumulated luck from the raw outcomes in starting-player perspective. The raw endpoint probabilities are board-clamped before correction. Corrected trial contributions can fall outside the interval from zero to one.

Each unfinished cube branch has its own luck accumulator. For that branch, convert each `p[r]` through the Janowski model **before** weighting. Use the same cube state and efficiency to convert the played probabilities. In unlimited play the value is the cubeful equity multiplied by `current_cube / starting_cube`. In match play it is match-winning chance (MWC). Add the difference between played value and mean on the starting player's turn, and subtract it on the opponent's. Nonlinear cube conversion makes a weighted mean of branch values different from a conversion of the mean probabilities. [Trial implementation](cpp/src/rollout.cpp) · [Cubeful equity and Janowski conversion](MULTI-PLY.md)

The engine skips the move-0 correction when the number of trials is divisible by 36. With at least 288 trials it also skips odd half-moves at or beyond `ultra_late_threshold`. When VR is off, both luck accumulators stay at zero. A ready opening-reply entry and a matching later-board cache entry can supply the probabilities and indices for the 21-roll baseline.

## 5. The unified trial function

`run_trial_unified` takes a starting board, its pre-roll or post-move convention, the cube branches, a trial's dice sequence and the opening caches. On a pre-roll start the starting player moves on even half-moves. On a post-move start the board flips first, so the starting player moves on odd half-moves. During play the board, cube ownership and away scores refer to the player on roll.

1. **Resolve cube decisions.** Skip the opening half-move because its branch definitions already specify the opening cube action. From move 1, test eligible unfinished branches before throwing the dice. A pass settles only that branch. If all branches pass, obtain static pre-roll cubeless probabilities from the board where play stopped, apply the accumulated correction and finish.
2. **Generate legal boards.** Use `possible_boards_unsorted` to enumerate unique legal endpoints in first-seen order. Compute all 21 roll lists for an uncached VR mean, or the played roll where needed. For a ready opening reply, its entry supplies the game state and selection.
3. **Compute the VR baseline.** Obtain the 21 static selections and their probabilities, then form the cubeless and per-branch means. With VR disabled or thinned the trial proceeds directly to the checker choice.
4. **Play the sampled dice.** Apply the opening cache, shared reply, or checker strategy appropriate to the half-move. For a later base-level choice an available VR winner can supply the played index. The first unfinished branch drives the board when cube-aware selection is active. Evaluate the actual correction and add it to the starting player's accumulators.
5. **Settle or advance.** A terminal checker play produces the exact single, gammon or backgammon result. Settle unfinished branches at their respective cube values, then subtract the accumulated luck. Otherwise flip the board, reverse ownership and swap match away scores for the next half-move. Reaching the horizon invokes the truncation evaluator.

At trial entry the engine enables cube processing if at least one branch has a cube below its structural cap or an unlimited cap. Each subsequent cube decision checks which branches remain unfinished and can legally double. Match legality and the match-equity tables separately handle score-dependent dead cubes. [Trial phases](cpp/src/rollout.cpp) · [Legality and scoring](cpp/include/bgbot/cube.h)

## 6. Position rollouts

`rollout_position` takes a post-move board and estimates cubeless probabilities and equity from the perspective of the player who moved. The opponent rolls first. The trials carry no cube branches. An already terminal post-move board returns its exact probabilities and equity with zero standard errors.

`cubeful_rollout_position` also handles a terminal board directly. Otherwise it flips the post-move board and cube owner and swaps the away scores. It calls the two-branch cube decision from the opponent's perspective, selects the opponent's legal optimal action, then negates that optimal normalized equity for the player who moved. It inverts the five cubeless probabilities from those same trials. The win standard error is invariant under inversion and the gammon/backgammon standard errors exchange sides. The chosen ND or DT branch supplies the cubeful standard error. A pass has zero cubeful standard error.

Public `post_move_analytics` uses this native cubeful position path when `cubeful_trial_moves` is enabled. With the switch off it takes a cubeless rollout and applies the static cube conversion to its mean probabilities. The public checker-play evaluator uses the native cubeful position result when a cube context is supplied. [Native position results](cpp/src/rollout.cpp) · [Python analytics](python/bgsage/analyzer.py)

## 7. Cube decision rollout

`cubeful_cube_decision` starts with the player on roll. Both branches use the input cube value as their basis. **ND** retains the input cube and owner. **DT** starts at twice that cube with the opponent owning it. Both branches see the same dice and checker board. They can have different future cube decisions and finish on different half-moves. Their estimates are correlated. Cubeless probabilities and equity come from this same trial trajectory.

The public cube decision compares `ND` with `min(DT, DP)`. An offer requires a legal double and a strictly greater value for doubling, subject to the automatic post-Crawford double rule in section 15. A take is selected at `DT ≤ DP`. If the player cannot offer the cube, ND determines the action even though the native routine computes a hypothetical DT value.

In unlimited play DP is +1 in units of the starting cube. When beavers are enabled and the simulated DT value is negative for the doubler, the public comparison uses twice that value. At the public boundary ND and DT carry their branch standard errors. In match play the comparison uses MWC with the score-specific pass value, then converts the displayed actions and errors to normalized equity. [Cube-decision binding](cpp/pybind/bindings.cpp) · [Native branch simulation](cpp/src/rollout.cpp)

## 8. Checker selection during trials

At half-move 0 the opening table uses the configured early checker evaluator. Its reply table uses the base evaluator for half-move 1. At subsequent half-moves `ultra_late_threshold` selects the base 1P evaluator first, then `late_threshold` selects the late checker evaluator, otherwise the early evaluator applies. The threshold values are half-move indices, including the opening roll.

When `cubeful_trial_moves` is enabled and a cube is active, the trial supplies the ordered active cube states to the multi-cube selector, subject to `cubeful_late_threshold`. The move selected for ND drives the shared board while ND is active. If ND finishes first, the remaining branch drives it. A single board per trial is a shared-board approximation to independent optimal checker play in both branches. The precomputed opening and reply boards use the selection context supplied at prefill.

At 1P the built-in backgame-aware pair evaluator resolves the model from the pre-move board once per candidate list. It evaluates the first nonterminal candidate in full and reuses its hidden-layer sums for sibling input differences. A list reaching 64 changed features triggers a dense forward for that sibling. Blended backgame models evaluate their components. Terminal outcomes are exact. Static cube-aware ranking uses board-clamped probabilities, an equity tie tolerance of `1e-6` and cubeless equity as the tie-break. The selected endpoint's VR probabilities come from the active base evaluator, including its database wrapper.

Higher-ply checker play uses the multi-ply evaluator and an internal move filter of up to two moves within 0.03. Cubeless selection with a `MultiPlyStrategy` uses the cubeful tree with one dead cube. A configured inner rollout uses its own selection path. A hybrid strategy can use a separate filter evaluator with the main base evaluator at leaves and for VR. [Checker and cube-aware selection](cpp/src/rollout.cpp) · [Candidate evaluator](cpp/src/strategy.cpp) · [Multi-ply search details](MULTI-PLY.md)

## 9. Cube decisions during trials

The `cube` evaluator governs eligible offers and take/pass decisions throughout the game. Its default depth is `decision_ply`. Checker late thresholds affect checker play; `cube_late` is accepted in the configuration and the trial chooses `cube` at each half-move.

- **1P.** Use mover-perspective pre-roll probabilities, exact from the bearoff database when applicable, and the board's cube efficiency. Evaluate each legally available branch using the Janowski cube decision.
- **N-ply.** Compute one set of static screen probabilities for the board. For each eligible branch, a 1P double admits the branch to deeper evaluation. A 1P no-double also gets a screen at zero cube efficiency if both sides have at most two checkers on the board, including the bars, and at least one side has at most 24 pips. A double on this second screen admits the branch. Evaluate admitted states together with the configured N-ply cubeful search and deep prefilter. The move-1 cube-decision cache can supply this result.
- **Nested rollout.** For each legally eligible branch call the configured inner rollout's cube decision on its board and cube. The child uses one worker, its own trial count, checker depth and horizon, and VR.

On a take multiply the branch cube by two and transfer ownership to the taker. On a beaver multiply by four and give ownership to the beaverer. A pass awards the current cube to the doubler. In match play update the match-winning chance of that outcome using the away scores; in unlimited play normalize by the starting cube. Apply the accumulated branch luck on settlement. [Cube decision modes and screens](cpp/src/rollout.cpp)

## 10. Opening and shared move caches

| Cache | Contents and lifetime |
| --- | --- |
| `Move0Cache` | One selected post-move board for each of the 21 possible opening rolls, shared across the trials of the position. A forced pass stores the input board. |
| `Move1Cache` | One reply table per first roll for trials with a second half-move. Each contains race status, mover probabilities, cube efficiency, 21 selected reply boards, winner indices and probabilities, actual-play probabilities, and weighted cubeless means. Cube-aware prefill flips the supplied cube states into the reply mover's perspective. |
| Move-1 cube decisions | Cache the escalated N-ply branch decisions separately for each first roll, with an active-branch mask and a fingerprint of each decision cube. The engine rechecks the cube context before consuming those values. |
| Later VR selections | One 2,048-slot direct-mapped table per rollout position when VR is enabled, `n_trials ≥ 288` and `truncation_depth > 2`. The base evaluator and database are fixed for its lifetime. |

The VR table uses a board hash to locate a slot. A hit requires equality of the full board, ordered selection cubes and efficiency. Each cube comparison includes value, owner, both away scores, Crawford status, Jacoby, beaver, cap and efficiency override. A slot publishes one immutable set of 21 indices and five probabilities per index. Workers compute normally on a collision or an unpublished entry. A hit saves candidate selection and evaluation; the trial still generates its moves and computes weighted means in roll order.

Opening entries use atomic empty, computing and ready states. Readers acquire a published entry. An on-demand worker can claim an empty entry or wait for its owner. In a parallel single-batch cube decision the caller reserves the opening slots before dispatch. A worker computes a first roll and its dependent reply table, then can start trials once all prefill assignments have been claimed. Other workers can still be completing different rolls. The cubeless path and target-SE batched cube path complete prefill before their trial loop. [Cache structures](cpp/include/bgbot/rollout.h) · [Publication and reuse](cpp/src/rollout.cpp)

## 11. Truncation evaluation

After the last checker move, the trial board has already flipped to the next roller. A live cubeful trial at depth greater than 1P passes that pre-roll board and all unfinished cube states through one serial `cubeful_equity_nply_multi` traversal. The traversal returns branch values and cubeless probabilities together. Invert the latter to the last mover's perspective and clamp them against that player's post-move board. Unlimited branch equities scale by the branch cube over the starting cube. Match branch results convert to MWC at the branch cube and score before the starting-player perspective and luck correction are applied.

For cubeless truncation, use an exact post-move database lookup when the last mover's board is in range. Outside that range an N-ply horizon uses the pre-roll tree with a single dead cube and inverts its probabilities. A 1P horizon evaluates the last mover's board directly. For active branches at a 1P horizon, flip each branch's cube state into last-mover perspective and apply the appropriate Janowski conversion to these probabilities. When all branch cubes are structurally dead, the trial can use its corrected cubeless value scaled to the basis cube. [Horizon evaluation](cpp/src/rollout.cpp) · [Multi-ply tree](cpp/src/cube_eval.cpp)

## 12. Parallelization

Trial dispatch uses the persistent multi-ply worker pool. An explicit positive `n_threads` is limited to the number of trials. With automatic selection, use hardware concurrency with a minimum of one. For a native truncated N-ply configuration with automatic threads and `parallelize_trials = false`, use one worker. The public Python rollout analyzer enables trial parallelism. Searches and nested rollouts inside a trial use one worker.

Workers claim chunks of eight trials by atomic increment and write results by trial index. A single-thread cubeless rollout can accumulate directly. Parallel results reduce in index order. Dispatch clears the relevant thread-local hybrid and cubeful search caches at operation boundaries. A separate position cache shared by trial workers supports the hybrid cubeless recursion and all-dead-cube recursion under distinct keys; dispatch clears populated shared storage at its occupancy threshold. [Worker pool and search caches](MULTI-PLY.md) · [Rollout scheduling](cpp/src/rollout.cpp)

Progress callbacks report completed trials. The public checker interface combines trial progress across candidate moves and can report individual promotion progress. Parallel workers test the cancellation flag at chunk boundaries. Once the workers return, cancellation raises `RolloutCancelled`. The Python adapter translates the native exception; it also checks cancellation between candidate evaluations. [Python progress and cancellation](python/bgsage/analyzer.py) · [Native binding](cpp/pybind/bindings.cpp)

## 13. Statistical aggregation and target-SE batching

For a corrected per-trial observation `X`, accumulate the sum and sum of squares in double precision. For `N` trials the reported mean is `ΣX/N` and the standard error is `sqrt(max(0, ΣX²/N − mean²)/N)`. This estimates sampling dispersion under the chosen checker policy, cube policy and horizon.

Each probability, scalar cubeless equity and ND/DT branch has its own standard error. The reported cubeless equity can be obtained by applying the linear equity formula to the mean probabilities. Its error uses the per-trial equity observations, which preserve covariance among probability components. Match branch statistics accumulate in MWC; display conversion scales the standard error by the local slope of `mwc2eq`. DP is deterministic for a given cube and score.

For the public `cube_action` on a custom rollout, `target_se > 0` selects native target-SE batching. `n_trials` becomes the batch size. Prefill the opening caches once, then generate new stratified dice for each batch `b` with the unsigned 32-bit seed `seed + b × 0x9E3779B9`, starting at `b = 0`. Accumulate individual trial moments across all batches. After each complete batch test the pooled **ND** standard error in its native units (starting-cube units in unlimited play, MWC in match play). Stop at or below the target or on reaching `max_batches`. Native results expose `n_batches`, `total_trials` and `se_converged`. For fixed-budget runs the public binding selects the single-batch path. The native cubeful position call used by checker play and public post-move analytics uses one batch per invocation. [Aggregation and batch stopping](cpp/src/rollout.cpp) · [Public binding dispatch](cpp/pybind/bindings.cpp)

## 14. Bearoff database and neural evaluations

With a loaded one-sided bearoff database, the base evaluator used for rollout leaves and VR can return exact in-range probabilities. A cubeless static roll ranks terminal candidates by their result, in-range candidates by database equity and other candidates by NN equity. An all-bearoff cube-aware roll ranks exact post-move probabilities for each cube and keeps the first candidate at the largest exact cubeful value. At a mixed cube-aware boundary, the wrapper delegates the whole ranking to the base strategy. A selected winner in range still receives its exact probabilities when the VR evaluator scores that board. The public checker output also uses exact database probabilities for an in-range endpoint.

Probability clamping uses the board the evaluation describes. Once a side has borne off a checker, set its gammon and backgammon loss probabilities to zero. In a race, if that side has no checker on the bar or in its opponent's home board, set its backgammon loss probability to zero. Apply both rules symmetrically. Terminal positions have exact 0/1 outcome probabilities. Clamping a static or horizon input precedes cubeful conversion and VR subtraction. [Bearoff strategy](cpp/src/bearoff.cpp) · [Board clamping](cpp/include/bgbot/strategy.h) · [Neural evaluator](cpp/src/neural_net.cpp)

## 15. Match play

Match cubes carry both away scores and the Crawford flag. A perspective flip swaps away scores and player/opponent ownership. For a pre-roll value, reversing perspective maps MWC `p` to `1 − p`; differences in MWC reverse sign. Terminal and pass outcomes use the match equity table at the awarded points. Normalized public equity uses the win and loss anchors at the supplied cube and score.

Cube legality includes ownership, the cap and Crawford/score rules. When a legal offer is available in a non-Crawford game with the player more than one away and the opponent one away, the public decision applies the automatic double rule. Jacoby applies only to unlimited play with the rule enabled and the cube centered. A single win, gammon and backgammon otherwise pay one, two and three times the current cube. Beaver treatment is specific to unlimited play. [Cube rules](cpp/include/bgbot/cube.h) · [Match equity tables](cpp/src/match_equity.cpp)

## 16. Best move selection via rollout

Public checker analytics first scores every legal post-move board at 1P. With a positive `prefilter_threshold`, keep all candidates strictly within that equity distance of the leader. If several survive, rescore them at 2P. Apply the final equity window and move cap to the highest available screening level. The public default final filter keeps up to five moves strictly within 0.08. With the loose filter disabled, use the final window and cap on the original 1P scores.

The analyzer ensures two rollout candidates when its fallback pool permits and includes any legal `force_boards` supplied by the caller. The cubeful wrapper scores survivors with their native post-move rollout equity. Other moves keep their 2P or 1P screening results. If a shallow move becomes leader after the cubeful sort, roll it out, re-sort and repeat until the leader has a rollout result. The final order is descending equity, then descending cubeless equity, then lexicographic board order. Each move reports its evaluation level and `equity_diff = move_equity − best_equity`. [Public filtering, promotion and ranking](python/bgsage/analyzer.py)

The native strategy interface also supports selecting from an existing candidate batch. Cubeless `best_move_index` applies its configured filter to base equities and rolls out the survivors when more than one needs comparison. `best_move_index_cubeful_multi` ranks static candidates for each supplied cube and takes the union of its per-cube shortlists. If the union contains several survivors, it runs a cubeful position rollout for every survivor at every supplied cube and returns a winning index per cube. These operations serve strategy callers, including nested trials. [Native selection interfaces](cpp/src/rollout.cpp)

## 17. Configuration reference

| Native `RolloutConfig` field | Default | Meaning |
| --- | --- | --- |
| `n_trials`, `truncation_depth` | 36, 7 | Games per position or batch, and horizon in half-moves. Depth zero requests a full game. |
| `decision_ply`, `truncation_ply` | 1, −1 | Early checker depth and horizon depth. A horizon depth below one inherits `decision_ply`. |
| `late_ply`, `late_threshold`, `ultra_late_threshold` | −1, 20, 2 | Late checker depth (inherit `decision_ply` when unset), first late half-move and first base-only half-move. |
| `checker`, `checker_late` | unset | Per-purpose strategies. An unset late override uses an explicit `checker` or the effective late depth. |
| `cube`, `cube_late` | unset | An unset `cube` inherits `decision_ply`. In-trial take/pass uses `cube` throughout. |
| `cubeful_trial_moves`, `cubeful_late_threshold` | true, 0 | Select by branch cubeful equity up to the threshold. Zero inherits `ultra_late_threshold`. |
| `filter` | `TINY` (5, 0.08) | Outer native checker shortlist. Internal trial multi-ply strategies use the two-move, 0.03 filter. |
| `enable_vr`, `seed` | true, 42 | Enable corrections and initialize the stratified dice generator. |
| `n_threads`, `parallelize_trials` | 0, false | Native automatic worker selection and its truncated N-ply parallelism switch. |
| `target_se`, `max_batches` | 0, 50 | Fixed budget at zero target, or ND-SE target and batch cap for cube decisions. |
| `cancel_flag` | null | Optional shared atomic cancellation flag. |

`TrialEvalConfig` selects the checker or cube evaluator for a purpose. `ply` defaults to zero (inherit). Setting `ply ≥ 1` selects static or multi-ply search. Positive `rollout_trials` selects a child rollout, taking priority over `ply`. Child defaults are `rollout_depth = 5` and `rollout_ply = 1`, with one worker, VR and seed 42.

Python's `BgBotAnalyzer(eval_level="rollout")` defaults to 1,296 trials, depth zero and ultra-late threshold 9,999. Its `parallel_threads` argument maps to native `n_threads`. The default loose public checker prefilter is 0.15 for 2T and 3T. For full rollouts it is `max(0.15, filter_threshold + 0.05)`; for other custom truncated configurations and 1T it is disabled by default. The public constructor exposes the custom rollout and the fixed presets separately. [Native configuration and result types](cpp/include/bgbot/rollout.h) · [Public constructor](python/bgsage/analyzer.py)
