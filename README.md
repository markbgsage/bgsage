# The Open Sage Backgammon Bot Engine Library

Open Sage is a "backgammon bot engine" - that is, a library that you can ask questions like "what's the best
move in this situation, with all the analytics to back it up", or "is this a double, and if so, is it a take or a pass 
(including analytics)?".

## What it Contains

Complete, seeded game and match simulation is available through
`from bgsage import simulate_game, simulate_match, GameConfig`. It records
positions, dice, cube actions and scores for custom research. See
[SIMULATION.md](SIMULATION.md) for examples, policy customization and rules.

It is a joint Python/C++ library that includes:

* A neural network-based backgammon bot.
* Neural network training framework using both self play and supervised learning, including training code, benchmark scoring, with customizable inputs. Uses your NVIDIA GPU (via CUDA) if you have one.
* The post-training weights for several different versions of the bot engine.
* CPU-parallel search and simulation. See the [multi-ply search specification](MULTI-PLY.md) and the [rollout specification](ROLLOUT.md).
* Test framework.
* VIBING.md: information on how to use Claude Code-style tools to interact with and change it, and how to submit changes back to us (the maintainers). Most of this code was written by Anthropic's Claude Code and OpenAI's Codex.

## What Evaluation Levels Does It Support?

The engine follows the XG/eXtreme Gammon ply convention. A 1-ply evaluation uses the neural network, or the exact bearoff database when applicable. Each additional ply looks ahead one turn and averages over the 21 distinct dice rolls with their probabilities. The CPU cost depends on the position, the requested analytics and the number of threads.

It supports truncated rollout calculations, where it simulates the game several turns into the future and then stops the simulation, using bot evaluations at the leaves.

It also supports full rollout calculations, which are simulations playing out the game over and over to completion. 

Truncated and full rollouts both include variance reduction and efficiently parallelize on CPUs.

## What are Its Interfaces?

The engine exposes Python and C++ interfaces. In Python, import `BgBotAnalyzer` from `bgsage`, create an analyzer with `eval_level="3ply"` and reuse it across positions to keep the model loaded. Pass `parallel_threads` to set the worker count. Each analytics call takes one 26-element board and the cube and match context relative to the player whose perspective the board represents.

* `checker_play(board, die1, die2, ...)` ranks legal moves by equity and gives post-move probabilities and the evaluation level for each move.
* `post_move_analytics(board, ...)` returns cubeless probabilities, cubeless equity and cubeful equity for the player who just moved, before the opponent rolls.
* `cube_action(board, ...)` returns the optimal cube action, no-double, double/take and double/pass equities, and cubeless probabilities and equity for the player on roll.
* Game plan classification identifies both players' game plans.
* Game utilities (flip a board, etc).

## Benchmarking 3-ply analytics

The timing runner measures 100 checker calls, 100 cube calls and 100 post-move calls. Each 100-position set is split evenly between unlimited play and five-point matches, with 20 positions per game plan. The cube and post-move calls use the same physical positions from opposite players' perspectives. The runner saves every returned field and reports timings by API, game plan and play format. [Positions and commands](data/multi_ply_benchmark/README.md)

On the benchmarked 16-thread Ryzen 7 9700X under Windows, the original and optimized builds took 10.92 s and 3.96 s respectively for the 300 calls. Every reported value and decision matched in the paired Windows builds. The report also compares both versions with full-game rollout references and gives separate Linux CPU measurements. [Timings and accuracy](data/multi_ply_benchmark/experiments/README.md)

The [1T, 2T and 3T timing study](data/multi_ply_benchmark/experiments/truncated_16t/README.md)
compares all three public truncated-rollout presets with the same fixed
positions and full-game references. It includes the full analytics for
every timed call, per-position wall times and rollout standard errors.

## How Does it Compare to XG?

We tested money games and match play against eXtreme Gammon (XG), another popular backgammon analysis application. In particular, we tested Open Sage's 3T evaluations (truncated rollouts) to XG Roller ++ (an equivalent truncated rollout). The two bots were very close, with some weak evidence that Sage 3T is a bit stronger.

Details of the tests, and instructions on how to replicate them, are in XG_COMPARISON.md

## License

Open Sage is released under the **Mozilla Public License 2.0** — see [LICENSE](LICENSE).

That covers everything in this repository, including the **trained model weights** in
`models/` and the bearoff database in `data/`. Those aren't source code, so to be
unambiguous: they are licensed on exactly the same terms as the code, and you are free to
use them in a commercial product.

In plain English:

* **You can use Open Sage in a commercial, closed-source product.** Import it, link it,
  bundle the weights, ship it. Your own code stays yours, under whatever license you choose.
* **If you modify a file that came from Open Sage and distribute it, publish that file's
  source** under the MPL, so the improvement comes back to everyone. Files *you* write remain
  entirely yours, even when they sit right alongside ours.
* **Private use carries no obligations at all.** Experiment, retrain, tinker — if you aren't
  distributing it, nothing is asked of you.

The MPL is file-level copyleft, so the obligation is narrow and predictable: it attaches only
to the files you changed, and never to the code you build around them. If you do improve the
engine or train better weights, a pull request is very welcome.
