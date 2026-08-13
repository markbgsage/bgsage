# The Open Sage Backgammon Bot Engine Library

Open Sage is a "backgammon bot engine" - that is, a library that you can ask questions like "what's the best
move in this situation, with all the analytics to back it up", or "is this a double, and if so, is it a take or a pass 
(including analytics)?".

## What it Contains

It is a joint Python/C++ library that includes:
* A neural network-based backgammon bot.
* Neural network training framework using both self play and supervised learning, including training code, benchmark scoring, with customizable inputs. Uses your NVIDIA GPU (via CUDA) if you have one.
* The post-training weights for several different versions of the bot engine.
* Multi-ply and rollout calculations that efficiently parallelize on the CPU. Full technical specifications of the algorithms are in MULTI-PLY.md (N-ply search and the cubeful evaluation engine) and ROLLOUT.md (truncated and full rollouts with variance reduction).
* Test framework.
* VIBING.md: information on how to use Claude Code-style tools to interact with and change it, and how to submit changes back to us (the maintainers). Most of this code was written by Anthropic's Claude Code and OpenAI's Codex.

## What Evaluation Levels Does It Support?

The library supports "multi-ply" lookahead calculations. 1-ply is the raw neural network evaluation (we follow the XG/eXtreme Gammon
numbering convention; GNUbg calls this "0-ply"). Adding a ply makes the calculation roughly 20x slower. It efficiently parallelizes
these multi-ply calculations on your CPUs.

It supports truncated rollout calculations, where it simulates the game several turns into the future and then stops the simulation, using bot evaluations at the leaves.

It also supports full rollout calculations, which are simulations playing out the game over and over to completion. 

Truncated and full rollouts both include variance reduction and efficiently parallelize on CPUs.

## What are Its Interfaces?

It offers both Python and C++ interfaces for:
* Checker play analytics: given a list of checker positions, the two dice, and cube information, it returns you a list of information about the top possible moves, sorted in descending order of equity; for each it gives you equity and cubeless post-move probabilities. You can specify the evaluation level.
* Post-move position analytics: given a list of checker positions and the cube information, it returns you cubeful equity, cubeless equity, and the cubeless probabilities - for a post-move position (right before the opponent's turn).
* Cube action analytics: given a list of checker positions and the cube information, it returns you cubeful equity information about the three states (ND, D/T, and D/P), cubeless equity, and the cubeless probabilities - for a pre-roll position.
* Game plan classification: given a list of checkers, it returns the optimal game plans of the player and the opponent.
* Game utilities (flip a board, etc).

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
