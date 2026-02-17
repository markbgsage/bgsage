# The Backgammon Sage Backgammon Bot Engine Library

Backgammon Sage is a "backgammon bot engine" - that is, a library that you can ask questions like "what's the best
move in this situation, with all the analytics to back it up", or "is this a double, and if so, is it a take or a pass 
(including analytics)?".

## What it Contains

It is a joint Python/C++ library that includes:
* A neural network-based backgammon bot.
* Neural network training framework using both self play and supervised learning, including training code, benchmark scoring, with customizable inputs. Uses your NVIDIA GPU (via CUDA) if you have one.
* The post-training weights for several different versions of the bot engine.
* Multi-ply and rollout calculations that efficiently parallelize on the CPU.
* Test framework.
* Instructions on how to use Claude Code-style tools to interact with and change it. Most of this code was written by Claude Code (Opus 4.6) and OpenAI Codex (GPT 5.3-Codex-Spark).
* How to submit changes back to us (the maintainers)

## What Evaluation Levels Does It Support?

The library supports "multi-ply" lookahead calculations. 0-ply is the raw neural network evaluation, which is equivalent to 1-ply in
eXtreme Gammon (just a difference in numbering convention - we follow gnubg's convention). Adding a ply makes the calculation roughly
20x slower. It efficiently parallelizes these multi-ply calculations on your CPUs.

It also supports rollout calculations, which are Monte Carlo simulations playing out the game over and over. It includes
variance reduction and efficiently parallelizes on your CPUs.

## What are Its Interfaces?

It offers interfaces for:
* Checker play analytics: given a list of checker positions, the two dice, and cube information, it returns you a list of information about the top possible moves, sorted in descending order of equity; for each it gives you equity and cubeless post-move probabilities. You can specify the evaluation level.
* Cube action analytics: given a list of checker positions and the cube information, it returns you cubeful equity information about the three states (ND, D/T, and D/P) and the pre-roll cubeless probabilities.
* Game plan classification: given a list of checkers, it returns the optimal game plans of the player and the opponent.
* Game utilities (flip a board, etc).
