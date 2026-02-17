# Working with AI Agent Coding Tools

## MY EXPERIENCE (February 2026)

I used Claude Code, mostly with Opus 4.6, to write almost all the code in this project. It was a startling experience, and the first time that I built something really without much looking at the code. I gave it example code from an earlier personal bot project (that did not work very well!), a link to my old blog from when I tried to learn about neural networks by building a bot (compgammon.blotspot.com), and asked it to build an optimzed C++-based backgammon bot with a Python interface.

I gave it clear benchmarks for training the cubeless bot: the GNUBG Contact and Race benchmark scores, which I knew from my earlier project were well correlated with playing performance. And I gave it specific examples where I knew what the bot outputs should be, so it could regression test against those. Just four or five, though.

It definitely did not one-shot this. But it did put together a credible plan - which we evolved over time - and we incrementally built out the pieces. Big pieces at a time - for example, implementing a hand-rolled neural network class in C++ because it convinced me that this would be more efficient than using an existing library like PyTorch or Tensorflow, since the use case is so simple; or implementing a parallelized version of the multi-ply evaluation. 

I rarely looked at the actual code during this process - I just told it what specific types of interface it should create, and how to test those interfaces.

After Claude Code put together a first version, I asked it to profile and optimize its code, and it did significantly improve its performance. Then I pointed OpenAI's Codex at the same code, using GPT 5.3-Codex-Spark, and asked it to also profile and optimize, and that second round also resulted in significant performance improvements.

This all took about a week of work. I was mostly idle during this process, except for watching the chain of thought of the AI and passing it notes when I saw it starting to go astray. I would occasionally give it feedback about direction. I often asked it to create and run one-off tests for me, as well as to run the Contact and Race benchmarks, and create and run against a separate set of PR-like benchmarks. This process was like working with an amazingly quick and well-read junior developer in tight collaboration, and less like "programming in English".

## How to Use Claude Code Yourself

I am most familiar with how Claude Code works, as a desktop application and not CLI, so that is what I will describe here. I think the app is probably easier for new people to use who don't consider themselves strong developers, and that is who I'm aiming this at.

First, download the Claude desktop application to your computer. This application lets Claude access your file system and run commands as you, with various limits. It can read all the files inside a directory that you point it at.

Next, download git to your computer, if you have not already. Ask an AI how to do it if you're not sure. git is a tool to manage different versions of code, so you can always walk back and attribute changes. We'll use it to download the code.

Next, check out the markbgsage/bgsage git repository from github.com. If you do not know how to do this (like I do not), create a new folder (called bgsage) wherever you like on your computer, then run Claude Code and connect to that folder. Choose the Code tab at the top, then ask Claude Code to check out that repository, just like "check out the repository markbgsage/bgsage from github.com". It'll walk you through how to do it.

Now you've got all the files you need on your computer. There's a file called CLAUDE.md that has the context that Claude Code uses each time it starts a session - that acts like a memory for the project. It also has all the source code in various subdirectories.

How do you do stuff with it? You just ask at the chat prompt (in the Code tab of the Claude app), in regular English. Like "run a contact benchmark on the best cubeless strategy so far", and it'll generate the code to run that benchmark calculation, then run it. Or, "I want to try out a different algorithm for game plan classification [describe it], then train a new strategy that has one neural network per each category in my new classification, and see if it performs better than the current best strategy." Or, "I've found a new best strategy - change the library so that evaluations use it."

You can also write your own code that uses the library as a component - ask Claude Code what the specific interfaces are for things like checker play and cube action analytics. Then ask it to write whatever other code you want that ends up calling those interfaces. 

This library is released under an open source license, the AGPL (see the LICENSE file for details). You can use it for commercial use, but if you use a modified version, you need to make the source code for it available under the same license terms. If you're not worried about commercial use, and you're just using it youself, generally don't worry: make whatever changes you like!

## Submitting Changes

Let's say you play around with this and train a better model. You can change the library to use it, then ask Claude Code to submit a PR (a "pull request") that has the changes in it. Claude Code will wrap that up for you and send it to us, the maintainers, and we'll take a look. If we like it, we'll merge it in with the main branch of the library that everyone sees.
