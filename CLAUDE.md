# CLAUDE.md

@AGENTS.md

All project rules live in AGENTS.md. It is the single source of truth. Edit rules there, not here. The notes below are specific to Claude Code.

## Working style in this repo

- For any change touching more than one file, or anything in `core/`, present a short plan first: which files, where each piece lives and why (core vs catwatch), and how you'll verify it. Then implement.
- Before editing `core/`, re-check the Pi-side import constraint and the invariants list in AGENTS.md.
- End every coding task by running `make format`, `make verify`, and make typecheck`. Report the actual results, including failures. Don't summarize a run you didn't do.
- When reporting results, separate what you verified (commands run, files inspected) from what you're inferring. Say "I haven't verified X" plainly.
- Correct your own earlier mistakes as soon as you notice them. Name the error and the fix.

## Things Claude Code should not do here

- Don't open, read, or `cat` `.env`. Use `.env.example` to learn which variables exist.
- Don't read large numbers of images from `data/raw/` into context. If you need
  to inspect data, use `captures.csv`, `labels.csv`, or targeted scripts that print counts.
- Don't launch the labeler (`python -m catwatch.label`) or `--show` modes. They need an interactive GUI and a human.
- Don't start long training runs without asking. Say roughly how long you expect them to take (CPU vs GPU) and what you'll compare against.
- Don't run anything on the Raspberry Pi. Give the user the exact commands to run there instead.