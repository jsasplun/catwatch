# AGENTS.md

Instructions for AI coding agents working in this repository. Read this whole file before making changes.

## What this project is

A Raspberry Pi camera looks down at a cat water bowl. A small image classifier (MobileNetV3-Small, fine-tuned with PyTorch) decides each frame whether the bowl region shows: `empty`, `black_patch_cat`, `orange_patch_cat`, or `both_cats`. Per-frame predictions are smoothed into visit events and logged to `data/events.csv`.

Two environments, with different constraints:

| Where | Runs | Has PyTorch? | How it's set up |
|---|---|---|---|
| Desktop (Windows + WSL + Docker dev container) | labeling, training, evaluation, export, tests | Yes | `Dockerfile`, `.devcontainer/`, `requirements.txt` |
| Raspberry Pi (Raspberry Pi OS Bookworm, native venv, no Docker) | `catwatch.collect`, `catwatch.check_crop`, `catwatch.monitor` | **No** | apt packages + `requirements-pi.txt` |

The pipeline:

1. collect raw images on the Pi
2. label on the desktop
3. version the data with DVC and the labels with git
4. train
5. evaluate
6. export to ONNX, with a parity check
7. copy `models/<run>/` to the Pi
8. run the monitor

## Repository layout and the core rule

- `core/`: Reusable, domain-agnostic building blocks (camera, motion, capture storage, labels, labeler, splits, preprocessing, training, evaluation, ONNX export/inference, event smoothing, run records).
- `catwatch/`: This project's scripts and glue (settings, data joining, entry points run with `python -m catwatch.<name>`).
- `tests/`: Pytest tests.
- `config.yaml`: Every project-specific name and number.
- `data/`: Location for all data collected or labeled.

**Hard rule: `core/` must never import `catwatch/`, and nothing in `core/` may mention cats, bowls, or this project's classes.** `tests/test_core_independence.py` enforces the import half. You enforce the naming half.

When adding code, decide placement first:

- Would it make sense in a different computer-vision project? Then it goes in `core/`.
- Does it know about this project's classes, paths, or config? Then it goes in `catwatch/`.

Project-specific values go in `config.yaml`, never hard-coded.

## Pi-side import constraint

`catwatch.monitor`, `catwatch.collect`, and `catwatch.check_crop` run on the Pi, where PyTorch, torchvision, scikit-learn, and onnx are **not installed**. Everything they import, directly or indirectly, must work with only these:

- the standard library
- NumPy and OpenCV (from apt)
- picamera2 (from apt)
- onnxruntime, PyYAML, python-dotenv

Currently Pi-safe modules: `core.camera`, `core.motion`, `core.csv_log`, `core.capture_store`, `core.labels`, `core.splits`, `core.preprocessing`, `core.onnx_classifier`, `core.events`, `core.run_records`, `catwatch.settings`, `catwatch.hardware`, `catwatch.data`.

Do not add torch or sklearn imports to any of them. If a Pi-side script needs new functionality, put it in a module that stays Pi-safe.

`picamera2` and `libcamera` are imported lazily inside `Picamera2Source`, so `core/camera.py` still loads on the desktop. Keep it that way.

## Invariants that protect the data and the metrics

Breaking any of these silently corrupts results. Don't change them without the user's explicit OK, and explain the consequence when you ask.

1. **Raw data is immutable.** Never modify, rename, move, or delete anything under `data/raw/`, including `captures.csv`. Images are referenced by their path relative to `data/raw/`.
2. **`data/labels.csv` is append-only.** The newest row for an image wins. Never rewrite, sort, deduplicate, or hand-edit it. Corrections are new rows (written through `core.labels.append_label`).
3. **Class order in `config.yaml` → `classes` defines model output indices.** Never reorder or rename existing classes. New classes go at the end and require retraining.
4. **Split assignment is deterministic.** `core.splits.assign_split` (SHA-256 of the group id) and `catwatch.data.group_for` (one group per clock hour) decide which images are train/val/test. Changing either one moves images between splits. That invalidates every earlier comparison and can leak test images into training.
5. **Preprocessing parity.** Evaluation and deployed inference both go through `core.preprocessing.to_model_input`. Changing it requires retraining, re-export, and re-running the export parity check. Training augmentation must never shift hue: the two classes differ only by patch color.
6. **The model and its crop travel together.** `catwatch.monitor` reads the crop and class names from `models/<run>/model_card.json`, not from today's `config.yaml`. Keep it that way.
7. **The test split is used once**, for the final chosen model. Tune and compare on `val`. Never add code that selects models or hyperparameters using `test`.
8. **Never use model predictions as labels.** Event snapshots saved by the monitor must be labeled fresh by a human.

## Commands

Run these inside the dev container:

| Command | What it does |
|---|---|
| `make lint` | `ruff check core catwatch tests` |
| `make typecheck` | `pyright` |
| `make format` | `black`, then `ruff check --fix` |
| `make test` | `pytest` |
| `make verify` | clean import → ruff → black --check → pytest (the full gate) |

- Ruff: line length 88, target py311, rule families `E4, E9, F, I` only. Other families are staged for later; don't enable them unasked.
- The type checker is **pyright**, not mypy.
- Naming: snake_case functions and variables, PascalCase classes, UPPER_CASE constants (PEP 8).

**Definition of done for any code change:**

1. `make format` succeeds.
2. `make verify` passes.
3. `make typecheck` shows no new errors.
4. The behavior is verified against a real artifact (see Verification below).

## Code quality checklist

Apply per function or class touched, in order:

1. **Placement**: does it live where its domain responsibility says (see the core rule), not just where it historically landed?
2. **OOP-fit**: is there a real class or interface here, or is a set of plain functions fine?
3. **Library over hand-rolled**: does the standard library or an existing dependency already do this correctly? Don't add new dependencies without asking, and never add heavy ones to the Pi side.
4. **DRY**: only consolidate copies that are numerically and behaviorally identical. A near-duplicate with a different constant is not a DRY win.
5. **KISS**: the simplest design for the current problem.
6. **YAGNI**: no abstractions for hypothetical future needs.
7. **Law of Demeter**: apply at public or class API boundaries, not as blanket dot-counting inside a cohesive object graph.

North star: clean, easy to understand, reusable code. When rules conflict, choose readability over rule purity.

## Comments and clarity

- Write for a reader with zero context: a fresh college student with no domain knowledge and no history with this code.
- Explain *what* and *why* wherever the name alone doesn't. Unusual names and non-obvious mechanisms (hash splits, majority-vote smoothing, softmax stability, BGR ordering) get real explanations, not fragments.
- No backstory, refactor history, ticket numbers, or "changed X to Y" comments. That belongs in git.
- Line count is not the metric. Clarity is.

## Verification

- **Never trust results without checking a real artifact.** "It should work," log lines, and self-reports don't count. Check output files (`metrics_*.json`, `errors_*.csv`, `model_card.json` parity numbers, saved JPEGs), test runs, and actual command output.
- **Fix the root cause, not the symptom.** If you're filtering bad output, you haven't found the bug.
- **Read the docs fully before coding against a library**, especially Picamera2, torchvision transforms v2, `torch.onnx.export`, and onnxruntime. Most "hard" bugs are a skipped one-liner.
- **Performance or quality claims need a paired comparison.** Run the baseline and the change with the same seeds (at least 5, e.g. `for s in 0 1 2 3 4; do python -m catwatch.train --seed $s; done`) and evaluate on `val`. Compute the per-seed differences in macro-F1. Claim a win only if the mean difference exceeds 2 × (std of differences / √n). A single winning run is noise until shown otherwise.
- If the user says they're urgent, skip clarifying questions: fix it, verify it, and report.

## Safety habits

- **Worktrees for risky, exploratory, or parallel work**, e.g.
  `git worktree add ../cbm-exp-<topic> -b exp/<topic>`. New worktrees need `dvc checkout` to get images.
- **Zip a backup before any real cleanup or decomposition pass**, to the Desktop or outside the repo. Ask the user to confirm the backup exists before starting.
- **Safe-delete means move to `delete_me/`** (gitignored). Never `rm` project files. Nothing is truly deleted until the user gives explicit final OK.
- Never run without explicit user approval:
  - `git push --force`
  - history rewrites
  - `git clean`
  - `dvc gc`
  - `dvc push` / `dvc pull` against a remote
  - anything that touches the Pi over SSH
  - `sudo` commands
- Don't commit unless asked. When asked, use clear messages describing *what* changed. For label changes, include the count (e.g. "Labels: +240 images").

## Secrets

Secrets live only in `.env` (git-ignored). The container loads it with
`--env-file`, and `catwatch.settings.load_config()` loads it with
`python-dotenv`.

- Never print, log, echo, or commit `.env` contents. Never copy secrets into `config.yaml`, code, tests, or run records.
- New secret names go into `.env.example` with an empty value and a comment saying what they're for.
- `config.yaml` is saved verbatim into every `runs/*/run_info.json`, so it must never contain secrets.

## When unsure

- Don't guess hardware facts (Pi model, camera model, pin or connector details). Ask, or state the assumption explicitly in your response.
- If a requested change would break an invariant above, say so plainly, explain the mechanism, and propose an alternative before doing anything.