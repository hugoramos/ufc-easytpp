# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Hugo Ramos Soares' MSc dissertation repository at Universidade Federal do Ceara (UFC). The research proposes **HoTHP** (Hyperbolic Rotary Position Embedding-based Transformer Hawkes Process), which replaces the trigonometric rotary kernel of RoTHP with hyperbolic functions (inspired by HoPE) to improve length extrapolation in Temporal Point Processes.

There are two LaTeX documents being written in parallel:
- **BRACIS 2026 paper** (`overleaf-bracis/`) — 15-page LNCS format, deadline 2026-05-04
- **MSc dissertation** (`overleaf-dissertacao/`) — UFC ufctex template (abnTeX2-based)

## Repository Structure

- `overleaf-bracis/` — BRACIS paper (Springer LNCS class `llncs.cls`, main file: `samplepaper.tex`, bibliography: `refs.bib`)
- `overleaf-bracis/figures/` — Python scripts (`gen_*.py`) that generate PDF figures; run with `python3.9`
- `overleaf-dissertacao/` — Dissertation (main file: `documento.tex`, chapters in `2-textuais/`, bibliography in `3-pos-textuais/`)
- `notebooks/` — Jupyter notebooks for experiments and exploratory analysis
- `papers/` — Reference PDFs (Hawkes 1971, THP, RoTHP, HoPE, ALiBi, etc.)
- `data/` — Datasets (e.g., `earthquakes.csv`)
- `figures/` — Standalone illustration images
- `legacy/` — Old survival-analysis work, no longer active
- `easy_tpp/` — **Symlink** → `/Users/hugoramossoares/Sites/ufc-easytpp/easy_tpp` (model implementations: THP, RoTHP, HoTHP, etc.)

## Build Commands

### BRACIS paper figures
```bash
python3.9 overleaf-bracis/figures/gen_attention_profile.py
python3.9 overleaf-bracis/figures/gen_main_result.py
python3.9 overleaf-bracis/figures/gen_kernel_alignment.py
```

### Dissertation (local compilation)
```bash
cd overleaf-dissertacao && make compile   # runs pdflatex + bibtex + makeglossaries + makeindex
cd overleaf-dissertacao && make clean     # removes auxiliary files
```

Note: Both LaTeX projects are synced with Overleaf. Local compilation is optional; Overleaf handles the build in most workflows.

## Writing Guidelines

- The paper is drafted in **Portuguese** first, then translated to English. Figures and their labels are always in **English**.
- **Never discard or rewrite Hugo's existing text.** Preserve his voice and phrasing; only add new content around what already exists.
- Writing style should be **explanatory and didactic**, not overly formal or fancy — it should read like a student's work (per advisor guidance).
- Commit messages for paper writing follow the pattern: `paper: <action> <section>` (e.g., `paper: write Background section (2.1-2.5)`).
