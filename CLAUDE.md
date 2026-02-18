# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational documentation site built with **MkDocs Material** covering Mathematics, Machine Learning, Generative AI, System Design, and Algorithms. Hosted on GitHub Pages at https://nvgr-genai.github.io/my-learning/.

## Build & Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Local development server (localhost:8000, hot-reload)
mkdocs serve

# Production build (outputs to site/)
mkdocs build --clean
```

Deployment is automatic via GitHub Actions on push to `main` (`.github/workflows/deploy.yml`).

## Architecture

### Content Structure

All documentation lives in `docs/` with five major sections:

- `docs/math/` — Mathematics (foundations, linear algebra, calculus, probability, optimization, information theory, quantum computing)
- `docs/ml/` — Machine Learning (fundamentals, supervised/unsupervised/reinforcement/semi-supervised/deep learning, feature engineering, model evaluation, MLOps)
- `docs/genai/` — Generative AI (LLMs, transformers, prompt engineering, RAG, agents, fine-tuning, multimodal, providers, ethical AI)
- `docs/system-design/` — System Design (fundamentals, architecture, networking, data, distributed systems, scalability, security, observability, 92 practice problems)
- `docs/algorithms/` — Algorithms & Data Structures (arrays, trees, graphs, DP, sorting, searching, strings, greedy, backtracking)

### Navigation

`mkdocs.yml` contains the full site navigation (~1000 lines). The `nav:` section must stay in sync with files in `docs/`. When adding new content files, always add a corresponding nav entry.

### Key Configuration

- **MathJax** enabled for LaTeX rendering (`docs/javascripts/mathjax.js`)
- **Mermaid** diagrams supported via superfences
- **Markdown extensions**: admonitions, tabs, code annotations, task lists, footnotes
- **`navigation.indexes`** feature means `index.md` files serve as section landing pages

## Content Standards

This is a **theory-focused learning repository**, NOT an implementation guide.

**Target balance**: 60% theory/concepts, 30% real-world examples, 10% code

**Do**:
- Use ASCII diagrams showing data flow and architecture
- Include real company examples with scale (users, QPS, data size)
- Add comparison tables and decision trees
- Keep code snippets to 5-10 lines illustrating concepts
- Target 600-800 lines per file

**Don't**:
- Write full class implementations (50+ line code blocks)
- Include production-ready code with error handling
- Repeat the same concept across multiple sections
- Add multiple implementations of the same concept

**Section structure for each topic**:
1. Overview with visual diagram
2. How it works (step-by-step)
3. Trade-offs with concrete scenarios
4. Real-world examples (companies, scale, why chosen)
5. Comparison to alternatives

## Important Rules

- **Never delete content files.** If a file exists but isn't in navigation, add it to `mkdocs.yml` nav — don't remove the file.
- **`_bmad/`**, **`.claude/`**, **`.specify/`**, **`.github/agents/`** are gitignored tooling directories. Don't commit changes in these paths.
- Commit messages use Conventional Commits style.
