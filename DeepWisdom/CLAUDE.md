# CLAUDE.md

Guidance for Claude Code when working in **DeepWisdom** — an Obsidian vault that serves as a personal knowledge repository for deep learning (neural networks, architectures, training, math foundations, papers, and applied notes).

This is a **knowledge base, not a codebase**. The "source" is interlinked Markdown notes. Optimize every action for a future reader navigating the Obsidian graph, not for a compiler.

## What this repo is

- An **Obsidian vault** (`.obsidian/` holds config — do not edit it unless asked).
- Plain Markdown notes using **Obsidian Flavored Markdown**: `[[wikilinks]]`, `#tags`, YAML frontmatter ("properties"), callouts, and `$LaTeX$` math.
- Core plugins in use: daily notes, templates, properties, tags, backlinks, outgoing links, canvas, graph, **bases**. No community plugins — keep notes portable and avoid syntax that depends on plugins.

## Working principles

- **Prefer linking over duplicating.** When a concept already has (or should have) its own note, link to it with `[[Concept Name]]` instead of re-explaining it. A dense graph of small, well-linked notes is the goal.
- **One idea per note.** Atomic notes (a single concept, technique, or paper) link better than long monoliths. Split when a note starts covering multiple distinct ideas.
- **Link liberally, even to notes that don't exist yet.** An unresolved `[[wikilink]]` is a valid signal that a note is worth creating — it's not an error.
- **Don't invent technical content.** This vault is about correctness in a fast-moving field. If you're unsure about a fact, formula, hyperparameter, or paper claim, say so or verify it — never confabulate equations, citations, or results.
- **Match the surrounding style.** Mirror the structure, tone, depth, and frontmatter of nearby existing notes before introducing a new pattern.

## Note conventions

### Frontmatter (properties)
Start substantive notes with YAML frontmatter. Suggested baseline:

```yaml
---
title: Layer Normalization
type: concept        # concept | architecture | technique | paper | moc | daily | reference
tags: [normalization, training]
created: 2026-06-29
status: seedling     # seedling | budding | evergreen
aliases: [LayerNorm, LN]
---
```

For **paper notes**, also include `authors`, `year`, `venue`, `arxiv`/`url`, and `pdf` if available.

### Filenames & titles
- Use readable, human-friendly note titles (e.g. `Attention Is All You Need.md`, `Backpropagation.md`) so wikilinks read naturally.
- Keep the `title` property in sync with the filename.
- Use `aliases` for abbreviations and alternate spellings so `[[LayerNorm]]` resolves.

### Tags
- Lowercase, hyphenated, topical: `#architecture`, `#optimization`, `#regularization`, `#transformers`, `#cnn`, `#rnn`, `#paper`.
- Tag for retrieval, not exhaustively. A handful of meaningful tags beats a dozen noisy ones.

### Links
- `[[Note]]` to link; `[[Note|display text]]` to alias the display.
- `[[Note#Heading]]` to link to a section; `[[Note#^block-id]]` for a specific block.
- When you add a claim drawn from another note, link the source.
- **Never put math (`$...$`) in a link's display title.** The display text of a `[[Note|display text]]` (and of a bare `[[$x$]]`) must be plain prose — math doesn't render inside Obsidian link titles and reads as raw LaTeX. Keep the equation in the surrounding sentence and give the link a worded title instead.
  - Bad: `the [[Diffusion Models Root#Training objective|$\mathcal{L}_{\text{simple}}$ objective]]`
  - Good: `the [[Diffusion Models Root#Training objective|simplified training objective]] $\mathcal{L}_{\text{simple}}$`
  - This applies only to the display title — math is fine in a `#Heading` anchor if the target heading genuinely contains it.

### Math
- Inline: `$\sigma(x) = \frac{1}{1 + e^{-x}}$`
- Block:
  ```
  $$
  \text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
  $$
  ```
- Prefer standard notation; define non-obvious symbols the first time they appear in a note.

### Callouts
Use Obsidian callouts for emphasis sparingly:
```
> [!note] Intuition
> ...
> [!warning] Common pitfall
> ...
```

### Code
Fence with a language tag (` ```python `). Keep snippets minimal and runnable-in-spirit; this is a notes vault, not a project — don't add build tooling or dependencies.

## Structure & navigation

- **Maps of Content (MOCs):** Hub notes (e.g. `Deep Learning MOC`, `Optimization MOC`) that link out to related notes are the preferred way to organize, over deep folder hierarchies. Let the graph do the work.
- Folders are fine for coarse buckets (e.g. `Papers/`, `Concepts/`, `Daily/`), but **don't reorganize existing folders or move notes without being asked** — moving notes can break links and reshape the graph.
- Daily notes go through the daily-notes plugin's location; follow whatever pattern already exists.

## When asked to add or edit notes

1. Check for an existing note on the topic first (search by title and aliases) — **update it rather than create a duplicate.**
2. Place new notes consistently with where similar notes already live.
3. Add frontmatter, a clear `# H1` title, and at least one outbound `[[link]]` to connect it to the graph.
4. After creating a note, consider whether an existing MOC or related note should link *to* it, and add that backlink if it's natural.
5. Report unresolved links you intentionally left as future stubs.

## Don'ts

- Don't edit anything in `.obsidian/` unless explicitly asked.
- Don't bulk-rename, move, or delete notes without explicit confirmation — it's hard to reverse and breaks links.
- Don't add community-plugin-specific syntax (Dataview, Templater, etc.) — none are installed.
- Don't fabricate papers, equations, benchmarks, or citations.
- Don't delete `Welcome.md` unless asked (it's Obsidian's default starter note).
