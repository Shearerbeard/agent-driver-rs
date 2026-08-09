# TODO - agent-driver-rs

> **Demoted to a pointer 2026-08-09.** The canonical library board is now
> the repo-local boardkit board at `.boardkit/boards/adr/` (ruled
> 2026-08-09; seeding record:
> `.boardkit/boards/adr/docs/board/evidence/2026-08-09-todo-inventory.md`).
> This file's former item-level content survives in git history at
> `674a093`.

Where things live now:

- **Library work** (crate internals, driver-side protocol handling):
  cards in `.boardkit/boards/adr/docs/board/cards/` - start at the
  generated `INDEX.md` / `board.md` / `graph.md` views there, or run
  `boardkit check` / `boardkit dag` from this repo root (the
  `.boardkit/manifest.toml` resolves the board).
- **Aura integration** (the four Phase 1 deliverables): the
  aura-agent-driver epic, card `aura/P19` on the aura family board -
  see the board charter's route table.
- **The open merge gate** (standards-audit branch `27b70d2`, 0.2.0
  boundary): card `adr/A1`.
- ADR wave details: [docs/adr/README.md](docs/adr/README.md);
  longer-range phase mapping:
  [docs/internal/agent-driver-roadmap.md](docs/internal/agent-driver-roadmap.md).
