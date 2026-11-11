## Worktrees

`data/` and `assets/` are gitignored, so a fresh worktree lacks them and training, the smoke test, and data-dependent tests fail or silently skip. Before running any of those in a worktree, link the data from the main checkout and give the worktree its own `experiments/`:

```bash
MAIN=$(git worktree list --porcelain | head -1 | cut -d' ' -f2)
[ -e data ] || ln -s "$MAIN/data" data
[ -e assets ] || ln -s "$MAIN/assets" assets
mkdir -p experiments
```

## Agent skills

### Issue tracker

Issues live in GitHub Issues on the fork `shoang22/match` (not upstream), via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
