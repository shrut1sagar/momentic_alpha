# Developer Notes

## Phase 1 – Scaffolding & Docs
We start with structure and documentation to lock in boundaries early, make imports predictable, and avoid accidental coupling as logic arrives. A clear skeleton de-risks later refactors, keeps core functions stateless, and smooths migration to AWS batch/containers by separating I/O, orchestration, and pure computation from day one.

### Decisions and Assumptions
- Use the spec’s directory layout verbatim (docs, config, data/{raw,processed,results}, state, src subpackages, actions, tests).
- Initialize Python packages with `__init__.py` so modules resolve cleanly across src/*.
- Keep `main.py`/action stubs minimal; no strategy, backtest engine, or venue logic in Phase 1.
- Favor CSV contracts and explicit state folders even before logic exists to guide future I/O patterns.
- AWS readiness is architectural: stateless core + explicit state files; no cloud-specific code yet.
- Testing folder present but empty; real tests arrive with utilities/logic in later phases.
