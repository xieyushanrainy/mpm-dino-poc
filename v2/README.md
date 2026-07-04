# V2 Workspace

V2 adds effective reference geometry and local surface-deformation state to the
particle-grid surrogate. Read `../docs/v2/CONTEXT.md` and
`../docs/v2/PLAN.md` before implementation.

Intended layout:

```text
v2/src/mpm_dino_v2/   versioned package
v2/scripts/           preprocessing, training, evaluation and diagnostics
v2/tests/             unit and integration tests
```

V2 may read `data/shared/` and `data/v1/cache/`, but must write its versioned
cache schema only under `data/v2/` so V1 caches are not silently overwritten.
