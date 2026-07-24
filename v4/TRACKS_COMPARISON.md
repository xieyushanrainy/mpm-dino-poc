# V4 Architecture Comparison and Open Design Decision

This note records the current evidence and design trade-offs without selecting a
final architecture. The intended use case is a generalizing, collision-aware
particle dynamics model: given two state frames, initial velocity, and known
scene geometry, predict a physically plausible future path for unseen objects,
initial conditions, and obstacle configurations. Point-aligned DINO is intended
to provide stable object-specific visual information that may help long-horizon
prediction.

## The two current tracks

### Track A1: autoregressive particle graph dynamics

At every step, Track A1 receives two consecutive point states and applies a
fixed-reference particle graph:

`(X[t], X[t+1], graph, DINO, masks, environment) -> X[t+2]`.

It predicts a residual over constant velocity and is then rolled forward using
its own previous predictions. DINO is point-aligned: each point's projected
DINO feature is part of that point's graph-node input.

Strengths:

- Represents a reusable local transition rule rather than one fixed trace.
- Can start at arbitrary valid times and run for arbitrary horizons.
- Naturally accepts a changed initial velocity through the two input states or
  an explicit velocity field.
- Can condition on changing scene geometry at every predicted step.
- Preserves point-aligned DINO, so visual information can affect local
  deformation and contact behaviour.
- Has strong immediate prediction: frame-0 test H1 is approximately 7.3 mm.

Limitations:

- One-step errors enter the next input and compound during rollout.
- Teacher forcing creates an exposure mismatch: training sees clean histories,
  while inference sees model-generated histories.
- A missed collision or incorrect contact impulse can contaminate the entire
  future trajectory.
- The current floor-only environmental input is insufficient for new obstacle
  layouts.
- Existing DINO controls did not establish a robust improvement beyond matched
  zero/shuffled controls.

### Track B: one-shot graph-temporal trajectory generation

Track B predicts all future frames simultaneously from the first two frames:

`(X[0], X[1], graph, pooled DINO, masks, environment) -> X[2:61]`.

It uses an initial graph encoder, a gravity-aware ballistic reference, and four
factorized blocks. Each block applies fixed-kNN graph message passing across
points at each future time and temporal self-attention along each point's
59-frame future sequence.

The present DINO path is global rather than local:

`[N,384] -> [N,16] -> masked mean/max -> [33] -> [128] -> FiLM`.

The object-level condition produces one gamma and beta vector of width 128 per
block, broadcast over all points and times. Point-to-feature correspondence is
discarded after pooling.

Strengths:

- No recursive prediction feedback, so it avoids classical rollout drift.
- Directly optimizes temporal coherence over the complete predicted horizon.
- Stronger medium- and long-horizon test behaviour in the current fixed-task
  experiment.
- Three-seed zero-DINO mean RMSE is 57.40 mm at H8, 51.35 mm at H16, and
  77.88 mm at H59, better than frame-0 Track A1 and constant velocity at those
  horizons.

Limitations:

- It is a conditional trace generator, not a reusable stepwise simulator.
- The current interface is tied to frame 0/1 and a 59-frame output horizon.
- It has poor immediate anchoring: zero-DINO H1 is 23.81 mm versus 7.28 mm for
  constant velocity.
- The ballistic reference becomes badly wrong after contact, leaving the model
  to undo an increasingly implausible trajectory.
- Future temporal attention is non-causal. This is legitimate for one-shot
  prediction but does not represent online simulation.
- All point-time tokens are resident together, increasing memory and compute.
- Pooled DINO is a severe bottleneck, so its result is not a definitive test of
  point-aligned visual information.

## Current evidence

All figures below are object-weighted test RMSE in millimetres on the ten
held-out rigid+soft objects. Track B values are three-seed means; the seed-123
pair was capped at epoch 96, and zero seed 456 was capped at epoch 100.

| Model | H1 | H4 | H8 | H16 | H59 |
|---|---:|---:|---:|---:|---:|
| Constant velocity | 7.28 | 49.23 | 81.72 | 60.05 | 158.21 |
| Track A1, zero DINO | 7.30 | 49.42 | 82.44 | 65.07 | 249.74 |
| Track A1, real DINO | 7.29 | 49.32 | 82.05 | 64.81 | 254.34 |
| Track B, zero DINO | 23.81 | 28.04 | 57.40 | 51.35 | 77.88 |
| Track B, real pooled DINO | 29.76 | 34.36 | 62.90 | 57.79 | 64.76 |

Interpretation:

- Track A1/constant velocity are substantially better at H1.
- Track B zero DINO is substantially better at H8, H16, and H59.
- Track B has not met the full architectural acceptance criterion because its
  H1 degradation is material.
- Real pooled DINO is worse than zero DINO at H1/H4/H8/H16 for all three
  seeds. It improves aggregate H59, principally for rigid trajectories, but
  worsens the soft-body H59 mean.
- Therefore, pooled-DINO benefit is not established for the co-primary H8/H16
  goal. This does not rule out utility from persistent point-aligned DINO.

## Requirements implied by the intended use case

Neither current model has enough environmental representation to support a
claim of obstacle-layout generalization. A future collision-aware model should
receive local scene queries for each point at every predicted state, for
example:

- Signed distance to the nearest obstacle surface.
- Contact-surface normal or SDF gradient.
- Local obstacle velocity for moving environments.
- Optional scene geometry embedding when SDF queries alone are insufficient.

The training distribution must also vary independently over:

- Object identity and family.
- Initial linear/angular velocity, orientation, and drop height.
- Obstacle layout: floors, walls, ramps, boxes, wedges, and other relevant
  contact configurations.
- Contact material properties where these are in scope.

Evaluation should separately hold out objects, scene layouts, and velocity
regimes. A result on a single floor-only configuration cannot demonstrate
collision-aware scene generalization.

## Hybrid option: local dynamics plus long-horizon planning

A hybrid is a design option, not a selected direction. Its motivation is the
complementary empirical behaviour: Track A1/CV anchor the first step well,
while Track B produces a less drifting long-range plan.

One form is a two-part prediction:

`X_local[t+2] = A1(X[t], X[t+1], local scene queries, point DINO)`

`X_plan[2:T] = full_trajectory_model(X[0], X[1], scene, point/global visual features)`

The local transition is used for online rollout and immediate contact response.
The trajectory model supplies a long-range correction, consistency objective,
or future-state prior. A horizon-dependent blend is possible:

`X_hat[h] = alpha[h] * X_local[h] + (1 - alpha[h]) * X_plan[h]`.

Here `alpha[h]` should be learned or validated, not assumed. It would normally
be close to one near H1 and may decrease for later horizons. A less coupled
variant keeps Track B only as an auxiliary multi-horizon training target for an
autoregressive model, avoiding a direct blend at inference.

Potential advantages:

- Retains a causal, arbitrary-start, scene-reactive local simulator.
- Uses global temporal structure to discourage long-range drift.
- Can explicitly preserve H1 accuracy while improving H8+ stability.
- Allows persistent point-aligned DINO in the local model and a separate global
  visual summary in the planning model.

Potential risks:

- The two predictions can disagree at contact events.
- Blending positions can violate physical constraints or blur impacts.
- It increases training, tuning, and evaluation complexity on an already small
  object-level dataset.
- A planning branch can hide local-transition errors unless both branches are
  evaluated independently.

## Open decisions for the next discussion

1. Is the primary goal an online/reusable simulator, a fixed-horizon trajectory
   predictor, or both?
2. Which obstacle and initial-condition variations can be generated reliably in
   the data pipeline?
3. Should the next experiment first add scene-SDF conditioning to Track A1, add
   local DINO to Track B, or test a hybrid objective?
4. What minimum H1 constraint is acceptable while optimizing H8/H16/H59?
5. Should DINO be tested as persistent point tokens, cross-attended visual
   tokens, global FiLM, or a combination under matched controls?

Related raw results are in `runs/track_b/comparison_three_seed.json` and the
V4 experimental conclusions are in `RESULTS.md`.
