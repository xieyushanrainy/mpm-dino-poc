# V3 DINO-Centric Architecture Screen

## Summary

V3 should test three particle-native, future-compatible architecture families. All avoid dependence on continuous controller trajectories and instead use initial action metadata: impulse/force magnitude, direction/angle, contact point, `dt`, scale, and time since impulse.

The goal is to identify whether DINO is useful as direct particle conditioning, latent material conditioning, or action-conditioned token interaction.

## Candidate Architectures

- **Candidate 1: DINO-Conditioned Reference Graph Dynamics**
  - Main mechanism: local particle-particle interaction over the fixed reference graph.
  - Inputs: `x_t`, `v_t`, `x0`, `x_t - x0`, edge deformation, edge-relative velocity, final-DINO, action/contact features.
  - Uses 2-3 graph message-passing layers.
  - DINO is a per-particle node feature.
  - Tests whether DINO helps local recurrent graph dynamics directly.

- **Candidate 2: DINO / Geometry Latent Material Model**
  - Main mechanism: infer a compact object-level latent `z_object` from frame-0 geometry and DINO.
  - Latent encoder sees only `x0`, reference graph, DINO, validity/imputation flags.
  - Dynamics model uses graph message passing over recurrent state, conditioned by `z_object` and action embedding.
  - Use FiLM modulation first; add gates only if needed.
  - Tests whether DINO is better used as persistent material/object conditioning rather than per-particle input.

- **Candidate 3: Action-Conditioned Particle Transformer / Graph Hybrid**
  - Main mechanism: particles interact with initial action/contact tokens, then propagate through the reference graph.
  - Replace continuous controller tokens with future-compatible action tokens:
    - impulse vector token;
    - contact point token;
    - optional time token.
  - Particles cross-attend to these action tokens, then pass messages over the reference graph.
  - DINO can enter either as particle token input or latent conditioning.
  - Tests whether token-based action conditioning helps without relying on controller trajectories.

## Shared Design Rules

- No 3D U-Net or voxel grid in V3 main candidates.
- No continuous controller trajectory dependency.
- Use residual recurrent updates:
  - predict `delta_x` first;
  - derive velocity from displacement / `dt` unless a learned velocity update is explicitly ablated.
- Recompute edge deformation, relative velocity, contact-relative features, and action-time features at every recurrent step.
- Keep edge-vector and edge-length losses from V2.
- Select by recurrent H4/H8 validation, not one-step performance alone.

## DINO Ablations

Run DINO ablations after the first candidate screen, focusing on the strongest candidate.

Minimum ablations:

- final-DINO;
- zero-DINO;
- shuffled-DINO;
- geometry-only, for latent models.

DINO is considered meaningful only if real final-DINO beats zero/shuffled controls across recurrent horizons with a margin larger than seed variance.

## Experiment Plan

- Baseline:
  - reproduce V2 particle-only final-DINO under the same splits and seeds.

- Compact three-seed screen:
  - Candidate 1: DINO-conditioned reference graph;
  - Candidate 2: DINO/geometry latent graph;
  - Candidate 3: action-token transformer + graph hybrid.

- Evaluate:
  - H1, H4, H8, H16 particle error;
  - motion-stratified quartiles;
  - edge-vector and edge-length recurrent errors;
  - held-out rollout visualizations.

- Selection:
  - choose the best validation H4/H8 candidate across three seeds;
  - then run DINO control ablations on that candidate;
  - evaluate only the baseline and selected V3 winner on untouched test scenes.

## Interpretation Rules

- Candidate 1 wins: direct DINO-conditioned local graph dynamics is enough.
- Candidate 2 wins: DINO is most useful as object/material latent conditioning.
- Candidate 3 wins: action-token interaction is important even without continuous controller trajectories.
- No candidate beats V2 particle-only: current gains likely come from temporary controller information, dataset bias, or missing recurrent state rather than V3 architecture capacity.
- DINO controls fail: remove DINO from the main architecture and keep geometry/action graph dynamics as the future path.

## Assumptions

- Future inference provides initial force/impulse magnitude, direction/angle, and contact/application point.
- Object-level latent is the first latent granularity; part-level latents are deferred.
- Final-DINO means existing Block-11 DINOv3 ViT-B/16 features.
- Continuous controller points remain optional POC diagnostics, not required V3 inputs.
