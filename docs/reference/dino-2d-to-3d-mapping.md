# Rough DINO 2D-to-3D Feature Mapping

Based on the mapping described in MatPhys.

1. Reconstruct the object from a keyframe as 3D Gaussians or points with centres $x_i \in \mathbb{R}^3$.
2. Run a frozen DINO encoder on the same keyframe to obtain a dense 2D feature map $F$.
3. Project each visible 3D centre into the image using the known camera:

   $$u_i = \pi(x_i), \qquad \tilde u_i \sim K[R\mid t]\tilde x_i.$$

4. Sample the DINO feature map at the projected location and attach it to the 3D point:

   $$f_i^{\mathrm{DINO}} = F(u_i).$$

   In practice, bilinear sampling is appropriate because the DINO feature map is lower resolution than the input image.
5. Cluster the resulting per-point features to form semantic 3D parts. MatPhys uses five clusters.

```text
3D point/Gaussian centre
        -> camera projection
2D keyframe coordinate
        -> sample dense DINO map
per-point 3D semantic feature
        -> feature clustering
semantic 3D parts
```

Only visible surface points receive direct image features. MatPhys states that features for invisible points are propagated from symmetric visible counterparts when available. It does not specify the visibility test, symmetry detection, DINO layer, interpolation method, or fallback for points without a symmetric match; these require implementation choices in a reproduction.
