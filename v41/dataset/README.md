# V4.1 modular dynamics dataset

Use `collection.json` for the complete index or one JSONL file under `subsets/` for a purpose-specific experiment. Every manifest row points to one uniform `trajectory.npz` and one shared object `static.npz`.

Trajectory arrays: `trajectory_positions_m [61,2048,3]`, `point_velocities_m_s [61,2048,3]`, `point_active [61,2048]`, `times_s [61]`, and `point_ids [2048]`.

Static arrays: `reference_positions_m [2048,3]`, `dino_features [2048,384]`, `dino_valid [2048]`, and `point_material_ids [2048]`.

Reusable source meshes are stored by family under `source_glbs/`; their paths and SHA-256 hashes are recorded per object in `collection.json`.

Fluid is intentionally excluded. Soft contact penetration remains a known limitation; see `collection.json`.
