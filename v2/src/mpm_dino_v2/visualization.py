"""Reference-neighbour graph visualization utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import torch


def unique_edge_segments(payload: dict) -> torch.Tensor:
    indices, mask = payload["neighbour_indices"], payload["neighbour_mask"]
    rows = torch.arange(indices.shape[0])[:, None].expand_as(indices)
    keep = mask & (rows < indices)
    starts, ends = rows[keep], indices[keep]
    return torch.stack((payload["x0"][starts], payload["x0"][ends]), dim=1)


def save_graph_visualization(payload: dict, output: str | Path, title: str | None = None) -> None:
    """Save three orthogonal projections and a 3D view of the fixed graph."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    points = payload["x0"][payload["particle_mask"]].cpu().numpy()
    segments = unique_edge_segments(payload).cpu().numpy()
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    views = [("XY", 0, 1), ("XZ", 0, 2), ("YZ", 1, 2)]
    for plot_id, (label, a, b) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, plot_id)
        for segment in segments:
            ax.plot(segment[:, a], segment[:, b], color="#3268a8", alpha=0.18, linewidth=0.45)
        ax.scatter(points[:, a], points[:, b], s=2, color="#d95f02", alpha=0.7)
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
    ax3 = fig.add_subplot(2, 2, 4, projection="3d")
    ax3.add_collection3d(Line3DCollection(segments, colors="#3268a8", alpha=0.18, linewidths=0.45))
    ax3.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, color="#d95f02", alpha=0.7)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_title("3D")
    fig.suptitle(title or "Effective-reference mutual-neighbour graph")
    fig.savefig(output, dpi=180)
    plt.close(fig)
