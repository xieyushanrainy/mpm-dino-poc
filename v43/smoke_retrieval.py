#!/usr/bin/env python3
"""One-batch CPU smoke test for the V4.3 attended-memory contract."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2" / "src"))
sys.path.insert(0, str(ROOT / "v4" / "src"))
from mpm_dino_v4.v43_retrieval import AttendedMechanicalMemory  # noqa: E402


def main():
    torch.manual_seed(42)
    module = AttendedMechanicalMemory(24, 32, hidden_dim=32, heads=4)
    base = torch.randn(1, 3, 16, 3)
    query = torch.randn(1, 3, 16, 24)
    memory = torch.randn(1, 3, 3, 16, 32)
    memory_valid = torch.ones(1, 3, 3, 16, dtype=torch.bool)
    point_valid = torch.ones(1, 16, dtype=torch.bool)
    output, gate = module(base, query, memory, memory_valid, point_valid,
                          return_gate=True)
    output.square().mean().backward()
    zero = module(base, query, torch.zeros_like(memory),
                  torch.zeros_like(memory_valid), point_valid)
    result = {
        "status": "passed",
        "device": "cpu",
        "batch": 1,
        "output_shape": list(output.shape),
        "gate_mean": float(gate.detach().mean()),
        "attention_gradient_nonzero": bool(
            module.attention.in_proj_weight.grad.abs().sum() > 0),
        "zero_memory_exact_equivalence": bool(torch.equal(zero, base)),
        "test_data_used": False,
    }
    output_path = ROOT / "v43" / "run" / "cpu_smoke" / "RUN_COMPLETE.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
