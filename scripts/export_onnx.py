"""
CLI Script: Export MobileNetV2 to ONNX and Benchmark Edge Inference Latency
Verifies ONNX model validity and benchmarks CPU/GPU throughput and latency.
"""

import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_factory import load_model_from_checkpoint
from src.utils.plant_classes import NUM_CLASSES

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mobilenetv2_robust.pth",
                        help="Path to trained PyTorch model checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/mobilenetv2_robust.onnx",
                        help="Path to export ONNX file")
    parser.add_argument("--batch_size", type=int, default=1, help="Benchmark batch size")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark test iterations")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("MobileNetV2 Edge Deployment & ONNX Export")
    print("==================================================")

    # 1. Load PyTorch model
    device = "cpu"
    model, _ = load_model_from_checkpoint(args.checkpoint, model_type="robust", num_classes=NUM_CLASSES, device=device)
    model.eval()

    # 2. Dummy input
    dummy_input = torch.randn(args.batch_size, 3, 224, 224, dtype=torch.float32)

    # 3. Export to ONNX
    print(f"[Export] Exporting PyTorch model to '{out_path}'...")
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print(f"[Export] Model successfully exported to ONNX.")

    # 4. Check ONNX validity
    try:
        import onnx
        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)
        print("[Verify] ONNX model structure validated successfully.")
    except Exception as e:
        print(f"[Verify] Warning: {e}")

    # 5. Benchmark Latency with ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"\n[Benchmark] Benchmarking inference latency across {args.iterations} iterations...")
        session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        
        test_data = np.random.randn(args.batch_size, 3, 224, 224).astype(np.float32)

        # Warmup
        for _ in range(15):
            _ = session.run(None, {input_name: test_data})

        # Timing loop
        latencies = []
        for _ in range(args.iterations):
            t0 = time.perf_counter()
            _ = session.run(None, {input_name: test_data})
            latencies.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = np.mean(latencies)
        p95_ms = np.percentile(latencies, 95)
        fps = 1000.0 / mean_ms * args.batch_size

        print("==================================================")
        print("EDGE INFERENCE BENCHMARK RESULTS (CPU)")
        print("==================================================")
        print(f"Average Latency: {mean_ms:.2f} ms per batch")
        print(f"95th Percentile: {p95_ms:.2f} ms")
        print(f"Throughput:      {fps:.1f} FPS (Frames per second)")
        print(f"Model File Size: {out_path.stat().st_size / (1024*1024):.2f} MB")
        print("==================================================")
    except Exception as e:
        print(f"[Benchmark] Could not run ONNX Runtime benchmark: {e}")

if __name__ == "__main__":
    main()
