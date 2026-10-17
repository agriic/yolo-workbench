"""ONNX export of a run's best weights, in a process of its own (started by `TrainingManager.export_onnx`).

`python -m yolo_workbench.export_runner <spec.json>`. The spec names the framework, the weights, where the
.onnx goes, a scratch folder, and the export options. The weights are copied into the scratch folder and
exported there on the CPU (a training run may hold the GPU, and may rewrite best.pt meanwhile), and the
.onnx is moved into place only once it is complete: a failed or killed export leaves nothing at `output`.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .fsutil import replace


def export_ultralytics(weights: Path, spec: dict) -> Path:
    from ultralytics import YOLO

    options = {"format": "onnx", "device": "cpu", "dynamic": bool(spec.get("dynamic")), "simplify": bool(spec.get("simplify", True))}
    if spec.get("imgsz"):
        options["imgsz"] = spec["imgsz"]
    if spec.get("nms"):
        options["nms"] = True
    return Path(YOLO(str(weights)).export(**options))


def export_rfdetr(weights: Path, spec: dict) -> Path:
    from rfdetr import RFDETR

    # the checkpoint restores the trained resolution, the default input; another one must be divisible by its patch grid
    model = RFDETR.from_checkpoint(str(weights), device="cpu")
    imgsz = spec.get("imgsz")
    shape = None if not imgsz else (imgsz, imgsz) if isinstance(imgsz, int) else (imgsz[0], imgsz[1])
    return Path(model.export(output_dir=str(weights.parent / "onnx"), output_name=weights.stem, shape=shape, dynamic_batch=bool(spec.get("dynamic")), verbose=False))


def main(spec_path: str) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    weights, output, scratch = Path(spec["weights"]), Path(spec["output"]), Path(spec["scratch"])
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    try:
        copy = scratch / f"best{weights.suffix}"
        shutil.copy2(weights, copy)
        exported = export_rfdetr(copy, spec) if spec.get("framework") == "rfdetr" else export_ultralytics(copy, spec)
        if not exported.is_file():
            raise RuntimeError(f"The exporter reported {exported}, which does not exist")
        partial = output.with_name(f".{output.name}.partial")
        shutil.move(exported, partial)
        replace(partial, output)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"Exported {output.as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
