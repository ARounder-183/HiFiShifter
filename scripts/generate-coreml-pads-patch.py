#!/usr/bin/env python3
"""Generate coreml_pads_patch.txt for the NSF-HiFiGAN model.

The stock model derives Pad `pads` at runtime, which the CoreML EP cannot
compile.  For the fixed 4096-frame input the derived values are constant;
this script evaluates them with ONNX Runtime and writes the patch that
build.rs applies when generating pc_nsf_hifigan_coreml.onnx.

Usage:  python scripts/generate-coreml-pads-patch.py
Re-run this whenever the stock ONNX model changes.
"""
import os
import numpy as np
import onnx
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "backend", "src-tauri", "resources", "models", "nsf_hifigan", "pc_nsf_hifigan.onnx")
OUT = os.path.join(ROOT, "backend", "src-tauri", "resources", "models", "nsf_hifigan", "coreml_pads_patch.txt")

m = onnx.load(SRC)
init_names = {t.name for t in m.graph.initializer}
pad_nodes = [n for n in m.graph.node if n.op_type == "Pad"]

dyn = []
for n in pad_nodes:
    ins = list(n.input)
    pads_name = ins[1] if len(ins) > 1 else None
    if pads_name and pads_name not in init_names and pads_name not in dyn:
        dyn.append(pads_name)

if not dyn:
    print("No dynamic Pad pads found; nothing to patch.")
    raise SystemExit(0)

for name in dyn:
    m.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
mel = np.zeros((1, 128, 4096), dtype=np.float32)
f0 = np.full((1, 4096), 440.0, dtype=np.float32)
vals = sess.run(dyn, {"mel": mel, "f0": f0})

lines = ["# CoreML Pad pads patch: <pads-input-name>=v0,v1,..."]
for name, arr in zip(dyn, vals):
    flat = [int(x) for x in np.asarray(arr).flatten()]
    lines.append(f"{name}={','.join(map(str, flat))}")
    print(name, flat)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", OUT, os.path.getsize(OUT), "bytes")
