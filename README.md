# Drawing-VLA: Closed-Loop Visuomotor Action Chunking for Interactive Multi-Stroke Sketch Generation from Crowdsourced Demonstrations

[![Paper](https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader)](./Drawing_VLA.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

Official PyTorch implementation and web benchmark platform for the paper:  
**"Drawing-VLA: Closed-Loop Visuomotor Action Chunking for Interactive Multi-Stroke Sketch Generation from Crowdsourced Demonstrations"**  
**Author:** Seungjoon Son (sonsjoon715@gmail.com)

[Paper (PDF)](./Drawing_VLA.pdf) | [한국어 문서 (README_KO.md)](./README_KO.md)

---

## Overview

Autoregressive sequence models for vector sketch generation (e.g., SketchRNN) typically operate open-loop, predicting sequential coordinate displacements blind to the actual rendered canvas state. Consequently, they suffer from compounding trajectory drift and stroke collapse during multi-stroke transitions.

**Drawing-VLA** reformulates 2D canvas sketching as a closed-loop robot end-effector manipulation task. Using an architecture inspired by Action Chunking with Transformers (ACT), the agent observes a 2-channel visual canvas state (accumulated ink + target template) alongside pen proprioception and discrete task tokens, predicting a chunk of $K = 16$ future trajectory waypoints in a receding-horizon manner.

### Closed-Loop Rollout vs. Single-Step Baseline
![Figure 1: Rollout Comparison](./paper/figures/fig1_rollout_comparison.png)
*Figure 1: Comparison between single-step autoregressive rollout and closed-loop Action Chunking. Action Chunking prevents error accumulation and cleanly decouples multi-stroke trajectories on Apple (3 strokes) and Sprout (4 strokes).*

### Robustness to Real-Time Spatial Perturbations (+20px Displacement)
![Figure 4: Perturbation Recovery](./paper/figures/fig4_perturbation_recovery.png)
*Figure 4: Closed-loop visual feedback enables real-time error recovery when perturbed by an unexpected +20px spatial offset at step $t=150$, smoothly returning to the target contour within 42 steps where an open-loop replay baseline permanently drifts.*

---

## Key Technical Results

* **Multi-Stroke Decoupling:** Predicts $K = 16$ action chunks paired with an $L_1$ coordinate objective and a $10\times$ minority-class pen-up weighting, achieving 100.0% coverage and 100.0% precision (99.9–100.0/100 S-Tier) on multi-stroke targets.
* **Dual-Mode Synthesis:** Trained with 50% goal-channel dropout ($p_{\text{drop}} = 0.5$), enabling both guided template tracing and autonomous blank-canvas contour generation (0.887 F1) from memory.
* **Real-Time Perturbation Recovery:** Smoothly compensates for unexpected $+20\text{px}$ (+0.156 normalized) end-effector shifts within 42 control steps.
* **Edge Inference Efficiency:** Measures 6.11 ms per chunk (163 chunks/sec) on an Intel Core Ultra 5 325 edge CPU (4 threads), operating comfortably within a 60 Hz (16.67 ms) browser control budget.
* **Crowdsourced Flywheel:** 324 verified human trajectories collected via a gamified web benchmark with real-time signed distance field (SDF) scoring and asynchronous ghost battles.

---

## Repository Structure

```
drawing-vla/
├── collector/          # Zero-dependency Vanilla HTML/JS canvas benchmark & collection client
├── train/              # Visuomotor policy architectures, training, and closed-loop rollout
│   ├── policies.py     # ACT (Transformer) and Convolutional baseline policy models
│   ├── dataset.py      # Dual-channel observation rendering and trajectory sampling
│   ├── train_policy.py # Policy training pipeline (CUDA / Intel XPU / CPU)
│   ├── rollout_eval.py # Closed-loop evaluation engine with receding-horizon ensembling
│   └── brush.py        # Organic Catmull-Rom spline and pressure brush renderer
├── api_core.py         # Benchmark backend, trajectory verification, and scoring logic
├── scoring.py          # Exact grid-indexed contour matching and dual-tolerance scoring
├── submission.py       # Trajectory validation, human-motion verification, and anti-abuse
├── server.py           # Standalone local data collection and evaluation server
├── paper/              # LaTeX source, benchmark figures, and perturbation results
├── tests/              # Comprehensive test suite (scoring, anti-abuse, benchmark APIs)
└── Drawing_VLA.pdf     # Full paper PDF
```

---

## Quick Start

### 1. Requirements and Setup

The benchmark and collection server runs with zero external dependencies (Python standard library only). For training and evaluating visuomotor policies:

```bash
git clone https://github.com/5Color/drawing-vla.git
cd drawing-vla
pip install -r requirements.txt
```

### 2. Run Local Data Collection & Game Server
Start the standalone local collection server on port 8000:
```bash
python server.py
# Open http://localhost:8000 in your browser (tablet stylus or touch supported)
```

### 3. Run Benchmark Test Suite
Execute the test suite to verify scoring, trajectory generation, and security checks:
```bash
python -m unittest tests.test_benchmark
```

### 4. Training and Closed-Loop Rollout
```bash
# 1) Train unified dual-mode ACT policy
python train/train_policy.py --head transformer --epochs 60 --goal-drop 0.5

# 2) Evaluate closed-loop rollout with temporal ensembling and organic brush rendering
python train/rollout_eval.py --ckpt outputs/apple_vla_policy.pth --exec-steps 12 --ensemble --brush
```

---

## Citation

If you find this work or code useful in your research, please cite:

```bibtex
@article{son2026drawingvla,
  title={Drawing-VLA: Closed-Loop Visuomotor Action Chunking for Interactive Multi-Stroke Sketch Generation from Crowdsourced Demonstrations},
  author={Son, Seungjoon},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

This project is licensed under the [MIT License](./LICENSE).
