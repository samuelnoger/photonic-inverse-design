# 2D Photonic Inverse Design via Differentiable FDFD

A lightweight, educational framework for the inverse design and topology optimization of integrated silicon photonic devices. Built from scratch using **SciPy** (sparse 2D Finite-Difference Frequency-Domain solver) and **PyTorch** (custom `autograd` adjoint backpropagation, density continuation, circular neighborhood biasing, and discrete level-set boundary polishing).

Instead of hand-tuning waveguide geometries, this project treats the silicon permittivity distribution inside a compact design region as a pixel density grid and optimizes it via adjoint-based topology optimization inspired by the Stanford Nanoscale and Quantum Photonics Lab (Su et al., 2017; Piggott et al., 2017).

---

## Key Features

- **2D Scalar FDFD Helmholtz Solver ($E_z$ / TM):** 5-point finite-difference stencil with stretched-coordinate Perfectly Matched Layers (PML), exact 1D discrete slab eigenmode injection/projection, and 2D Silicon-on-Insulator (SOI) effective-index permittivities (`eps_si = 7.84`, `eps_bg = 2.08`).
- **Exact Adjoint Differentiation:** Custom `torch.autograd.Function` that caches the forward sparse LU factorization (`splu`) and computes exact analytical permittivity gradients ($\partial \mathcal{L} / \partial \varepsilon = -k_0^2 \Delta\varepsilon\,\text{Re}\{\lambda_{\text{adj}} E_z\}$) in a single transposed solve (`trans="T"`).
- **Dual Fabrication-Aware Optimization Pipelines:**
  - **Density Continuation (`method="beta"`, default for splitters):** Bounded-sigmoid parameterization with a Gaussian length-scale filter (`smooth_sigma`), exponential $\tanh(\beta)$ continuation ($\beta: 1.5 \to 32.0$), and optional erosion/dilation robustness (`robust_delta`).
  - **Two-Stage Neighbor Biasing + Discrete Level-Set (`method="nb"`, default for WDMs, following Su et al., 2017):**
    - **Stage 1 (Continuous + Neighbor Biasing):** Optimizes pixel densities in $z \in [0, 1]$ using a circular disk neighbor-biasing update (`bias_k`, `bias_k_avg`, `bias_p`, radius `smooth_sigma`) paired with a late-stage $\tanh(\beta)$ sharpening ramp ($\beta: 1.0 \to 16.0$) to suppress sub-wavelength speckles while closing the gray-to-binary gap.
    - **Stage 2 (Discrete Level-Set Boundary Polish):** Evolves a signed distance field $\phi$ via Adam with Gaussian-smoothed boundary velocities (`ls_vel_sigma`), morphological opening/closing (`drc_radius`) for minimum feature sizes, and a Straight-Through Estimator (STE) to optimize directly over binary DRC-cleaned layouts.
- **Clean Dataclass Configuration:** Centralized `PhotonicConfig` dataclass in `solver/config.py` with automatic grid/PML margin calculation and per-device presets (`splitter_1x2`, `splitter_1x3`, `wdm_1x2`, `wdm_1x3`).
- **Honest Binary Verification:** All final metrics, field plots, and broadband spectral sweeps are evaluated on the hard-thresholded binary silicon layout ($\rho \in \{0, 1\}$) via a fresh FDFD solve.

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/samuelnoger/photonics-inverse-design.git
cd photonics-inverse-design
pip install numpy scipy torch matplotlib tqdm
```

### 2. Optimize a Photonic Device

Explore the interactive notebooks (`fit_design.ipynb`, `fit_wdm.ipynb`) or run directly from Python:

```python
from design.fitter import PhotonicFitter
from verification.plotting import plot_result, plot_wdm_result, plot_spectrum

# 1x2 Power Splitter (40/60 split, beta continuation)
fitter_1x2 = PhotonicFitter(device_type="splitter_1x2", smooth_sigma=3.5)
res_1x2 = fitter_1x2.fit(target_ratio=0.4, init_mode="branched", steps=180)
plot_result(res_1x2)

# 1x3 Power Splitter (40/20/40 split, beta continuation)
fitter_1x3 = PhotonicFitter(device_type="splitter_1x3", smooth_sigma=3.5)
res_1x3 = fitter_1x3.fit(target_ratios=[0.4, 0.2, 0.4], init_mode="branched", steps=180)
plot_result(res_1x3)

# 2-Channel WDM Demultiplexer (1.31 µm -> Port 1, 1.55 µm -> Port 2, two-stage NB)
fitter_wdm2 = PhotonicFitter(device_type="wdm_1x2", smooth_sigma=3.0)
res_wdm2 = fitter_wdm2.fit(init_mode="filled", steps=250)
plot_wdm_result(res_wdm2, fitter_wdm2)
plot_spectrum(res_wdm2, fitter_wdm2)

# 3-Channel WDM Demultiplexer (1.50, 1.54, 1.58 µm, two-stage NB)
fitter_wdm3 = PhotonicFitter(device_type="wdm_1x3", smooth_sigma=3.0)
res_wdm3 = fitter_wdm3.fit(init_mode="filled", steps=300)
plot_wdm_result(res_wdm3, fitter_wdm3)
plot_spectrum(res_wdm3, fitter_wdm3)
```

---

## Repository Structure

```text
├── solver/
│   ├── config.py         # PhotonicConfig dataclass & device presets
│   └── solver.py         # 2D FDFD Helmholtz solver, PML, & 1D slab eigenmode solver
├── design/
│   ├── adjoint.py        # Custom PyTorch autograd function for FDFD adjoint gradients
│   ├── fitter.py         # PhotonicFitter (beta continuation & two-stage NB + Level-Set)
│   ├── geometry.py       # Waveguide masks, [0, 1] density initializers, & port pinning
│   └── losses.py         # Mode-overlap power measurement, WDM isolation, & TV regularization
└── verification/
    ├── plotting.py       # Permittivity, wavefield |Ez|², convergence, & spectrum plots
    └── stability.py      # Fabrication erosion/dilation robustness checks
```

---

## Scope & Limitations

This repository is designed as a fast, self-contained 2D effective-index sandbox for learning PDE-constrained photonic inverse design. It does not model 3D out-of-plane radiation losses, material dispersion, or full foundry DRC rules, and single-wavelength-per-channel WDM optimization produces narrowband interferometric passbands unless trained over multi-wavelength channel windows.

---

## References & Citation

The two-stage optimization pipeline, circular neighborhood biasing scheme (Equations 2–6 of Su et al., 2018), and three-channel WDM demultiplexer formulation implemented in this repository are based on the work of Jelena Vučković's group at Stanford University:

1. **L. Su, A. Y. Piggott, N. V. Sapra, J. Petykiewicz, and J. Vučković**, *"Inverse design and demonstration of a compact on-chip narrowband three-channel wavelength demultiplexer,"* ACS Photonics **5**(2), 301–305 (2018) / [arXiv:1709.08809](https://arxiv.org/abs/1709.08809) (2017).
2. **A. Y. Piggott, J. Petykiewicz, L. Su, and J. Vučković**, *"Fabrication-constrained nanophotonic inverse design,"* Scientific Reports **7**, 1786 (2017).
3. **A. Y. Piggott, J. Lu, K. G. Lagoudakis, J. Petykiewicz, T. M. Babinec, and J. Vučković**, *"Inverse design and demonstration of a compact and broadband on-chip wavelength demultiplexer,"* Nature Photonics **9**, 374–377 (2015).

---

## Acknowledgements

Built as a personal learning project on electromagnetic simulation and photonic inverse design, with AI coding assistants used for code structuring and debugging.