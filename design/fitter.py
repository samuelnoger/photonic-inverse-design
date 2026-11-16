import time
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import List, Optional

from solver.config import get_config, PhotonicConfig
from solver.solver import HelmholtzSolver2D
from design.geometry import build_geometry
from design.losses import splitter_loss, wdm_loss
from design.adjoint import solve_fdfd_differentiable


class PhotonicFitter:
    """
    Two-Stage Photonic Inverse Design Fitter (following Su et al., Stanford / Vuckovic group).
    All default parameters live in `solver/arguments.py` and can be overridden in `__init__` or `fit()`.
    """
    def __init__(self, device_type="splitter_1x2", device=None, smooth_sigma=None, flare_margin=None, **overrides):
        # Collect any explicit non-None arguments into overrides and build the PhotonicConfig
        if device is not None:
            overrides["device"] = device
        if smooth_sigma is not None:
            overrides["smooth_sigma"] = smooth_sigma
        if flare_margin is not None:
            overrides["flare_margin"] = flare_margin

        self.cfg: PhotonicConfig = get_config(device_type=device_type, **overrides)

        self.device = torch.device("cpu") if self.cfg.device in ("auto", "cpu") else torch.device(self.cfg.device)
        self.device_type = self.cfg.device_type

        # Build geometry and per-wavelength FDFD solvers 
        self.geometry = build_geometry(self.cfg, device=self.device)

        is_wdm = self.device_type.startswith("wdm")
        self.wls = self.geometry.wls if is_wdm else [self.cfg.wl]

        self.solvers = [HelmholtzSolver2D.from_args(self.cfg, geometry=self.geometry, wl_override=w) for w in self.wls]
        self.solver = self.solvers[0]
        self.loss_fn = wdm_loss if is_wdm else splitter_loss

        self.x0, self.x1 = self.geometry.x0, self.geometry.x1
        self.y0, self.y1 = self.geometry.y0, self.geometry.y1

        # Uniform circular disk kernel of radius r (Eq. 5-6 in Su et al. for z_avg)
        self.smooth_sigma = float(self.cfg.smooth_sigma)
        r = max(1, int(np.round(self.smooth_sigma)))
        self.disk_rad = r
        coords = torch.arange(-r, r + 1, device=self.device, dtype=torch.float32)
        kx, ky = torch.meshgrid(coords, coords, indexing="ij")
        disk = ((kx ** 2 + ky ** 2) <= (self.smooth_sigma ** 2)).float()
        self.disk_kernel = (disk / disk.sum()).unsqueeze(0).unsqueeze(0)

        # Structuring disk for Stage 2 discrete minimum hole/feature enforcement
        y_g, x_g = np.ogrid[-r : r + 1, -r : r + 1]
        self.struct_disk = (x_g ** 2 + y_g ** 2) <= (self.smooth_sigma ** 2)

        # Soft pre-filter kernel (radius ~ 1.1 px) to smooth pixel-grid staircasing on gradients
        g_rad = 2
        g_c = torch.arange(-g_rad, g_rad + 1, device=self.device, dtype=torch.float32)
        gx, gy = torch.meshgrid(g_c, g_c, indexing="ij")
        g_k = torch.exp(-0.5 * (gx ** 2 + gy ** 2) / (1.1 ** 2))
        self.grad_kernel = (g_k / g_k.sum()).unsqueeze(0).unsqueeze(0)
        self.g_pad = g_rad

        # Port & corner pinning in [0, 1] from self.cfg.flare_margin
        self.pin, self.pin_val = self.geometry.get_pin_masks(
            n_pin=r + 2, flare_margin=self.cfg.flare_margin
        )

        self.stage1_frac = self.cfg.stage1_frac

    def evaluate(self, rho_np, solver=None):
        s = solver if solver is not None else self.solver
        Ez = s.solve(s.make_eps(rho_np))
        powers = s.measure_port_powers(Ez)
        return Ez, powers

    @torch.no_grad()
    def _apply_neighbor_bias(self, z, k=0.012, k_avg=10.0, p=3):
        """
        Implements Su et al. Equations (2)-(6): Neighbor Biasing in [0, 1].
          z_avg = uniform circular disk average of z within radius r
          h(z)  = 0.5 + k_avg * (0.5 - z_avg)^(2p + 1)
          b_n(z)= (z - h(z)) / (1 - 2k) + 0.5
          z    <- clip(b_n(z), 0, 1)
        Note: Because (0.5 - z_avg) is in [-0.5, 0.5], (0.5 - z_avg)^7 has max magnitude
        (0.5)^7 = 1/128, so k_avg ~ 10-16 scales the threshold shift to ~0.10.
        """
        r = self.disk_rad
        z_4d = F.pad(z.unsqueeze(0).unsqueeze(0), (r,) * 4, mode="replicate")
        z_avg = F.conv2d(z_4d, self.disk_kernel).squeeze()

        h_z = 0.5 + k_avg * ((0.5 - z_avg) ** (2 * p + 1))
        z_biased = (z - h_z) / (1.0 - 2.0 * k) + 0.5
        z_clipped = torch.clamp(z_biased, 0.0, 1.0)
        z_pinned = self.pin * self.pin_val + (1.0 - self.pin) * z_clipped
        z.copy_(z_pinned)

    def _clean_discrete_levelset(self, rho):
        """
        Stage 2 Fabrication-Constrained Discrete Level-Set Projection:
        Enforces minimum hole width and minimum radius of curvature at every step in Stage 2.
        """
        rho_np = rho.detach().cpu().numpy()
        # Smooth level-set function slightly to enforce minimum radius of curvature (~40 nm)
        phi = gaussian_filter(rho_np - 0.5, sigma=max(0.8, 0.45 * self.smooth_sigma), mode="nearest")
        binary_raw = phi > 0.0

        # Pad with edge values so closing/opening never erodes silicon touching the design box wall
        pad = self.disk_rad + 2
        b_pad = np.pad(binary_raw, pad, mode="edge")
        # Close oxide holes smaller than struct_disk, then open silicon islands smaller than struct_disk
        b_clean = binary_closing(b_pad, structure=self.struct_disk)
        b_clean = binary_opening(b_clean, structure=self.struct_disk)
        binary_clean = b_clean[pad:-pad, pad:-pad]

        pin_np = self.pin.detach().cpu().numpy() > 0.5
        pin_v = self.pin_val.detach().cpu().numpy() > 0.5
        binary_clean = np.where(pin_np, pin_v, binary_clean).astype(np.float32)

        target_t = torch.from_numpy(binary_clean).to(rho.device)
        return rho + (target_t - rho).detach()

    def _project_density(self, z, symmetric=False, shift=0.0, discrete_stage=False):
        if symmetric:
            z = 0.5 * (z + torch.flip(z, dims=[-1]))
        z_p = self.pin * self.pin_val + (1.0 - self.pin) * torch.clamp(z, 0.0, 1.0)

        # Lightly smooth pixel-level staircasing so boundary gradients move smooth level-set contours
        z_4d = F.pad(z_p.unsqueeze(0).unsqueeze(0), (self.g_pad,) * 4, mode="replicate")
        rho = F.conv2d(z_4d, self.grad_kernel).squeeze()

        if shift != 0.0:
            eta_shift = shift * 0.15 if abs(shift) > 0.12 else shift
            rho = torch.clamp(rho + eta_shift * (1.0 - self.pin), 0.0, 1.0)

        rho = self.pin * self.pin_val + (1.0 - self.pin) * rho
        if discrete_stage:
            rho = self._clean_discrete_levelset(rho)
        return rho

    def _assemble_eps_norm(self, rho):
        eps = self.geometry.base_eps_norm.clone()
        eps[0, 0, self.x0 : self.x1, self.y0 : self.y1] = rho
        return eps

    def _resolve_targets(self, target_ratio, target_ratios):
        if self.device_type.startswith("wdm"):
            # N x N identity matrix: wavelength i targets port i
            return torch.eye(self.geometry.n_ports, dtype=torch.float32, device=self.device)
        elif target_ratios is not None:
            r = np.asarray(target_ratios, dtype=np.float32)
            t_list = [(r / r.sum()).tolist()]
        elif self.geometry.n_ports == 2:
            tr = 0.5 if target_ratio is None else float(target_ratio)
            t_list = [[tr, 1.0 - tr]]
        else:
            n = self.geometry.n_ports
            t_list = [[1.0 / n] * n]
        return torch.tensor(t_list, dtype=torch.float32, device=self.device)

    def fit(
        self,
        target_ratio: Optional[float] = None,
        target_ratios: Optional[List[float]] = None,
        total_target: float = 0.96,
        steps: int = 250,
        lr: float = 0.03,
        init_mode: str = "filled",
        symmetric: Optional[bool] = None,
        robust_delta: float = 0.0,
        robust_start: float = 0.5,
        bias_k: float = 0.015,
        bias_k_avg: float = 8.0,
        bias_p: int = 3,
    ):
        wl_targets = self._resolve_targets(target_ratio, target_ratios)
        t0_np = wl_targets[0].cpu().numpy()
        if symmetric is None:
            symmetric = (not self.device_type.startswith("wdm")) and np.allclose(t0_np, t0_np[::-1], atol=1e-3)

        z = self.geometry.init_latent(init_mode=init_mode).requires_grad_(True)

        stage1_steps = int(self.stage1_frac * steps)
        stage2_steps = max(steps - stage1_steps, 1)

        optimizer = torch.optim.Adam([z], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.25)

        history = {"loss": [], "powers": [], "powers_all": [], "total": [], "t1": [], "t2": []}
        pbar = tqdm(range(1, steps + 1), desc=f"Fitting [{self.device_type}]")

        wl_mean = float(np.mean(self.wls))
        stage1_steps = int(self.stage1_frac * steps)
        t0 = time.perf_counter()

        for step in pbar:
            discrete_stage = (step > stage1_steps)

            optimizer.zero_grad()
            use_robust = robust_delta > 0 and step > robust_start * steps
            shifts = (0.0, robust_delta, -robust_delta) if use_robust else (0.0,)

            loss = 0.0
            nom_powers = []
            for shift in shifts:
                rho_s = self._project_density(
                    z, symmetric=symmetric, shift=shift, discrete_stage=(discrete_stage and shift == 0.0)
                )
                eps_in = self._assemble_eps_norm(rho_s)
                pass_powers = []
                for s_idx, solver in enumerate(self.solvers):
                    field = solve_fdfd_differentiable(eps_in, solver)
                    l, p_vec = self.loss_fn(
                        field,
                        rho_s,
                        solver,
                        targets=wl_targets[s_idx],
                        total_target=total_target,
                        bin_weight=0.0,  # Discretization is handled by Neighbor Biasing, NOT self-biasing loss!
                    )
                    wl_weight = (solver.wl / wl_mean) ** 2 if len(self.solvers) > 1 else 1.0
                    loss = loss + (wl_weight * l) / (len(shifts) * len(self.solvers))
                    pass_powers.append((p_vec * wl_targets[s_idx]).sum())
                    if shift == 0.0:
                        nom_powers.append(p_vec.detach().cpu().numpy())

                if len(pass_powers) > 1:
                    p_stack = torch.stack(pass_powers)
                    loss = loss + 1.5 * p_stack.var(unbiased=False) / len(shifts)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Apply Neighbor Biasing update (Eqs. 2-6 of Su et al.):
            # Gently ramp k during Stage 1 so waveguides/holes form first and then discretize cleanly
            ramp = min(step / max(stage1_steps, 1), 1.0)
            k_step = bias_k * (0.25 + 0.75 * (ramp ** 1.5))
            self._apply_neighbor_bias(z, k=k_step, k_avg=bias_k_avg, p=bias_p)

            p0 = nom_powers[0]
            history["loss"].append(loss.item())
            history["powers"].append(p0.tolist())
            history["powers_all"].append([p.tolist() for p in nom_powers])
            history["t1"].append(float(p0[0]))
            history["t2"].append(float(p0[1]))

            if len(nom_powers) > 1:
                tot_v = float(np.mean([np.sum(p) for p in nom_powers]))
                ports_str = "|".join(f"λ{i+1}:{nom_powers[i][i]*100:.1f}%" for i in range(len(nom_powers)))
            else:
                tot_v = float(np.sum(p0))
                ports_str = "/".join(f"{p*100:.1f}%" for p in p0)

            history["total"].append(tot_v)
            stage_label = "Disc" if discrete_stage else "Cont+NB"
            pbar.set_postfix(Ports=ports_str, Tot=f"{tot_v*100:.1f}%", Stage=stage_label)

        elapsed = time.perf_counter() - t0

        with torch.no_grad():
            rho_final = self._project_density(z, symmetric=symmetric, discrete_stage=True)
            rho_bin_t = (rho_final > 0.5).float()
            rho_bin_np = rho_bin_t.cpu().numpy()
            eps_norm_np = self._assemble_eps_norm(rho_bin_t).squeeze().cpu().numpy()

        Ez_list, powers_list = [], []
        for s in self.solvers:
            Ez_b, p_b = self.evaluate(rho_bin_np, solver=s)
            Ez_list.append(Ez_b)
            powers_list.append(p_b)

        p_main = powers_list[0]
        if len(powers_list) > 1:
            msg = " | ".join(
                f"λ={w:.2f}um: {[round(x*100, 1) for x in p]}% (train {[round(x*100, 1) for x in g]}%)"
                for w, p, g in zip(self.wls, powers_list, nom_powers)
            )
            print(f" [{self.device_type} Fit in {elapsed:.1f}s] {msg}")
        else:
            print(
                f" [{self.device_type} Fit in {elapsed:.1f}s] Binary: {[round(x*100, 1) for x in p_main]}% "
                f"(Train: {[round(x*100, 1) for x in nom_powers[0]]}%)"
            )

        return {
            "device_type": self.device_type,
            "rho": rho_bin_np,
            "eps_norm": eps_norm_np,
            "Ez_fit": Ez_list[0],
            "Ez_all": Ez_list,
            "powers_fit": p_main,
            "powers_all": powers_list,
            "t1_fit": p_main[0],
            "t2_fit": p_main[1],
            "z": z.detach().cpu(),
            "history": history,
            "target_ratios": t0_np.tolist(),
            "target_ratio": float(t0_np[0]),
            "elapsed_sec": elapsed,
        }