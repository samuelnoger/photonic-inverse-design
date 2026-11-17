import time
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, distance_transform_edt, gaussian_filter
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
    Photonic Inverse Design Fitter supporting both:
      - "beta": bounded-sigmoid -> Gaussian filter -> tanh(beta) continuation
      - "nb": Two-Stage Neighbor Biasing + Discrete Level-Set (following Su et al.)
    All default parameters live in `solver/config.py` and can be overridden in `__init__` or `fit()`.
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

        # Soft pre-filter kernel (radius ~ 1.1 px) to smooth pixel-grid staircasing on gradients
        g_rad = 2
        self.grad_kernel, self.g_pad = self._make_gauss_kernel(sigma=1.1, radius=g_rad)
        self.g_pad = g_rad

        # Gaussian length-scale filter for the "beta" method
        k_rad = int(np.ceil(2.5 * self.smooth_sigma))
        self.gauss_kernel, self.gauss_pad = self._make_gauss_kernel(
                    sigma=self.smooth_sigma, radius=int(np.ceil(2.5 * self.smooth_sigma))
                )
        self.gauss_pad = k_rad

        # Port & corner pinning in [0, 1] from self.cfg.flare_margin
        self.pin, self.pin_val = self.geometry.get_pin_masks(
            n_pin=r + 2, flare_margin=self.cfg.flare_margin
        )

        self.drc_rad = int(self.cfg.drc_radius)
        rr = self.drc_rad
        yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]
        self.drc_struct = (xx ** 2 + yy ** 2) <= (rr ** 2 + 0.5)
        self.pin_np = self.pin.cpu().numpy() > 0.5
        self.pin_val_np = self.pin_val.cpu().numpy() > 0.5

        self.stage1_frac = self.cfg.stage1_frac

    def _make_gauss_kernel(self, sigma: float, radius: int):
        c = torch.arange(-radius, radius + 1, device=self.device, dtype=torch.float32)
        kx, ky = torch.meshgrid(c, c, indexing="ij")
        k = torch.exp(-0.5 * (kx ** 2 + ky ** 2) / (sigma ** 2))
        return (k / k.sum()).unsqueeze(0).unsqueeze(0), radius

    def evaluate(self, rho_np, solver=None):
        s = solver if solver is not None else self.solver
        Ez = s.solve(s.make_eps(rho_np))
        powers = s.measure_port_powers(Ez)
        return Ez, powers

    @staticmethod
    def _sdf(b):
        b = b.astype(bool)
        return np.where(b, distance_transform_edt(b) - 0.5,
                        0.5 - distance_transform_edt(~b)).astype(np.float32)

    def _drc_clean(self, b):
        pad = self.drc_rad + 2
        bp = np.pad(b.astype(bool), pad, mode="edge")
        bp = binary_closing(bp, structure=self.drc_struct)
        bp = binary_opening(bp, structure=self.drc_struct)
        return np.where(self.pin_np, self.pin_val_np, bp[pad:-pad, pad:-pad])

    def _loss_and_powers(self, rho, wl_targets, total_target):
        eps_in = self._assemble_eps_norm(rho)
        wl_mean = float(np.mean(self.wls))
        loss, powers, pass_p = 0.0, [], []
        for i, solver in enumerate(self.solvers):
            field = solve_fdfd_differentiable(eps_in, solver)
            l, p = self.loss_fn(field, rho, solver, targets=wl_targets[i],
                                total_target=total_target, bin_weight=0.0)
            w = (solver.wl / wl_mean) ** 2 if len(self.solvers) > 1 else 1.0
            loss = loss + w * l / len(self.solvers)
            pass_p.append((p * wl_targets[i]).sum())
            powers.append(p.detach().cpu().numpy())
        if len(pass_p) > 1:
            loss = loss + 1.5 * torch.stack(pass_p).var(unbiased=False)
        return loss, powers

    @staticmethod
    def _log(history, loss, powers):
        p0 = powers[0]
        history["loss"].append(float(loss))
        history["powers"].append(p0.tolist())
        history["powers_all"].append([p.tolist() for p in powers])
        history["t1"].append(float(p0[0]))
        history["t2"].append(float(p0[1]))
        tot = float(np.mean([np.sum(p) for p in powers]))
        history["total"].append(tot)
        return tot

    def _levelset_stage(self, b0, wl_targets, total_target, steps, symmetric=False, history=None):
        c = self.cfg
        b = self._drc_clean(b0)
        phi_t = torch.tensor(self._sdf(b), dtype=torch.float32, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([phi_t], lr=0.15)
        best_b, best_p, best_loss = b.copy(), None, np.inf

        pbar = tqdm(range(1, steps + 1), desc=f"Stage 2 LS [{self.device_type}]")
        for it in pbar:
            opt.zero_grad()
            if symmetric:
                phi_sym = 0.5 * (phi_t + torch.flip(phi_t, dims=[-1]))
            else:
                phi_sym = phi_t

            # Smooth ramp in [-2, 2] for gradients, DRC-cleaned binary in forward pass (STE)
            rho_soft = torch.clamp(0.5 + 0.25 * phi_sym, 0.0, 1.0)
            rho_soft = self.pin * self.pin_val + (1.0 - self.pin) * rho_soft

            b_clean = self._drc_clean((rho_soft.detach().cpu().numpy()) > 0.5)
            rho_bin = torch.from_numpy(b_clean.astype(np.float32)).to(self.device)
            rho_ste = rho_soft + (rho_bin - rho_soft).detach()

            loss, powers = self._loss_and_powers(rho_ste, wl_targets, total_target)

            # Save the exact evaluated design BEFORE opt.step() mutates phi_t
            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_b = b_clean.copy()
                best_p = [p.copy() for p in powers]

            loss.backward()

            # Smooth gradient spatially before Adam update
            with torch.no_grad():
                g_np = gaussian_filter(phi_t.grad.cpu().numpy(), c.ls_vel_sigma)
                phi_t.grad.copy_(torch.from_numpy(g_np).to(self.device))

            opt.step()

            if history is not None:
                tot = self._log(history, loss_val, powers)
                pbar.set_postfix(Tot=f"{tot*100:.1f}%")

        return best_b, best_p, best_loss

    @torch.no_grad()
    def _apply_neighbor_bias(self, z, k=0.012, k_avg=10.0, p=3):
        """
        Implements Su et al. Equations (2)-(6): Neighbor Biasing in [0, 1].
          z_avg = uniform circular disk average of z within radius r
          h(z)  = 0.5 + k_avg * (0.5 - z_avg)^(2p + 1)
          b_n(z)= (z - h(z)) / (1 - 2k) + 0.5
          z    <- clip(b_n(z), 0, 1)
        """
        r = self.disk_rad
        z_4d = F.pad(z.unsqueeze(0).unsqueeze(0), (r,) * 4, mode="replicate")
        z_avg = F.conv2d(z_4d, self.disk_kernel).squeeze()

        h_z = 0.5 + k_avg * ((0.5 - z_avg) ** (2 * p + 1))
        z_biased = (z - h_z) / (1.0 - 2.0 * k) + 0.5
        z_clipped = torch.clamp(z_biased, 0.0, 1.0)
        z_pinned = self.pin * self.pin_val + (1.0 - self.pin) * z_clipped
        z.copy_(z_pinned)

    def _project_density(self, z, symmetric=False, shift=0.0, beta=1.0):
        if symmetric:
            z = 0.5 * (z + torch.flip(z, dims=[-1]))
        z_p = self.pin * self.pin_val + (1.0 - self.pin) * torch.clamp(z, 0.0, 1.0)

        # Lightly smooth pixel-level staircasing so boundary gradients move smooth level-set contours
        z_4d = F.pad(z_p.unsqueeze(0).unsqueeze(0), (self.g_pad,) * 4, mode="replicate")
        rho = F.conv2d(z_4d, self.grad_kernel).squeeze()

        if shift != 0.0:
            eta_shift = shift * 0.15 if abs(shift) > 0.12 else shift
            rho = torch.clamp(rho + eta_shift * (1.0 - self.pin), 0.0, 1.0)

        if beta > 1.0:
            rho = (np.tanh(0.5 * beta) + torch.tanh(beta * (rho - 0.5))) / (2.0 * np.tanh(0.5 * beta))

        rho = self.pin * self.pin_val + (1.0 - self.pin) * rho
        return rho

    def _project_density_beta(self, z, beta, symmetric=False, shift=0.0):
        if symmetric:
            z = 0.5 * (z + torch.flip(z, dims=[-1]))
        raw = self.pin * self.pin_val + (1.0 - self.pin) * torch.sigmoid(z)   # bounded in [0,1]
        x = F.pad(raw.unsqueeze(0).unsqueeze(0), (self.gauss_pad,) * 4, mode="replicate")
        rf = F.conv2d(x, self.gauss_kernel).squeeze()                          # length-scale filter
        eta = 0.5 - shift * (1.0 - self.pin)                                   # shift>0 dilates
        b = torch.as_tensor(beta, dtype=rf.dtype, device=rf.device)
        num = torch.tanh(b * eta) + torch.tanh(b * (rf - eta))
        den = torch.tanh(b * eta) + torch.tanh(b * (1.0 - eta))
        return self.pin * self.pin_val + (1.0 - self.pin) * (num / den)

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

    def fit(self, *args, method=None, **kwargs):
        method = method or self.cfg.method
        if method == "nb":
            return self._fit_nb(*args, **kwargs)
        return self._fit_beta(*args, **kwargs)

    def _fit_beta(self, target_ratio=None, target_ratios=None, total_target=None, steps=None,
                  lr=None, init_mode=None, symmetric=None, robust_delta=None, robust_start=None,
                  beta_start=1.5, beta_end=32.0):
        c = self.cfg
        total_target = c.total_target if total_target is None else total_target
        steps = c.steps if steps is None else steps
        lr = 0.05 if lr is None else lr
        init_mode = c.init_mode if init_mode is None else init_mode
        robust_delta = c.robust_delta if robust_delta is None else robust_delta
        robust_start = c.robust_start if robust_start is None else robust_start

        wl_targets = self._resolve_targets(target_ratio, target_ratios)
        t0_np = wl_targets[0].cpu().numpy()
        if symmetric is None:
            symmetric = (not self.device_type.startswith("wdm")) and np.allclose(t0_np, t0_np[::-1], atol=1e-3)

        z = torch.logit(self.geometry.init_latent(init_mode=init_mode).clamp(0.05, 0.95)).requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
        history = {"loss": [], "powers": [], "powers_all": [], "total": [], "t1": [], "t2": [], "gap": []}

        t0 = time.perf_counter()
        gap = float("nan")
        pbar = tqdm(range(1, steps + 1), desc=f"Fitting beta [{self.device_type}]")
        for step in pbar:
            frac = min((step - 1) / max(0.7 * steps, 1), 1.0)               # reach beta_end at 70%, then hold
            beta = beta_start * (beta_end / beta_start) ** frac
            use_robust = robust_delta > 0 and step > robust_start * steps
            shifts = (0.0, robust_delta, -robust_delta) if use_robust else (0.0,)

            opt.zero_grad()
            loss, nom = 0.0, None
            for shift in shifts:
                rho_s = self._project_density_beta(z, beta, symmetric, shift)
                l, pw = self._loss_and_powers(rho_s, wl_targets, total_target)
                loss = loss + l / len(shifts)
                if shift == 0.0:
                    nom = pw
            loss.backward()

            if step % 10 == 0 or step == steps:                              # gray-vs-binary gap
                with torch.no_grad():
                    rb = (self._project_density_beta(z, beta, symmetric) > 0.5).float()
                    _, pb = self._loss_and_powers(rb, wl_targets, total_target)
                    gap = max(float(np.abs(a - b).max()) for a, b in zip(nom, pb))
            opt.step()
            sched.step()

            tot = self._log(history, loss.item(), nom)
            history["gap"].append(gap)
            pbar.set_postfix(Tot=f"{tot*100:.1f}%", beta=f"{beta:.0f}", gap=f"{gap*100:.1f}%")

        elapsed = time.perf_counter() - t0
        with torch.no_grad():
            rho_bin = (self._project_density_beta(z, beta_end, symmetric) > 0.5).float()
        rho_bin_np = rho_bin.cpu().numpy()
        eps_norm_np = self._assemble_eps_norm(rho_bin).squeeze().cpu().numpy()

        Ez_list, powers_list = [], []
        for s in self.solvers:
            Ez_b, p_b = self.evaluate(rho_bin_np, solver=s)
            Ez_list.append(Ez_b)
            powers_list.append(p_b)
        p_main = powers_list[0]
        print(f" [{self.device_type} Fit in {elapsed:.1f}s] Final binary: "
              f"{[[round(x*100,1) for x in p] for p in powers_list]}%")
        return {
            "device_type": self.device_type, "rho": rho_bin_np, "eps_norm": eps_norm_np,
            "Ez_fit": Ez_list[0], "Ez_all": Ez_list, "powers_fit": p_main, "powers_all": powers_list,
            "t1_fit": p_main[0], "t2_fit": p_main[1], "z": z.detach().cpu(), "history": history,
            "target_ratios": t0_np.tolist(), "target_ratio": float(t0_np[0]), "elapsed_sec": elapsed,
        }

    def _fit_nb(self, 
                target_ratio=None, 
                target_ratios=None, 
                total_target=None, 
                steps=None, 
                lr=None,
                init_mode=None, 
                symmetric=None, 
                robust_delta=None, 
                robust_start=None,
                bias_k=None, 
                bias_k_avg=None, 
                bias_p=None):
        
        c = self.cfg
        
        total_target = c.total_target if total_target is None else total_target
        steps = c.steps if steps is None else steps
        lr = c.lr if lr is None else lr
        init_mode = c.init_mode if init_mode is None else init_mode
        robust_delta = c.robust_delta if robust_delta is None else robust_delta
        robust_start = c.robust_start if robust_start is None else robust_start
        bias_k = c.bias_k if bias_k is None else bias_k
        bias_k_avg = c.bias_k_avg if bias_k_avg is None else bias_k_avg
        bias_p = c.bias_p if bias_p is None else bias_p

        wl_targets = self._resolve_targets(target_ratio, target_ratios)
        t0_np = wl_targets[0].cpu().numpy()
        if symmetric is None:
            symmetric = (not self.device_type.startswith("wdm")) and np.allclose(t0_np, t0_np[::-1], atol=1e-3)

        stage1_steps = max(int(self.stage1_frac * steps), 1)
        stage2_steps = max(steps - stage1_steps, 1)

        z = self.geometry.init_latent(init_mode=init_mode).requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_steps, eta_min=lr * 0.1)
        history = {"loss": [], "powers": [], "powers_all": [], "total": [], "t1": [], "t2": [],
                   "gray_z": [], "gap": []}

        t0 = time.perf_counter()
        gap = float("nan")
        pbar = tqdm(range(1, stage1_steps + 1), desc=f"Stage 1 NB [{self.device_type}]")
        for step in pbar:
            optimizer.zero_grad()
            use_robust = robust_delta > 0 and step > robust_start * stage1_steps
            shifts = (0.0, robust_delta, -robust_delta) if use_robust else (0.0,)
            loss, nom_powers = 0.0, None
            for shift in shifts:
                beta_nb = 1.0 if step < 0.5 * stage1_steps else 1.0 * (16.0 ** ((step - 0.5 * stage1_steps) / (0.5 * stage1_steps)))
                rho_s = self._project_density(z, symmetric=symmetric, shift=shift, beta=beta_nb)
                l, pw = self._loss_and_powers(rho_s, wl_targets, total_target)
                loss = loss + l / len(shifts)
                if shift == 0.0:
                    nom_powers = pw
            loss.backward()

            # --- binarization diagnostics (before z changes) ---
            with torch.no_grad():
                gray_z = float(((z > 0.05) & (z < 0.95)).float().mean())
                if step % 10 == 0 or step == stage1_steps:
                    rb = (self._project_density(z, symmetric=symmetric) > 0.5).float()
                    _, pb = self._loss_and_powers(rb, wl_targets, total_target)
                    gap = max(float(np.abs(a - b).max()) for a, b in zip(nom_powers, pb))

            optimizer.step()
            scheduler.step()
            self._apply_neighbor_bias(z, k=bias_k, k_avg=bias_k_avg, p=bias_p)

            tot = self._log(history, loss.item(), nom_powers)
            history["gray_z"].append(gray_z)
            history["gap"].append(gap)
            pbar.set_postfix(Tot=f"{tot*100:.1f}%", grayZ=f"{gray_z*100:.0f}%", gap=f"{gap*100:.1f}%")

        # ---- threshold, then Stage 2: level set with design rules ----
        with torch.no_grad():
            b0 = (self._project_density(z, symmetric=symmetric) > 0.5).cpu().numpy()
            _, p_thr = self._loss_and_powers(torch.from_numpy(b0.astype(np.float32)).to(self.device),
                                             wl_targets, total_target)
        print(f" Stage-1 thresholded powers: {[[round(x*100,1) for x in p] for p in p_thr]}%")
        best_b, best_p, best_loss = self._levelset_stage(
            b0, wl_targets, total_target, steps=stage2_steps, symmetric=symmetric, history=history)
        self._log(history, best_loss, best_p)          # last curve point = the final design
        elapsed = time.perf_counter() - t0

        rho_bin_np = best_b.astype(np.float32)
        eps_norm_np = self._assemble_eps_norm(torch.from_numpy(rho_bin_np).to(self.device)).squeeze().cpu().numpy()
        Ez_list, powers_list = [], []
        for s in self.solvers:
            Ez_b, p_b = self.evaluate(rho_bin_np, solver=s)
            Ez_list.append(Ez_b)
            powers_list.append(p_b)
        p_main = powers_list[0]
        print(f" [{self.device_type} Fit in {elapsed:.1f}s] Final binary: "
              f"{[[round(x*100,1) for x in p] for p in powers_list]}%")

        return {
            "device_type": self.device_type, "rho": rho_bin_np, "eps_norm": eps_norm_np,
            "Ez_fit": Ez_list[0], "Ez_all": Ez_list,
            "powers_fit": p_main, "powers_all": powers_list,
            "t1_fit": p_main[0], "t2_fit": p_main[1],
            "z": z.detach().cpu(), "history": history,
            "target_ratios": t0_np.tolist(), "target_ratio": float(t0_np[0]),
            "elapsed_sec": elapsed,
        }