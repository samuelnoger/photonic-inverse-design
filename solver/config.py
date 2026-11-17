from dataclasses import dataclass, field, replace
from typing import List, Optional


@dataclass
class PhotonicConfig:
    device_type: str = "splitter_1x2"

    # Physics & Waveguide Geometry
    dx: float = 0.02
    wl: float = 1.55
    wls: List[float] = field(default_factory=lambda: [1.55])
    npml: int = 32
    pml_sigma_max: float = 8.0
    eps_bg: float = 2.08
    eps_si: float = 7.84

    wg_half: int = 10
    port_offset: int = 36
    design_len_x: int = 120
    design_half_y: int = 48
    x_src: int = 60
    x_in_end: int = 88
    src_amp: float = 100.0

    # Auto-computed grid dimensions (computed in __post_init__ if left as None)
    x_out_start: Optional[int] = None
    x_meas: Optional[int] = None
    nx: Optional[int] = None
    ny: Optional[int] = None

    # Optimization & Neighbor Biasing Defaults
    smooth_sigma: float = 3.0
    flare_margin: int = 6
    init_mode: str = "branched"
    steps: int = 200
    lr: float = 0.03
    total_target: float = 0.96
    robust_delta: float = 0.0
    robust_start: float = 0.5
    bias_k: float = 0.012
    bias_k_avg: float = 12.0
    bias_p: int = 3
    stage1_frac: float = 0.60
    device: str = "cpu"
    drc_radius: int = 2        
    ls_step_px: float = 0.5        
    ls_clean_every: int = 10        
    ls_vel_sigma: float = 1.5  
    
    method: str = "beta"      # "beta" (simple) or "nb" (neighbor biasing + level set)

    def __post_init__(self):
        if self.x_out_start is None:
            self.x_out_start = self.x_in_end + self.design_len_x
        if self.x_meas is None:
            self.x_meas = self.x_out_start + 32
        if self.nx is None:
            self.nx = self.x_meas + 24 + self.npml
        if self.ny is None:
            self.ny = 2 * (self.design_half_y + 48 + self.npml)


# Device Presets
DEVICE_PRESETS = {
    "splitter_1x2": dict(design_len_x=120, design_half_y=48, port_offset=36, wls=[1.55], init_mode="branched", smooth_sigma=3.5, flare_margin=6, steps=180),
    "splitter_1x3": dict(design_len_x=150, design_half_y=72, port_offset=50, wls=[1.55], init_mode="branched", smooth_sigma=3.5, flare_margin=6, steps=160),
    "wdm_1x2":      dict(design_len_x=200, design_half_y=65, port_offset=35, wls=[1.31, 1.55], init_mode="filled", smooth_sigma=3.0, flare_margin=500, steps=250, method="nb"),
    "wdm_1x3":      dict(design_len_x=260, design_half_y=105, port_offset=60, wls=[1.50, 1.54, 1.58], init_mode="filled", smooth_sigma=2.0, flare_margin=500, steps=300, method="nb"),
}


def get_config(device_type: str = "splitter_1x2", **overrides) -> PhotonicConfig:
    """Builds a PhotonicConfig for `device_type` and applies any keyword overrides."""
    if device_type not in DEVICE_PRESETS:
        raise ValueError(f"Unknown device_type '{device_type}'. Choose from {list(DEVICE_PRESETS.keys())}")
    params = {"device_type": device_type, **DEVICE_PRESETS[device_type], **overrides}
    return PhotonicConfig(**params)


# Backwards-compatible alias so solver.py / geometry.py don't even need to change imports!
def get_args(device_type: str = "splitter_1x2", custom_args=None, **overrides) -> PhotonicConfig:
    return get_config(device_type=device_type, **overrides)