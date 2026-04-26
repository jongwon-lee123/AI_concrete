import argparse
import csv
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================
# User-configurable defaults (override via CLI flags)
# ============================================================
@dataclass
class Config:
    # Paths
    data_dir: str = r"C:\Users\whddn\Desktop\AI\AI_concrete"
    save_dir: str = r"C:\Users\whddn\Desktop\AI\AI_concrete\outputs"

    # Dataset / patch sampling
    patch_size: int = 256
    ignore_bottom_px: int = 60
    hflip: bool = True
    vflip: bool = False
    num_workers: int = 0  # Windows-friendly default
    dataset_size: int = 20000  # virtual length for random patch sampling

    # Training
    seed: int = 42
    batch_size: int = 8
    max_steps: int = 20000
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_interval: int = 50

    # Diffusion
    timesteps: int = 1000
    beta_schedule: str = "cosine"  # ["linear", "cosine"]

    # UNet (memory-aware)
    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2)  # minimum 2-level down/up
    time_emb_dim: int = 256

    # Sampling / checkpoint
    num_samples: int = 16
    sample_interval: int = 1000
    ckpt_interval: int = 2000

    # Optional debug/eval
    save_real_debug_grid: bool = True
    eval_interval: int = 1000
    pore_threshold_mode: str = "otsu"  # ["otsu", "fixed"]
    pore_threshold_fixed: int = 85

    # Mixed precision
    amp: bool = True


ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train DDPM for grayscale SEM-BSE patch generation")
    defaults = Config()
    for k, v in asdict(defaults).items():
        arg_name = f"--{k}"
        if isinstance(v, bool):
            parser.add_argument(arg_name, type=str, default=str(v))
        elif isinstance(v, tuple):
            parser.add_argument(arg_name, type=str, default=",".join(map(str, v)))
        else:
            parser.add_argument(arg_name, type=type(v), default=v)

    args = parser.parse_args()
    cfg = Config()
    for k in asdict(cfg).keys():
        val = getattr(args, k)
        if isinstance(getattr(cfg, k), bool):
            val = str(val).lower() in {"1", "true", "yes", "y"}
        elif isinstance(getattr(cfg, k), tuple):
            val = tuple(int(x.strip()) for x in str(val).split(",") if x.strip())
        setattr(cfg, k, val)

    if cfg.beta_schedule not in {"linear", "cosine"}:
        raise ValueError("beta_schedule must be one of [linear, cosine]")
    if cfg.pore_threshold_mode not in {"otsu", "fixed"}:
        raise ValueError("pore_threshold_mode must be one of [otsu, fixed]")
    if cfg.num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    return cfg


class BSEPatchDataset(Dataset):
    """
    Loads many SEM-BSE images, converts to grayscale (L), removes bottom overlay region,
    and returns random patches without downscaling the full image.
    """

    def __init__(
        self,
        data_dir: str,
        patch_size: int,
        ignore_bottom_px: int,
        dataset_size: int,
        hflip: bool = True,
        vflip: bool = False,
    ):
        self.patch_size = patch_size
        self.ignore_bottom_px = max(0, ignore_bottom_px)
        self.dataset_size = dataset_size
        self.hflip = hflip
        self.vflip = vflip

        root = Path(data_dir)
        if not root.exists():
            raise FileNotFoundError(f"data_dir not found: {data_dir}")

        self.files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
        if not self.files:
            raise RuntimeError(f"No image files found under {data_dir}")

    def __len__(self) -> int:
        return self.dataset_size

    def _safe_crop_patch(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        valid_h = max(1, h - self.ignore_bottom_px)

        # Apply valid region first (strip bottom overlay)
        img = img.crop((0, 0, w, valid_h))
        w, h = img.size

        if w < self.patch_size or h < self.patch_size:
            # Fallback: upsize only as needed to safely crop patch
            scale = max(self.patch_size / max(1, w), self.patch_size / max(1, h))
            new_w = max(self.patch_size, int(round(w * scale)))
            new_h = max(self.patch_size, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.BICUBIC)
            w, h = img.size

        x0 = random.randint(0, w - self.patch_size)
        y0 = random.randint(0, h - self.patch_size)
        return img.crop((x0, y0, x0 + self.patch_size, y0 + self.patch_size))

    def __getitem__(self, idx: int) -> torch.Tensor:
        # random file choice decoupled from idx to improve variation
        path = random.choice(self.files)
        with Image.open(path) as im:
            im = im.convert("L")
            patch = self._safe_crop_patch(im)

        if self.hflip and random.random() < 0.5:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if self.vflip and random.random() < 0.5:
            patch = patch.transpose(Image.FLIP_TOP_BOTTOM)

        arr = np.asarray(patch, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0  # [-1, 1]
        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
        return tensor


def get_timestep_embedding(timesteps: torch.Tensor, emb_dim: int) -> torch.Tensor:
    half = emb_dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_dim)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.res(x, t_emb)
        d = self.down(h)
        return d, h


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.res = ResBlock(out_ch + skip_ch, out_ch, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb)


class UNetEps(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_ch: int = 64, channel_mults: Tuple[int, ...] = (1, 2), time_dim: int = 256):
        super().__init__()
        assert len(channel_mults) >= 2, "Use at least 2 levels for down/up path"
        chs = [base_ch * m for m in channel_mults]

        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.downs.append(DownBlock(chs[i], chs[i + 1], time_dim))

        mid_ch = chs[-1]
        self.mid1 = ResBlock(mid_ch, mid_ch, time_dim)
        self.mid2 = ResBlock(mid_ch, mid_ch, time_dim)

        self.ups = nn.ModuleList()
        skip_chs = list(reversed(chs[1:]))
        out_chs = list(reversed(chs[:-1]))
        in_ch_cur = chs[-1]
        for skip_ch, out_ch_i in zip(skip_chs, out_chs):
            self.ups.append(UpBlock(in_ch_cur, skip_ch, out_ch_i, time_dim))
            in_ch_cur = out_ch_i

        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_conv = nn.Conv2d(chs[0], out_ch, 3, padding=1)
        self.time_dim = time_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = get_timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x = self.in_conv(x)
        skips = []
        h = x
        for down in self.downs:
            h, skip = down(h, t_emb)
            skips.append(skip)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for up in self.ups:
            skip = skips.pop()
            h = up(h, skip, t_emb)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, schedule: str = "cosine", device: str = "cpu"):
        self.timesteps = timesteps
        self.device = torch.device(device)

        if schedule == "linear":
            betas = torch.linspace(1e-4, 2e-2, timesteps, dtype=torch.float32)
        elif schedule == "cosine":
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError("schedule must be 'linear' or 'cosine'")

        self.betas = betas.to(self.device)
        self.alphas = (1.0 - self.betas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]], dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas.float(), 1e-5, 0.999)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_omb * noise

    def p_sample_step(self, model: nn.Module, xt: torch.Tensor, t_scalar: int) -> torch.Tensor:
        b = xt.size(0)
        t = torch.full((b,), t_scalar, device=xt.device, dtype=torch.long)

        eps_pred = model(xt, t)

        a_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_a_bar_t = torch.sqrt(a_bar_t)
        sqrt_one_minus_a_bar_t = torch.sqrt(1.0 - a_bar_t)

        # x0_pred = (xt - sqrt(1-a_bar)*eps_pred) / sqrt(a_bar)
        x0_pred = (xt - sqrt_one_minus_a_bar_t * eps_pred) / torch.clamp(sqrt_a_bar_t, min=1e-8)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1, 1)
        posterior_mean = coef1 * x0_pred + coef2 * xt

        if t_scalar == 0:
            return posterior_mean

        posterior_var = self.posterior_variance[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(xt)
        return posterior_mean + torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: Tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
        xt = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            xt = self.p_sample_step(model, xt, t)
        return xt


def tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    # img in [-1,1], shape [H,W]
    arr = ((img.clamp(-1, 1) + 1.0) * 127.5).byte().cpu().numpy()
    return arr


def save_grid(images: torch.Tensor, path: Path, nrow: int = 4) -> None:
    # images: [N,1,H,W], values in [-1,1]
    n, _, h, w = images.shape
    nrow = max(1, nrow)
    ncol = int(math.ceil(n / nrow))
    canvas = np.zeros((ncol * h, nrow * w), dtype=np.uint8)

    for i in range(n):
        r = i // nrow
        c = i % nrow
        arr = tensor_to_uint8(images[i, 0])
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = arr

    Image.fromarray(canvas, mode="L").save(path)


def otsu_threshold_uint8(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]

    sigma_b2 = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-12)
    return int(np.argmax(sigma_b2))


def connected_component_sizes(binary: np.ndarray) -> List[int]:
    # binary: True for pore
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    sizes: List[int] = []

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(h):
        for x in range(w):
            if not binary[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            cnt = 0
            while stack:
                cy, cx = stack.pop()
                cnt += 1
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            sizes.append(cnt)

    return sizes


def compute_batch_stats(images: torch.Tensor, threshold_mode: str, threshold_fixed: int) -> Dict[str, float]:
    # images [N,1,H,W] in [-1,1]
    imgs_uint8 = np.stack([tensor_to_uint8(images[i, 0]) for i in range(images.shape[0])], axis=0)

    mean_val = float(imgs_uint8.mean())
    var_val = float(imgs_uint8.var())

    pore_fracs = []
    comp_sizes_all = []
    for img in imgs_uint8:
        thr = otsu_threshold_uint8(img) if threshold_mode == "otsu" else threshold_fixed
        pore = img <= thr
        pore_fracs.append(float(pore.mean()))
        comp_sizes_all.extend(connected_component_sizes(pore))

    if comp_sizes_all:
        bins = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 10**9]
        hist, _ = np.histogram(comp_sizes_all, bins=bins)
        psd_entropy = float(-(hist / max(hist.sum(), 1e-12) * np.log((hist / max(hist.sum(), 1e-12)) + 1e-12)).sum())
        mean_comp = float(np.mean(comp_sizes_all))
    else:
        psd_entropy = 0.0
        mean_comp = 0.0

    return {
        "mean_gray": mean_val,
        "var_gray": var_val,
        "phi_pore": float(np.mean(pore_fracs)) if pore_fracs else 0.0,
        "psd_mean_component_size": mean_comp,
        "psd_entropy": psd_entropy,
    }


def append_eval_csv(csv_path: Path, row: Dict[str, float]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def train(cfg: Config) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = BSEPatchDataset(
        data_dir=cfg.data_dir,
        patch_size=cfg.patch_size,
        ignore_bottom_px=cfg.ignore_bottom_px,
        dataset_size=cfg.dataset_size,
        hflip=cfg.hflip,
        vflip=cfg.vflip,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model = UNetEps(
        in_ch=1,
        out_ch=1,
        base_ch=cfg.base_channels,
        channel_mults=cfg.channel_mults,
        time_dim=cfg.time_emb_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    diffusion = GaussianDiffusion(timesteps=cfg.timesteps, schedule=cfg.beta_schedule, device=str(device))

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    model.train()
    data_iter = iter(loader)

    # Optional debug output: real patches before training
    if cfg.save_real_debug_grid:
        debug_batch = next(data_iter)
        save_grid(debug_batch[: min(16, debug_batch.size(0))], save_dir / "real_patches_debug.png", nrow=4)

    start = time.time()
    for step in range(1, cfg.max_steps + 1):
        try:
            x0 = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x0 = next(data_iter)

        x0 = x0.to(device)
        b = x0.size(0)
        t = torch.randint(0, cfg.timesteps, (b,), device=device).long()
        noise = torch.randn_like(x0)
        xt = diffusion.q_sample(x0, t, noise=noise)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)

        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % cfg.log_interval == 0 or step == 1:
            elapsed = time.time() - start
            print(f"step={step:6d} | loss={loss.item():.6f} | elapsed={elapsed:.1f}s")

        if step % cfg.sample_interval == 0:
            model.eval()
            with torch.no_grad():
                sampled = diffusion.sample(model, (cfg.num_samples, 1, cfg.patch_size, cfg.patch_size), device=device)
            save_grid(sampled, save_dir / f"samples_step_{step}.png", nrow=int(math.sqrt(cfg.num_samples)))
            model.train()

        if step % cfg.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                sampled = diffusion.sample(model, (cfg.num_samples, 1, cfg.patch_size, cfg.patch_size), device=device)
                real_for_eval = x0[: cfg.num_samples]

            gen_stats = compute_batch_stats(sampled, cfg.pore_threshold_mode, cfg.pore_threshold_fixed)
            real_stats = compute_batch_stats(real_for_eval, cfg.pore_threshold_mode, cfg.pore_threshold_fixed)
            row = {"step": step}
            row.update({f"gen_{k}": v for k, v in gen_stats.items()})
            row.update({f"real_{k}": v for k, v in real_stats.items()})
            append_eval_csv(save_dir / "eval_stats.csv", row)
            model.train()

        if step % cfg.ckpt_interval == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": asdict(cfg),
            }
            torch.save(ckpt, save_dir / f"ddpm_ckpt_step_{step}.pt")

    print("Training finished.")


def main() -> None:
    cfg = parse_args()
    print("==== train_ddpm_bse.py config ====")
    for k, v in asdict(cfg).items():
        print(f"{k}: {v}")
    print("==================================")
    train(cfg)


if __name__ == "__main__":
    main()
