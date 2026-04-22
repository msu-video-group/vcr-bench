"""BMTC (Background Mixing Transfer-based) black-box video attack.

Ported from Attacks/BMTC_TransferAttackVid/Videos/train.py + attack.py.

Offline preparation (prepare_offline):
  1. Creates TSM-surrogate model.
  2. Extracts one background frame per selected video via GrabCut.
  3. Collects one frame per class (up to 400 classes) from the full dataset.
  4. Trains the Mixer network (policy-gradient RL over adversarial background fusion).
  5. Saves mixer checkpoint + background_images.npy to save_root.

Attack (attack_query):
  1. Loads pre-trained mixer + background frames from save_root.
  2. Applies adversarial background fusion via the mixer.
  3. Optionally refines with BTG (gradient-based temporal consistency).
  4. Returns adversarial video tensor.
"""
from __future__ import annotations

import gc
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from vcr_bench.attacks.query_base import BaseQueryVideoAttack
from vcr_bench.attacks.utils import flatten_temporal_video, restore_temporal_video
from vcr_bench.models.base import BaseVideoClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Surrogate adapters
# ---------------------------------------------------------------------------

class _ThwcSurrogate:
    """Wrap BaseVideoClassifier: (T, H, W, C) float → (1, num_classes)."""

    def __init__(self, model: BaseVideoClassifier) -> None:
        self._model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        inp = x.unsqueeze(0) if x.ndim == 4 else x  # (1, T, H, W, C)
        with torch.no_grad():
            s = self._model(inp, input_format="NTHWC", enable_grad=False)
        return s.reshape(1, -1)

    def parameters(self):
        m = getattr(self._model, "model", self._model)
        return m.parameters()


class _TchwSurrogate:
    """Wrap BaseVideoClassifier for BTG: (T, C, H, W) → (1, num_classes) with grad."""

    def __init__(self, model: BaseVideoClassifier) -> None:
        self._model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is (T, C, H, W), possibly requires_grad=True
        inp = x.permute(0, 2, 3, 1).unsqueeze(0)  # (1, T, H, W, C) = NTHWC
        enable_grad = x.requires_grad
        s = self._model(inp, input_format="NTHWC", enable_grad=enable_grad)
        return s.reshape(1, -1)

    def parameters(self):
        m = getattr(self._model, "model", self._model)
        return m.parameters()


# ---------------------------------------------------------------------------
# Vendored Mixer (adapted from BMTC Videos/models/Mixer.py)
# Uses torchvision.models.resnet50 instead of local ResNet.
# ---------------------------------------------------------------------------

class _Mixer(nn.Module):
    """Policy-gradient adversarial background mixer.

    Args:
        epsilon:   L∞ perturbation budget (pixel units matching input range).
        action_dim: number of action categories (== number of classes).
        surrogate_fn: callable (T, H, W, C) → (1, num_classes) scores.
        target_fns:   list of callables with the same signature (for transfer reward).
        gamma:    background mixing weight.
        w1, w2:   reward weights for attack and transfer rewards.
    """

    def __init__(
        self,
        epsilon: float,
        action_dim: int,
        surrogate_fn,
        target_fns: list,
        gamma: float,
        w1: float,
        w2: float,
        device,
    ) -> None:
        super().__init__()
        import torchvision.models as tv_models
        resnet = tv_models.resnet50(weights=None)
        self.epsilon = epsilon
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, action_dim)
        self.surrogate_fn = surrogate_fn
        self.target_fns = target_fns
        self.gamma_value = gamma
        self.w1 = w1
        self.w2 = w2
        self.device = device

    def forward(self, x: torch.Tensor, y_true: torch.Tensor, background_images: torch.Tensor, idx: int):
        """
        Args:
            x: (T, H, W, C) float frame tensor.
            y_true: scalar label tensor.
            background_images: (N_bg, H_bg, W_bg, C) background frame pool.
            idx: training step index (used for transfer reward frequency).
        Returns:
            (loss, x_adv) where x_adv is (T, H, W, C).
        """
        T, H, W, C = x.shape
        # Extract per-frame features using ResNet backbone
        features = self.feature_extractor(x.permute(0, 3, 1, 2))  # (T, 2048, 1, 1)
        features = features.view(features.size(0), -1)             # (T, 2048)
        action_logits = self.fc(features)                          # (T, action_dim)
        action_probs = F.softmax(action_logits, dim=-1)
        m = torch.distributions.Categorical(action_probs)
        actions = m.sample()
        log_probs = m.log_prob(actions)

        # Resize first background image to match frame dimensions
        B_frames = background_images[0][None].repeat(T, 1, 1, 1)           # (T, H_bg, W_bg, C)
        B_frames = F.interpolate(
            B_frames.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)                                               # (T, H, W, C)

        x_adv = self.gamma_value * x + (1 - self.gamma_value) * B_frames
        x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)

        R_attack = self._attack_reward(x, y_true, x_adv)
        R_transfer = self._transfer_reward(x_adv, y_true) if (idx % 5 == 0 and self.target_fns) else 0.0

        if isinstance(R_attack, torch.Tensor):
            R_attack = R_attack.item()
        R_total = self.w1 * R_attack + self.w2 * R_transfer

        loss = -log_probs.sum() * float(R_total)
        return loss, x_adv

    def _attack_reward(self, x: torch.Tensor, y_true: torch.Tensor, x_adv: torch.Tensor) -> float:
        out_orig = self.surrogate_fn(x)        # (1, num_classes)
        out_adv = self.surrogate_fn(x_adv)     # (1, num_classes)
        y = int(y_true.item())
        return float((out_adv[0, y] - out_orig[0, y]).item())

    def _transfer_reward(self, x_adv: torch.Tensor, y_true: torch.Tensor) -> float:
        rewards = []
        y = int(y_true.item())
        for fn in self.target_fns:
            with torch.no_grad():
                out = fn(x_adv)                # (1, num_classes)
            out = out[0]                       # (num_classes,)
            z_y = out[y]
            mask = torch.ones_like(out, dtype=torch.bool)
            mask[y] = False
            z_best = out.masked_fill(~mask, float("-inf")).max()
            rewards.append(float((z_best - z_y).item()))
        return float(np.mean(rewards)) if rewards else 0.0


# ---------------------------------------------------------------------------
# Vendored BTG (adapted from BMTC Videos/BTG_methods.py)
# Operates in TCHW format; uses a surrogate callable that supports gradients.
# ---------------------------------------------------------------------------

class _BTG:
    """Gradient-based temporal consistency refinement.

    Args:
        surrogate_fn: callable (T, C, H, W with requires_grad=True) → (1, num_classes).
        epsilon:  L∞ budget (same units as input).
        steps:    number of gradient update steps.
        gamma:    temporal cosine-similarity regularisation weight.
        device:   torch device for computation.
    """

    def __init__(
        self,
        surrogate_fn,
        epsilon: float,
        steps: int = 10,
        gamma: float = 0.1,
        device: torch.device | None = None,
    ) -> None:
        self.surrogate_fn = surrogate_fn
        self.epsilon = epsilon
        self.steps = steps
        self.gamma = gamma
        self.step_size = epsilon / max(steps, 1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device or torch.device("cpu")

    def __call__(self, video: torch.Tensor, label: int) -> torch.Tensor:
        """Refine adversarial video using gradient steps.

        Args:
            video: (T, C, H, W) float tensor.
            label: true/original class index (we maximise loss to cause misclassification).
        Returns:
            Refined (T, C, H, W) float tensor on CPU.
        """
        dev = self.device

        orig = video.clone().detach().cpu()
        adv = video.clone().detach().cpu()
        T = adv.shape[0]

        label_t = torch.tensor([label], dtype=torch.long, device=dev)

        mistakes = 0
        for step in range(self.steps):
            torch.cuda.empty_cache()
            cur = adv.clone().to(dev).requires_grad_(True)
            pert = (adv - orig).to(dev)

            # Temporal cosine similarity on perturbation directions
            cos_loss = torch.tensor(0.0, device=dev)
            if T > 1:
                dirs = []
                for t in range(T):
                    g = pert[t].reshape(-1)
                    g = g / (g.norm(p=2) + 1e-8)
                    dirs.append(g)
                for t in range(T - 1):
                    cos_loss = cos_loss + (1 - F.cosine_similarity(dirs[t], dirs[t + 1], dim=0))

            outputs = self.surrogate_fn(cur)     # (1, num_classes)
            loss = self.loss_fn(outputs, label_t) + self.gamma * cos_loss
            loss.backward()
            grad = cur.grad.data.clone().cpu()

            if int(outputs.argmax(dim=-1).item()) != label:
                mistakes += 1
                if mistakes >= 5:
                    break

            with torch.no_grad():
                adv = adv + self.step_size * grad.sign()
                delta = torch.clamp(adv - orig, -self.epsilon, self.epsilon)
                adv = torch.clamp(orig + delta, 0.0, 255.0)

            del cur, outputs, grad, loss
            if step % 10 == 0:
                gc.collect()

        return adv


# ---------------------------------------------------------------------------
# Video loading helpers
# ---------------------------------------------------------------------------

def _load_video_frames_nthwc(path: str, max_frames: int | None = None) -> np.ndarray | None:
    """Load video frames as uint8 THWC numpy array."""
    try:
        from vcr_bench.utils.io import decode_video_nthwc
        frames = decode_video_nthwc(str(path))
        if hasattr(frames, "numpy"):
            arr = frames.numpy()
        else:
            arr = np.array(frames)
        if arr.ndim == 5 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4 or arr.shape[-1] not in (1, 3):
            raise ValueError(f"Unexpected decoded video shape for {path}: {tuple(arr.shape)}")
        if max_frames is not None and arr.shape[0] > max_frames:
            arr = arr[:max_frames]
        return arr.astype(np.uint8)
    except Exception as e:
        logger.warning("Failed to decode video %s: %s", path, e)
        return None


def _extract_bg_frame_grabcut(frame_rgb: np.ndarray) -> np.ndarray:
    """Extract background from a single HWC uint8 RGB frame using GrabCut."""
    import cv2
    try:
        h, w = frame_rgb.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
        # cv2.grabCut expects BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.grabCut(frame_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 1, 0).astype(np.uint8)
        bg_frame = frame_bgr * mask2[:, :, np.newaxis]
        mask_inv = ((mask2 == 0).astype(np.uint8) * 255)
        bg_inpaint = cv2.inpaint(bg_frame, mask_inv, 3, cv2.INPAINT_TELEA)
        return cv2.cvtColor(bg_inpaint, cv2.COLOR_BGR2RGB)
    except Exception:
        return frame_rgb


def _resize_frame(frame: np.ndarray, h: int, w: int) -> np.ndarray:
    import cv2
    if frame is None or frame.size == 0:
        raise ValueError("Cannot resize an empty frame")
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)


def _load_background_frame(path: Path, h: int, w: int) -> torch.Tensor | None:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = _resize_frame(img, h, w)
    return torch.from_numpy(img).float()


# ---------------------------------------------------------------------------
# Main attack class
# ---------------------------------------------------------------------------

class BMTCAttack(BaseQueryVideoAttack):
    """BMTC background-mixing transfer-based black-box attack.

    Offline preparation trains a Mixer network using a TSM-surrogate model.
    At attack time, the pre-trained mixer fuses the video with background
    frames to craft the adversarial perturbation, optionally refined by BTG.

    prepare_offline() must be called once before running attack_query().
    The trained mixer checkpoint and background images are saved to save_root.

    Default hyper-parameters match the original implementation.
    """

    attack_name = "bmtc"

    def __init__(
        self,
        eps: float = 16.0,
        alpha: float = 1.0,
        steps: int = 10,          # BTG refinement steps
        num_classes: int = 400,
        gamma: float = 0.8,       # background mixing gamma
        w1: float = 1.0,
        w2: float = 0.3,
        num_epochs: int = 20,     # mixer training epochs
        attacked_frames: int = 16,
        enable_btg: bool = True,
        btg_gamma: float = 0.1,
        save_root: str = "bmtc_artifacts",
    ) -> None:
        super().__init__(eps=eps, alpha=alpha, steps=steps)
        self.num_classes = int(num_classes)
        self.gamma = float(gamma)
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.num_epochs = int(num_epochs)
        self.attacked_frames = int(attacked_frames)
        self.enable_btg = bool(enable_btg)
        self.btg_gamma = float(btg_gamma)
        self.save_root = str(save_root)
        self.verbose = False

        # Lazily loaded artifacts
        self._mixer: _Mixer | None = None
        self._background_images: torch.Tensor | None = None
        self._surrogate: BaseVideoClassifier | None = None

    # ------------------------------------------------------------------
    # Offline preparation
    # ------------------------------------------------------------------

    def prepare(self, *, model, dataset, indices, seed, targeted, pipeline_stage, verbose) -> None:
        del model, dataset, indices, seed, targeted, pipeline_stage
        self.verbose = bool(verbose)

    def supports_offline_prepare(self) -> bool:
        return True

    def prepare_offline(
        self,
        *,
        dataset: object,
        indices: list[int],
        seed: int,
        targeted: bool,
        pipeline_stage: str,
        verbose: bool,
        device: str = "cpu",
        **kwargs: Any,
    ) -> dict:
        """Train Mixer and collect background images; save to save_root.

        Args:
            dataset:  dataset without loading pipeline; dataset[idx] → VideoSampleRef.
            indices:  indices of the attack videos (same as attack run).
        """
        from vcr_bench.models import create_model

        dev = torch.device(device)
        rng = random.Random(seed)
        save_root = Path(kwargs.get("save_root", self.save_root))
        num_epochs = int(kwargs.get("num_epochs", self.num_epochs))
        background_sets = kwargs.get("background_sets")
        bg_dir = save_root / "bg_frames"
        mixer_ckpt_dir = save_root / "mixer_checkpoint"
        for d in (save_root, bg_dir, mixer_ckpt_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ---- Load surrogate model ----
        if verbose:
            print("[BMTC] Loading TSM-surrogate model...")
        surrogate = create_model("tsm_surrogate", device=device)
        surrogate.eval()
        surrogate_fn = _ThwcSurrogate(surrogate)

        # ---- Step 1-4: Extract background frame per attack video ----
        bg_npy_path = save_root / "background_images.npy"
        ckpt_path = mixer_ckpt_dir / "mixer.pth"

        selected_info: list[tuple[str, str, int]] = []  # (path, basename, T)
        bg_files = list(bg_dir.glob("*.png"))
        need_bg_extraction = len(bg_files) < len(indices)

        if need_bg_extraction:
            if verbose:
                print(f"[BMTC] Extracting background frames for {len(indices)} videos...")
            for idx in indices:
                item = dataset[idx]
                path = str(item.path)
                basename = Path(path).stem
                frames = _load_video_frames_nthwc(path)
                if frames is None:
                    continue
                T = frames.shape[0]
                frame_idx = rng.randint(0, T - 1)
                frame = frames[frame_idx]  # (H, W, C) uint8 RGB
                bg_frame = _extract_bg_frame_grabcut(frame)
                bg_path = bg_dir / f"{basename}.png"
                try:
                    import cv2
                    cv2.imwrite(str(bg_path), cv2.cvtColor(bg_frame, cv2.COLOR_RGB2BGR))
                except Exception:
                    from PIL import Image
                    Image.fromarray(bg_frame).save(str(bg_path))
                selected_info.append((path, basename, T))
        else:
            for idx in indices:
                item = dataset[idx]
                path = str(item.path)
                basename = Path(path).stem
                frames = _load_video_frames_nthwc(path, max_frames=1)
                T_probe = frames.shape[0] if frames is not None else 16
                selected_info.append((path, basename, T_probe))

        # ---- Step 5: Collect 400 background frames (one per class) ----
        if bg_npy_path.exists():
            if verbose:
                print("[BMTC] Loading cached background_images.npy...")
            background_images = torch.tensor(np.load(str(bg_npy_path)), dtype=torch.float32).to(dev)
        else:
            bg_list: list[torch.Tensor] = []
            background_sets_dir = Path(str(background_sets)) if background_sets else None
            use_preextracted = bool(background_sets_dir and background_sets_dir.is_dir())

            if use_preextracted:
                if verbose:
                    print(f"[BMTC] Loading background frames from {background_sets_dir}...")
                bg_files = sorted(
                    p for p in background_sets_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
                )
                for path in bg_files[: self.num_classes]:
                    bg_tensor = _load_background_frame(path, 320, 568)
                    if bg_tensor is not None:
                        bg_list.append(bg_tensor)
            else:
                if verbose:
                    print("[BMTC] Collecting 400 class background frames...")
                # Group dataset items by label
                class_to_paths: dict[int, list[str]] = {}
                n = len(dataset)  # type: ignore[arg-type]
                for i in range(n):
                    try:
                        item = dataset[i]
                        lbl = item.label
                        if lbl is not None:
                            class_to_paths.setdefault(int(lbl), []).append(str(item.path))
                    except Exception:
                        continue

                all_classes = list(class_to_paths.keys())
                n_classes = min(len(all_classes), self.num_classes)
                selected_classes = rng.sample(all_classes, n_classes) if len(all_classes) >= n_classes else all_classes
                background_paths = [rng.choice(class_to_paths[c]) for c in selected_classes]

                for bp in background_paths:
                    frames = _load_video_frames_nthwc(bp, max_frames=None)
                    if frames is None or frames.shape[0] == 0:
                        if bg_list:
                            bg_list.append(bg_list[-1])
                        else:
                            bg_list.append(torch.zeros(320, 568, 3, dtype=torch.float32))
                        continue
                    fi = rng.randint(0, frames.shape[0] - 1)
                    frame = frames[fi]  # (H, W, C) uint8
                    frame_resized = _resize_frame(frame, 320, 568)
                    bg_list.append(torch.from_numpy(frame_resized).float())

            while len(bg_list) < self.num_classes:
                bg_list.append(bg_list[-1] if bg_list else torch.zeros(320, 568, 3, dtype=torch.float32))
            bg_list = bg_list[:self.num_classes]

            background_images = torch.stack(bg_list, dim=0).to(dev)  # (400, 320, 568, 3)
            np.save(str(bg_npy_path), background_images.cpu().numpy())
            if verbose:
                print(f"[BMTC] background_images shape: {tuple(background_images.shape)}")

        # ---- Step 6: Train Mixer ----
        if ckpt_path.exists():
            if verbose:
                print(f"[BMTC] Mixer checkpoint already exists at {ckpt_path}, skipping training.")
        else:
            if verbose:
                print(f"[BMTC] Training Mixer for {num_epochs} epochs over {len(selected_info)} videos...")

            mix_model = _Mixer(
                epsilon=self.eps,
                action_dim=self.num_classes,
                surrogate_fn=surrogate_fn,
                target_fns=[],
                gamma=self.gamma,
                w1=self.w1,
                w2=self.w2,
                device=dev,
            ).to(dev).train()

            optimizer = optim.SGD(mix_model.parameters(), lr=0.001, momentum=0.9)
            total_loss = 0.0

            for epoch in range(num_epochs):
                total_loss = 0.0
                torch.cuda.empty_cache()
                gc.collect()
                for idx_i, (path, basename, T) in enumerate(selected_info):
                    frames = _load_video_frames_nthwc(path, max_frames=self.attacked_frames)
                    if frames is None:
                        continue
                    # Resize to match background dimensions (320, 568)
                    resized = np.stack([_resize_frame(f, 320, 568) for f in frames], axis=0)
                    x_tensor = torch.from_numpy(resized).float().to(dev)  # (T, 320, 568, 3)

                    # Get ground-truth label
                    try:
                        item = dataset[indices[idx_i] if idx_i < len(indices) else 0]
                        y_true = torch.tensor(int(item.label), device=dev) if item.label is not None else torch.tensor(0, device=dev)
                    except Exception:
                        y_true = torch.tensor(0, device=dev)

                    loss, _ = mix_model(x_tensor, y_true, background_images, idx_i)
                    optimizer.zero_grad()
                    if isinstance(loss, torch.Tensor):
                        loss.backward()
                        optimizer.step()
                        total_loss += float(loss.item())

                if verbose:
                    print(f"[BMTC] epoch {epoch + 1}/{num_epochs}  loss={total_loss:.4f}")

            torch.save(
                {
                    "epoch": num_epochs,
                    "state_dict": mix_model.state_dict(),
                    "loss": total_loss,
                },
                str(ckpt_path),
            )
            if verbose:
                print(f"[BMTC] Mixer saved to {ckpt_path}")

        return {
            "save_root": str(save_root),
            "mixer_checkpoint": str(ckpt_path),
            "background_images_shape": list(background_images.shape),
            "num_bg_frames": int(background_images.shape[0]),
            "num_attack_videos": len(selected_info),
        }

    # ------------------------------------------------------------------
    # Lazy artifact loading
    # ------------------------------------------------------------------

    def _load_artifacts(self, device: torch.device) -> tuple[_Mixer, torch.Tensor, BaseVideoClassifier]:
        """Load mixer, background images, and surrogate; cache on self."""
        if self._mixer is not None and self._background_images is not None and self._surrogate is not None:
            return self._mixer, self._background_images, self._surrogate

        from vcr_bench.models import create_model

        save_root = Path(self.save_root)
        bg_npy_path = save_root / "background_images.npy"
        ckpt_path = save_root / "mixer_checkpoint" / "mixer.pth"

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"BMTC mixer checkpoint not found at {ckpt_path}. "
                "Run prepare_offline() first (e.g. python -m vcr_bench.cli.prepare --attack bmtc ...)."
            )
        if not bg_npy_path.exists():
            raise FileNotFoundError(
                f"BMTC background_images.npy not found at {bg_npy_path}. "
                "Run prepare_offline() first."
            )

        dev_str = str(device)
        surrogate = create_model("tsm_surrogate", device=dev_str)
        surrogate.eval()
        surrogate_fn = _ThwcSurrogate(surrogate)

        background_images = torch.tensor(
            np.load(str(bg_npy_path)), dtype=torch.float32, device=device
        )

        mix_model = _Mixer(
            epsilon=self.eps,
            action_dim=self.num_classes,
            surrogate_fn=surrogate_fn,
            target_fns=[],
            gamma=self.gamma,
            w1=self.w1,
            w2=self.w2,
            device=device,
        ).to(device).eval()

        ck = torch.load(str(ckpt_path), map_location=device)
        mix_model.load_state_dict(ck.get("state_dict", ck), strict=False)

        self._mixer = mix_model
        self._background_images = background_images
        self._surrogate = surrogate
        return mix_model, background_images, surrogate

    # ------------------------------------------------------------------
    # attack_query
    # ------------------------------------------------------------------

    def attack_query(
        self,
        model: BaseVideoClassifier,
        x: torch.Tensor,
        *,
        query_label,
        input_format: str = "NTHWC",
        y=None,
        targeted: bool = False,
        video_path: str | None = None,
    ) -> torch.Tensor:
        del query_label  # BMTC queries the model directly

        x_ref = self._ensure_float_attack_tensor(x.detach().clone())
        x_flat, restore_shape = flatten_temporal_video(x_ref)  # (T, H, W, C)
        T, H, W, C = x_flat.shape
        device = x_flat.device

        hi = 255.0 if float(x_flat.max().item()) > 1.5 else 1.0
        lo = 0.0
        eps_sc = self.eps if hi > 1.5 else self.eps / 255.0

        # Query model for benign label
        with torch.no_grad():
            benign_scores = model(x_ref, input_format=input_format, enable_grad=False).reshape(-1)
        benign_label = int(benign_scores.argmax().item())
        if self.verbose:
            benign_conf = float(F.softmax(benign_scores.float(), dim=0)[benign_label].item())
            print(f"[bmtc] benign label={benign_label} conf={benign_conf:.4f}", flush=True)

        if targeted:
            target_label = int(benign_scores.argmin().item()) if y is None else (y if isinstance(y, int) else int(y.item()))
        else:
            target_label = benign_label

        # Load pre-trained artifacts
        mix_model, background_images, surrogate = self._load_artifacts(device)

        # ---- Apply Mixer ----
        x_thwc = x_flat[:self.attacked_frames] if T > self.attacked_frames else x_flat  # (T', H, W, C)
        y_tensor = torch.tensor(target_label, device=device)

        with torch.no_grad():
            _, x_adv_thwc = mix_model(x_thwc, y_tensor, background_images, 0)
        # Clamp to valid L∞ range
        lower = torch.clamp(x_thwc - eps_sc, lo, hi)
        upper = torch.clamp(x_thwc + eps_sc, lo, hi)
        x_adv_thwc = torch.clamp(x_adv_thwc, lower, upper)

        # Pad / replace frames beyond attacked_frames with originals
        if T > self.attacked_frames:
            x_adv_full = x_flat.clone()
            x_adv_full[:self.attacked_frames] = x_adv_thwc
        else:
            x_adv_full = x_adv_thwc

        # ---- Optional BTG refinement ----
        query_count = 1  # one benign query above
        if self.enable_btg and self.steps > 0:
            btg_surrogate_fn = _TchwSurrogate(surrogate)
            btg = _BTG(
                surrogate_fn=btg_surrogate_fn,
                epsilon=eps_sc,
                steps=self.steps,
                gamma=self.btg_gamma,
                device=device,
            )
            x_adv_tchw = x_adv_full.permute(0, 3, 1, 2)  # (T, C, H, W)
            x_adv_tchw_refined = btg(x_adv_tchw, target_label)  # (T, C, H, W) on CPU
            x_adv_full = x_adv_tchw_refined.to(device).permute(0, 2, 3, 1)  # (T, H, W, C)
            # Re-clamp after BTG
            x_adv_full = torch.clamp(x_adv_full, lo, hi)

        # Query target model for final label
        with torch.no_grad():
            final_scores = model(
                restore_temporal_video(x_adv_full, restore_shape),
                input_format=input_format,
                enable_grad=False,
            ).reshape(-1)
        final_label = int(final_scores.argmax().item())
        query_count += 1
        if self.verbose:
            benign_conf_after = float(F.softmax(final_scores.float(), dim=0)[benign_label].item())
            print(
                f"[bmtc] final label={final_label} benign_conf={benign_conf_after:.4f}",
                flush=True,
            )

        self.last_result = {
            "iter_count": self.steps if self.enable_btg else 0,
            "query_count": query_count,
            "target_label": target_label if targeted else -1,
            "benign_label": benign_label,
            "final_label": final_label,
        }

        return restore_temporal_video(x_adv_full, restore_shape).detach()


ATTACK_CLASS = BMTCAttack


def create(**kwargs) -> BMTCAttack:
    return BMTCAttack(**kwargs)
