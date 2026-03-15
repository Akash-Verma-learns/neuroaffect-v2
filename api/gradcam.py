"""
gradcam.py  —  Grad-CAM++ for MRIEncoder (SimpleCNN)

Uses torch.autograd.grad() instead of backward hooks —
works reliably across all PyTorch versions.
"""

import io, base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAMPP:
    """
    Grad-CAM++ on the last ConvBlock of MRIEncoder.
    Targets features[3].block[2] = ReLU output (after Conv+BN, before MaxPool).
    Hooking after activation gives non-zero, spatially meaningful feature maps.
    """

    def __init__(self, mri_encoder: torch.nn.Module):
        self.encoder = mri_encoder
        # block[0]=Conv2d  block[1]=BN  block[2]=ReLU  block[3]=MaxPool
        # Hook after ReLU: captures activated maps at full 8×8 spatial resolution
        self.target_layer = mri_encoder.features[3].block[2]

    def generate(self, img_tensor: torch.Tensor,
                 class_idx: int = None,
                 out_size: int = 224) -> np.ndarray:
        """
        img_tensor : (1, 1, H, W)  any device
        Returns    : normalised heatmap (out_size, out_size) float32 in [0,1]
        """
        self.encoder.eval()

        # Fresh tensor with gradient tracking
        x = img_tensor.detach().float()

        # ── Forward pass — capture activations at target layer ──────────
        activations = []

        def fwd_hook(module, inp, out):
            activations.append(out)

        handle = self.target_layer.register_forward_hook(fwd_hook)

        try:
            # Run forward with grad enabled
            with torch.enable_grad():
                x = x.requires_grad_(True)
                emb, logits = self.encoder(x)

                if class_idx is None:
                    class_idx = int(logits.argmax(-1).item())

                # ── Compute gradients via autograd.grad ──────────────────
                if not activations:
                    return self._uniform(out_size)

                A = activations[0]  # (1, C, h, w)

                # Score for target class
                score = logits[0, class_idx]

                # Grad of score w.r.t. activation map
                grads = torch.autograd.grad(
                    outputs=score,
                    inputs=A,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True,
                )[0]

                if grads is None:
                    return self._uniform(out_size)

        finally:
            handle.remove()

        # ── Grad-CAM++ weights ───────────────────────────────────────────
        A_np = A.detach().squeeze(0)      # (C, h, w)
        g_np = grads.detach().squeeze(0)  # (C, h, w)

        g2 = g_np.pow(2)
        g3 = g_np.pow(3)
        denom   = 2.0 * g2 + (A_np * g3).sum((-2, -1), keepdim=True) + 1e-7
        alpha   = g2 / denom
        weights = (alpha * F.relu(g_np)).sum((-2, -1))  # (C,)

        cam = (weights.view(-1, 1, 1) * A_np).sum(0)   # (h, w)
        cam = F.relu(cam)

        # Upsample to output size
        cam_up = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Normalise
        mn, mx = cam_up.min(), cam_up.max()
        if mx - mn > 1e-7:
            cam_up = (cam_up - mn) / (mx - mn)
        else:
            # Fallback: use raw activation energy when gradients are flat
            act_energy = A_np.pow(2).mean(0)
            act_energy = F.interpolate(
                act_energy.unsqueeze(0).unsqueeze(0),
                size=(out_size, out_size), mode="bilinear", align_corners=False
            ).squeeze().cpu().numpy()
            ae_mn, ae_mx = act_energy.min(), act_energy.max()
            cam_up = ((act_energy - ae_mn) / (ae_mx - ae_mn + 1e-7)
                      if ae_mx > ae_mn else np.zeros((out_size, out_size)))

        return cam_up.astype(np.float32)

    @staticmethod
    def _uniform(size):
        return np.zeros((size, size), dtype=np.float32)


# ── Colour helpers ──────────────────────────────────────────────────────────────

def heatmap_to_rgb(cam: np.ndarray) -> np.ndarray:
    """Jet colourmap  blue → cyan → green → yellow → red  (uint8 RGB)."""
    r = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_heatmap(mri_tensor: torch.Tensor,
                    cam: np.ndarray,
                    alpha: float = 0.55,
                    out_size: int = 224) -> np.ndarray:
    """Blend greyscale MRI with Grad-CAM++ heatmap. Returns RGB uint8."""
    # MRI → greyscale uint8
    mri = mri_tensor.squeeze().detach().cpu().numpy().astype(float)
    mri = np.clip((mri + 1.0) / 2.0, 0.0, 1.0)
    mri_pil = Image.fromarray((mri * 255).astype(np.uint8), mode="L")
    mri_rgb  = np.array(mri_pil.resize((out_size, out_size),
                                        Image.BILINEAR).convert("RGB"))

    # Resize cam and colourise
    cam_pil  = Image.fromarray((cam * 255).astype(np.uint8), mode="L")
    cam_rs   = np.array(cam_pil.resize((out_size, out_size),
                                        Image.BILINEAR)).astype(np.float32) / 255.0
    heat_rgb = heatmap_to_rgb(cam_rs)

    # Alpha blend
    out = alpha * heat_rgb.astype(float) + (1.0 - alpha) * mri_rgb.astype(float)
    return out.clip(0, 255).astype(np.uint8)


def arr_to_b64(arr: np.ndarray) -> str:
    """numpy RGB array → base64 PNG string."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def mri_to_b64(mri_tensor: torch.Tensor, size: int = 224) -> str:
    """MRI tensor (1,1,H,W) in [-1,1] → base64 greyscale PNG."""
    mri = mri_tensor.squeeze().detach().cpu().numpy()
    mri = np.clip((mri + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(mri, mode="L").resize(
        (size, size), Image.BILINEAR).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")