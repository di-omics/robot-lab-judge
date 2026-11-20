"""The frozen vision tower: SigLIP 2 inside LFM2.5-VL-450M, last hidden state BEFORE the multimodal projector.

The pictures are already sized to a multiple of 16 px, so no further resize happens. Preprocessing copies the
checkpoint's own image processor: scale to 0-1, normalize with mean 0.5 and std 0.5, cut 16 x 16 patches in
row-major order with channels last. All pictures in 1 call have the same size, so there is no padding.
"""

import numpy as np

from . import MODEL_ID

PATCH = 16
MEAN = 0.5
STD = 0.5


def to_patches(image) -> tuple[np.ndarray, tuple[int, int]]:
    """(patches of shape (h*w, 16*16*3), (h, w)) for 1 picture whose sides are multiples of 16."""
    a = np.asarray(image, dtype=np.float32)
    h, w, c = a.shape
    if h % PATCH or w % PATCH:
        raise ValueError(f"picture {w} x {h} is not a multiple of {PATCH}")
    a = (a / 255.0 - MEAN) / STD
    gh, gw = h // PATCH, w // PATCH
    p = a.reshape(gh, PATCH, gw, PATCH, c).transpose(0, 2, 1, 3, 4).reshape(gh * gw, -1)
    return p, (gh, gw)


class Tower:
    def __init__(self, model_id: str = MODEL_ID):
        from mlx_vlm import load

        from ..judge_mlx import local_snapshot

        self.path = local_snapshot(model_id)
        self.snapshot = self.path.name
        model, _ = load(str(self.path))
        self.vision = model.vision_tower
        self.n_layers = len(self.vision.encoder.layers)
        self.hidden = int(model.config.vision_config.hidden_size)
        self.dtype = str(self.vision.embeddings.patch_embedding.weight.dtype)
        del model

    def encode(self, images) -> tuple[np.ndarray, tuple[int, int]]:
        """Last hidden state (after the tower's own post layer norm, before the projector) of same-size pictures.

        @return float32 array (n_pictures, h*w, hidden) and the patch grid (h, w).
        """
        import mlx.core as mx

        packed = [to_patches(im) for im in images]
        grids = {g for _, g in packed}
        if len(grids) != 1:
            raise ValueError(f"pictures in 1 call must share a size, got grids {grids}")
        grid = grids.pop()
        x = mx.array(np.stack([p for p, _ in packed]))
        shapes = mx.array(np.array([grid] * len(packed), dtype=np.int32))
        _, _, last = self.vision(x, output_hidden_states=False, spatial_shapes=shapes, pixel_attention_mask=None)
        out = np.array(last.astype(mx.float32))
        return out, grid
