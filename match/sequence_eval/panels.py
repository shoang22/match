"""Labelled panels of the sequence-evaluation grid video, and the grid layout.

A panel turns a FrameContext into one image per grid-video frame that shows that source frame
(`ctx.output_frame_indices`). Panels showing one image per source frame subclass StaticPanel and
implement `image`; panels that change within a source frame's hold (e.g. an orbiting camera) implement
`render` directly. To add a panel, write a class here and add it to `build_panels` in
scripts/match/eval_sequence.py. The layout code doesn't change.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABEL_HEIGHT = 32
LABEL_FONT_SIZE = 22


class Panel:
    label = ''

    def render(self, ctx):
        """Returns a list of (3,H,W) float images in [0,1], one per index in ctx.output_frame_indices."""
        raise NotImplementedError


class StaticPanel(Panel):
    """A panel that shows the same image for the whole hold of a source frame."""

    def image(self, ctx):
        raise NotImplementedError

    def render(self, ctx):
        img = self.image(ctx)
        return [img] * len(ctx.output_frame_indices)


class PredictionPanel(StaticPanel):
    """MATCH prediction from the fixed novel viewpoint, with that held-out camera's real calibration."""

    def __init__(self, camera_id):
        self.label = f'MATCH prediction (cam {camera_id})'

    def image(self, ctx):
        return ctx.fixed_view.pred


class RealPanel(StaticPanel):
    """Real held-out image from the fixed viewpoint, background outside the foreground mask painted white."""

    def __init__(self, camera_id):
        self.label = f'Real held-out (cam {camera_id})'

    def image(self, ctx):
        return ctx.fixed_view.real_on_background(1.0)


def _to_uint8(img):
    """(3,H,W) float tensor in [0,1] -> (H,W,3) uint8 array."""
    return (img.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def label_image(img, label):
    """(3,H,W) float tensor -> (H+LABEL_HEIGHT, W, 3) uint8 with the label in a black band above the image."""
    pixels = _to_uint8(img)
    h, w, _ = pixels.shape
    canvas = Image.new('RGB', (w, h + LABEL_HEIGHT), (0, 0, 0))
    canvas.paste(Image.fromarray(pixels), (0, LABEL_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=LABEL_FONT_SIZE)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    draw.text(((w - (right - left)) // 2 - left, (LABEL_HEIGHT - (bottom - top)) // 2 - top), label,
              fill=(255, 255, 255), font=font)
    return np.asarray(canvas)


def grid_shape(n_panels):
    """(rows, cols) of the grid: 2 panels side by side, 3-4 panels in a 2x2 grid, and so on."""
    cols = math.ceil(math.sqrt(n_panels))
    return math.ceil(n_panels / cols), cols


def compose_grid(images, labels):
    """Lays out equally sized labelled panels row-major in a grid. Returns (H,W,3) uint8; empty cells are black."""
    tiles = [label_image(img, label) for img, label in zip(images, labels)]
    th, tw, _ = tiles[0].shape
    if any(t.shape != tiles[0].shape for t in tiles):
        raise ValueError(f'All panels must have the same size, got {[t.shape for t in tiles]}')
    rows, cols = grid_shape(len(tiles))
    grid = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
    return grid


def render_grid_frames(panels, ctx):
    """Renders every panel for the source frame and yields (output_frame_index, grid image) pairs in order."""
    rendered = [panel.render(ctx) for panel in panels]
    for p, images in zip(panels, rendered):
        if len(images) != len(ctx.output_frame_indices):
            raise ValueError(f"Panel '{p.label}' returned {len(images)} images for "
                             f'{len(ctx.output_frame_indices)} output frames')
    labels = [p.label for p in panels]
    for j, k in enumerate(ctx.output_frame_indices):
        yield k, compose_grid([images[j] for images in rendered], labels)
