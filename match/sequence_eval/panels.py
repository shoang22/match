"""Labelled panels of the sequence-evaluation grid video, and the grid layout.

A panel turns a FrameContext into one image per grid-video frame that shows that source frame
(`ctx.output_frame_indices`). Panels showing one image per source frame subclass StaticPanel and
implement `image`; panels that change within a source frame's hold (e.g. an orbiting camera) implement
`render` directly. To add a panel, write a class here and add it to `build_panels` in
scripts/match/eval_sequence.py. The layout code doesn't change.
"""
import atexit
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import gin
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from match.utils import mesh_util
from match.utils.camera_sweep_util import (SWEEP_FOCAL_SCALE, SWEEP_HORIZ_RANGE_DEG, SWEEP_VERT_RANGE_DEG,
                                           generate_camera_sweep_extrinsics, sweep_intrinsics)

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


def _render_grey_mesh(verts, faces, W2C, K, height, width, color):
    """pyrender of one mesh on white. Runs in TempehMeshPanel's worker process. Returns (H,W,3) float32 in [0,1]."""
    from match.utils import render_util
    return render_util.render_mesh(
        vertices=verts[None], faces=faces, camera_extrinsics=W2C[None], camera_intrinsics=K[None],
        image_size=(height, width), vertex_colors=np.full((1, len(verts), 3), color, dtype=np.float32),
        background_color=np.ones(3, dtype=np.float32))[0]


class TempehMeshPanel(StaticPanel):
    """The bare TEMPEH coarse mesh MATCH got as input, plainly shaded, from the fixed viewpoint.

    Rendered with the held-out camera's real calibration at the panel size, so it lines up with the
    prediction and real panels. Shows whether jitter in the prediction is inherited from the mesh.

    pyrender runs in a separate spawned process: in the MATCH process, after the first forward pass,
    its software GL (Mesa llvmpipe) spins for minutes compiling shaders (killed after 6), while a fresh
    process renders a frame in about a second.
    """
    color = (0.45, 0.45, 0.45)  # mid grey; the renderer's 4 directional lights brighten it

    def __init__(self, camera_id, mesh_path):
        """mesh_path: OBJ of the coarse-mesh topology (Options.template_mesh_path), drawn with all its faces."""
        self.label = f'TEMPEH coarse mesh (cam {camera_id})'
        self.faces = mesh_util.load_obj(mesh_path)['vi'].astype(np.int32)
        self._worker = None

    def image(self, ctx):
        view = ctx.fixed_view
        h, w = ctx.height, ctx.width
        verts = ctx.batch['coarse_verts'][0].detach().float().cpu().numpy()  # (V,3), dataset coordinates
        if verts.size == 0:
            raise ValueError(f'Frame {ctx.frame_id} has no TEMPEH coarse mesh (coarse_verts is empty)')
        W2C = torch.linalg.inv(view.C2W.detach().float().cpu())[:3, :4].numpy()
        fx, fy, cx, cy = view.fxfycxcy.detach().float().cpu().tolist()  # normalised by the panel's W,H
        K = np.array([[fx * w, 0, cx * w], [0, fy * h, cy * h]], dtype=np.float32)
        if self._worker is None:
            self._worker = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'))
            atexit.register(self._worker.shutdown)
        rendered = self._worker.submit(_render_grey_mesh, verts, self.faces, W2C, K, h, w, self.color).result()
        return torch.from_numpy(rendered).permute(2, 0, 1)


@gin.configurable
class OrbitPanel(Panel):
    """A slow orbit around the head while time plays, one camera pose per grid-video frame.

    The path is the camera-sweep path (scripts/match/save_camera_sweeps.py) with its default angle range and
    intrinsics, rendered on a centred copy of the frame's Gaussians. It starts and ends at the frontal pose and
    loops every `orbit_frames` video frames. The source frame shown changes with the other panels.
    """
    label = 'Orbit'

    def __init__(self, orbit_frames=200, vert_range_deg=SWEEP_VERT_RANGE_DEG, horiz_range_deg=SWEEP_HORIZ_RANGE_DEG,
                 focal_scale=SWEEP_FOCAL_SCALE, render_chunk=8):
        # One extra step, dropped, so the frontal end pose isn't shown twice when the orbit loops.
        path = generate_camera_sweep_extrinsics(
            vert_range_deg=vert_range_deg, horiz_range_deg=horiz_range_deg, num_steps=orbit_frames + 1)
        self.C2W = torch.from_numpy(path[:orbit_frames]).float()
        self.focal_scale = focal_scale
        self.render_chunk = render_chunk

    def render(self, ctx):
        gaussians = dict(ctx.gaussians)
        xyz = gaussians['xyz']
        gaussians['xyz'] = xyz - xyz.mean(dim=-2, keepdim=True)  # the path looks at the origin
        C2W = self.C2W[[k % len(self.C2W) for k in ctx.output_frame_indices]].to(xyz.device)
        fxfycxcy = sweep_intrinsics(ctx.batch['fxfycxcy'], self.focal_scale).expand(len(C2W), 4)
        images = [ctx.render(C2W[i:i + self.render_chunk], fxfycxcy[i:i + self.render_chunk], gaussians=gaussians)
                  for i in range(0, len(C2W), self.render_chunk)]
        return list(torch.cat(images))


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
