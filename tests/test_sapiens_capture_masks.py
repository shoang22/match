"""Tests for scripts/sapiens/capture_masks.py (interim Sapiens foreground masks).

The script runs in the `sapiens_lite` env and is not part of the `match` package,
so it is loaded by path. The Sapiens checkpoint is never loaded here: the mask
writer takes any segmenter, and the Sapiens segmenter takes any torch module.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sapiens" / "capture_masks.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("capture_masks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture_masks = _load_script()

# Goliath class indices as Sapiens' segmentation head emits them (28 classes).
BACKGROUND, APPAREL, FACE_NECK, HAIR, TONGUE = 0, 1, 3, 4, 27


def _make_capture(root, frames, cams, size=(40, 30)):
    """A capture folder with a space in its name and raw RGB PNGs."""
    capture = root / "background 1"
    images = capture / "images"
    images.mkdir(parents=True)
    w, h = size
    for f in frames:
        for c in cams:
            rgb = np.full((h, w, 3), (10 * c, 20, 30 + f), dtype=np.uint8)
            Image.fromarray(rgb).save(images / f"frame{f:06d}_cam{c:03d}.png")
    return capture


def _striped_segmenter(classes):
    """Fake segmenter: vertical stripes cycling through `classes`, per image."""

    def segment(images):
        out = []
        for img in images:
            h, w = img.shape[:2]
            cols = np.arange(w) % len(classes)
            out.append(np.broadcast_to(np.asarray(classes)[cols], (h, w)).copy())
        return out

    return segment


def test_writes_one_binary_mask_per_image_with_same_filename_and_raw_size(tmp_path):
    capture = _make_capture(tmp_path, frames=[0, 1], cams=[0, 1, 2], size=(40, 30))
    out = tmp_path / "masks sapiens"
    classes = [BACKGROUND, APPAREL, FACE_NECK, HAIR, TONGUE]

    capture_masks.generate_masks(capture, out, _striped_segmenter(classes))

    image_names = sorted(p.name for p in (capture / "images").iterdir())
    assert sorted(p.name for p in out.glob("*.png")) == image_names
    for name in image_names:
        mask = Image.open(out / name)
        assert mask.mode == "L"
        assert mask.size == (40, 30)
        cols = np.asarray(mask)[0]
        # Column pattern: Background, then four foreground classes, repeating.
        assert list(cols[:5]) == [0, 255, 255, 255, 255]
        assert set(np.unique(np.asarray(mask))) == {0, 255}


class _RedIsHairModel:
    """Stand-in for the Sapiens segmentation module.

    Like Sapiens, it takes a normalized RGB batch (B, 3, H, W) and returns class
    logits at quarter resolution. It predicts Hair wherever the red channel is
    above the Sapiens mean, Background elsewhere.
    """

    num_classes = 28

    def __init__(self):
        self.input_shapes = []
        self.inputs = []

    def __call__(self, x):
        import torch
        import torch.nn.functional as F

        self.input_shapes.append(tuple(x.shape))
        self.inputs.append(x.float().clone())
        red = F.avg_pool2d(x[:, :1].float(), 4)
        logits = torch.zeros(x.shape[0], self.num_classes, *red.shape[2:])
        logits[:, HAIR] = red[:, 0]
        return logits.to(x.dtype)


def test_sapiens_segmenter_returns_class_maps_at_raw_image_size():
    import torch

    model = _RedIsHairModel()
    segment = capture_masks.SapiensSegmenter(
        model, input_shape=(64, 48), device="cpu", dtype=torch.float32
    )
    h, w = 50, 70
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, : w // 2] = (230, 20, 20)  # red half -> Hair
    img[:, w // 2 :] = (20, 20, 230)  # blue half -> Background

    (class_map,) = segment([img])

    assert model.input_shapes == [(1, 3, 64, 48)]
    assert class_map.shape == (h, w)
    assert np.all(class_map[:, : w // 2 - 3] == HAIR)
    assert np.all(class_map[:, w // 2 + 3 :] == BACKGROUND)


def test_contact_sheet_shows_mask_overlay_for_every_camera_of_the_chosen_frame(tmp_path):
    cams = [0, 1, 2, 3, 4]
    capture = _make_capture(tmp_path, frames=[0, 7], cams=cams, size=(40, 30))
    out = tmp_path / "masks"

    def left_half_is_face(images):
        maps = []
        for img in images:
            m = np.zeros(img.shape[:2], dtype=np.int64)
            m[:, : img.shape[1] // 2] = FACE_NECK
            maps.append(m)
        return maps

    sheet_path = capture_masks.generate_masks(
        capture, out, left_half_is_face, contact_sheet_frame=7, contact_sheet_columns=3
    )

    assert sheet_path == out / "contact_sheets" / "frame000007.png"
    # The sheet is not a mask and must not look like one to the dataset.
    assert sorted(p.name for p in out.glob("*.png")) == sorted(
        p.name for p in (capture / "images").iterdir()
    )
    sheet = np.asarray(Image.open(sheet_path).convert("RGB")).astype(int)
    # 5 cameras in 3 columns: a 2 x 3 grid of equal tiles, the last one empty.
    th, tw = sheet.shape[0] // 2, sheet.shape[1] // 3
    assert (2 * th, 3 * tw) == sheet.shape[:2]
    assert tw / th == pytest.approx(40 / 30, rel=0.05)
    tiles = [sheet[r * th : (r + 1) * th, c * tw : (c + 1) * tw] for r in range(2) for c in range(3)]
    assert np.all(tiles[-1] == tiles[-1][0, 0])
    tiles = tiles[: len(cams)]
    for tile in tiles:
        th, tw = tile.shape[:2]
        fg = tile[th // 3 : 2 * th // 3, tw // 8 : 3 * tw // 8]
        bg = tile[th // 3 : 2 * th // 3, 5 * tw // 8 : 7 * tw // 8]
        # The source image is flat, so any difference comes from the overlay.
        assert np.abs(fg.mean(axis=(0, 1)) - bg.mean(axis=(0, 1))).sum() > 30


def test_contact_sheet_is_not_written_unless_requested(tmp_path):
    capture = _make_capture(tmp_path, frames=[0], cams=[0, 1], size=(40, 30))
    out = tmp_path / "masks"

    result = capture_masks.generate_masks(capture, out, _striped_segmenter([0, 1]))

    assert result is None
    assert not (out / "contact_sheets").exists()


def test_sapiens_segmenter_letterboxes_instead_of_squashing():
    # Capture images are square, Sapiens' input is portrait: the face must keep its
    # aspect ratio, and the class map must still line up with the raw image.
    import torch

    model = _RedIsHairModel()
    segment = capture_masks.SapiensSegmenter(
        model, input_shape=(96, 64), device="cpu", dtype=torch.float32
    )
    img = np.full((60, 60, 3), 200, dtype=np.uint8)
    img[:, :, 0] = 20  # not red
    img[:15] = (230, 20, 20)  # red top band -> Hair

    (class_map,) = segment([img])

    pixels = model.inputs[0][0] * torch.tensor(segment.STD).view(3, 1, 1) + torch.tensor(
        segment.MEAN
    ).view(3, 1, 1)
    rows, cols = np.nonzero(pixels.sum(dim=0).numpy() > 1)
    content_h, content_w = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
    assert content_w == 64 and content_h == 64  # square stays square
    assert class_map.shape == (60, 60)
    assert np.all(class_map[:12] == HAIR)
    assert np.all(class_map[18:] == BACKGROUND)


def _export_fake_checkpoint(path):
    """A torch.export'ed module with Sapiens' segmentation I/O, saved like the .pt2.

    Like the real sapiens_1b seg .pt2, it declares a dynamic batch dimension (2..32).
    """
    import torch

    class RedIsHair(torch.nn.Module):
        def forward(self, x):
            red = torch.nn.functional.avg_pool2d(x[:, :1], 4)
            others = torch.zeros_like(red).expand(-1, 27, -1, -1)
            # Channel order: Background, Apparel, Face_Neck, ... ; put "red" at Hair.
            return torch.cat([others[:, :HAIR], red, others[:, HAIR:]], dim=1)

    example = (torch.zeros(2, 3, 64, 48),)
    batch = {0: torch.export.Dim("batch", min=2, max=32)}
    exported = torch.export.export(RedIsHair(), example, dynamic_shapes=(batch,))
    torch.export.save(exported, str(path))


def test_cli_writes_masks_and_contact_sheet_for_capture_with_space_in_name(tmp_path):
    capture = _make_capture(tmp_path, frames=[0, 1], cams=[0, 1, 2], size=(40, 30))
    red = np.zeros((30, 40, 3), dtype=np.uint8)
    red[:, :20] = (230, 20, 20)
    Image.fromarray(red).save(capture / "images" / "frame000001_cam001.png")
    checkpoint = tmp_path / "fake seg.pt2"
    _export_fake_checkpoint(checkpoint)
    out = tmp_path / "sapiens masks"

    subprocess.run(
        [
            sys.executable, str(SCRIPT), str(capture), str(out),
            "--checkpoint", str(checkpoint), "--device", "cpu",
            "--shape", "64", "48", "--contact-sheet-frame", "1",
            "--batch-size", "4",  # 6 images: the last batch holds 2
        ],
        check=True,
    )

    assert sorted(p.name for p in out.glob("*.png")) == sorted(
        p.name for p in (capture / "images").iterdir()
    )
    mask = np.asarray(Image.open(out / "frame000001_cam001.png"))
    assert mask.shape == (30, 40)
    assert np.all(mask[:, :16] == 255) and np.all(mask[:, 24:] == 0)
    assert (out / "contact_sheets" / "frame000001.png").is_file()


def test_help_documents_usage():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=True
    )
    for expected in ["sapiens_lite", "capture_dir", "output_dir", "--contact-sheet-frame",
                     "Background", "frame{FFFFFF}_cam{CCC}.png"]:
        assert expected in result.stdout



def test_capture_without_matching_images_is_an_error(tmp_path):
    capture = tmp_path / "background 1"
    (capture / "images").mkdir(parents=True)
    (capture / "images" / "notes.png").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="frame"):
        capture_masks.generate_masks(capture, tmp_path / "masks", _striped_segmenter([0]))
