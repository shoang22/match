"""Interim foreground masks for a calibrated multi-view capture, from Sapiens.

Runs the Sapiens 1B Goliath segmentation checkpoint over a capture's raw images
(CAPTURE_DIR/images/frame{FFFFFF}_cam{CCC}.png) and writes one mask per image to
OUTPUT_DIR, with the same filename:

  - single-channel 8-bit PNG, same size as the raw image (no undistortion, no
    resize), so it goes through the same loading path as the real masks;
  - 255 where the predicted class is anything other than Background, 0 where it
    is Background.

Images are read as sRGB (no gamma correction) and letterboxed into the model's
1024x768 input, so square captures are not squashed.

Write the masks to their own folder (e.g. next to, not inside, the real mask
folder). OUTPUT_DIR is one flat folder of masks, usable directly as the dataset's
mask_directory for a single capture. For a root holding several captures, write
each capture to MASK_ROOT/<capture folder name> and point mask_directory at
MASK_ROOT. Switching the dataset config from these to the real masks later only
changes the mask folder path.

With --contact-sheet-frame N, also writes OUTPUT_DIR/contact_sheets/frameNNNNNN.png:
every camera of frame N with its mask overlaid (foreground tinted red with a green
outline, background dimmed), to check mask quality (hair against foliage, a white
shirt against the sky) before running TEMPEH and MATCH.

Run it in the sapiens_lite conda env (see scripts/sapiens/install_environment.sh),
from any directory. Quote paths that contain spaces:

  conda activate sapiens_lite
  python scripts/sapiens/capture_masks.py \
      "/path/to/datasets/background 1" "/path/to/datasets/background 1 masks_sapiens" \
      --contact-sheet-frame 0
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

IMAGE_NAME = re.compile(r"^frame(\d{6})_cam(\d{3})\.png$")
BACKGROUND_CLASS = 0  # index of "Background" in Sapiens' GOLIATH_CLASSES


def _capture_images(capture_dir):
    images_dir = Path(capture_dir) / "images"
    paths = sorted(p for p in images_dir.iterdir() if IMAGE_NAME.match(p.name))
    if not paths:
        raise FileNotFoundError(f"no frame{{FFFFFF}}_cam{{CCC}}.png images in {images_dir}")
    return paths


def _read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"could not read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def foreground_mask(class_map):
    """255 where the predicted class is not Background, 0 elsewhere."""
    return np.where(np.asarray(class_map) != BACKGROUND_CLASS, 255, 0).astype(np.uint8)


class SapiensSegmenter:
    """Runs a Sapiens segmentation module on raw RGB images.

    Preprocessing follows third_party/sapiens/lite/demo: letterbox into the
    model's input shape (keeping the aspect ratio), RGB, Sapiens mean/std. The
    logits of the image region are upsampled back to each image's raw size
    before the argmax, so class maps are in raw image space.
    """

    MEAN = (123.5, 116.5, 103.5)
    STD = (58.5, 57.0, 57.5)

    def __init__(self, model, input_shape=(1024, 768), device="cuda:0", dtype=None):
        import torch

        self.model = model
        self.input_shape = tuple(input_shape)  # (height, width)
        self.device = device
        self.dtype = dtype if dtype is not None else torch.bfloat16

    def _letterbox(self, rgb):
        """Fits the image into the input shape at its own aspect ratio, black-padded.

        Same as Sapiens' `pad=True` preprocessing (zip_image_dataset.py): pads are
        rounded down to multiples of 16. Returns the padded image and the
        (top, left, height, width) box the image occupies.
        """
        in_h, in_w = self.input_shape
        h, w = rgb.shape[:2]
        scale = min(in_h / h, in_w / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        top = int(((in_h - new_h) / 2 // 16) * 16)
        left = int(((in_w - new_w) / 2 // 16) * 16)
        padded = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        padded[top : top + new_h, left : left + new_w] = cv2.resize(
            rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        return padded, (top, left, new_h, new_w)

    def _normalize(self, rgb):
        import torch

        img = torch.from_numpy(rgb).permute(2, 0, 1).float()
        mean = torch.tensor(self.MEAN).view(3, 1, 1)
        std = torch.tensor(self.STD).view(3, 1, 1)
        return (img - mean) / std

    def __call__(self, images):
        import torch
        import torch.nn.functional as F

        padded, boxes = zip(*(self._letterbox(img) for img in images))
        batch = torch.stack([self._normalize(p) for p in padded])
        with torch.no_grad():
            logits = self.model(batch.to(self.device, self.dtype)).float().cpu()
        logits = F.interpolate(logits, size=self.input_shape, mode="bilinear")
        class_maps = []
        for img, img_logits, (top, left, h, w) in zip(images, logits, boxes):
            crop = img_logits[None, :, top : top + h, left : left + w]
            up = F.interpolate(crop, size=img.shape[:2], mode="bilinear")
            class_maps.append(up[0].argmax(dim=0).numpy())
        return class_maps


def _overlay(rgb, mask):
    """Foreground tinted red with a green outline, background dimmed."""
    fg = (mask.astype(np.float32) / 255.0)[..., None]
    red = np.array([255, 0, 0], dtype=np.float32)
    img = rgb.astype(np.float32)
    out = fg * (0.6 * img + 0.4 * red) + (1 - fg) * (0.35 * img)
    out = out.astype(np.uint8)
    contours, _ = cv2.findContours(
        (mask >= 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    thickness = max(1, round(max(mask.shape) / 400))
    cv2.drawContours(out, contours, -1, (0, 255, 0), thickness)
    return out


def write_contact_sheet(capture_dir, mask_dir, frame, path, columns=None, tile_width=512):
    """Grid of mask overlays, one tile per camera of `frame`, labelled by camera."""
    frame_images = [
        p for p in _capture_images(capture_dir) if int(IMAGE_NAME.match(p.name)[1]) == frame
    ]
    if not frame_images:
        raise ValueError(f"no images for frame {frame} in {Path(capture_dir) / 'images'}")
    columns = columns or int(np.ceil(np.sqrt(len(frame_images))))
    rows = int(np.ceil(len(frame_images) / columns))

    tiles = []
    for image_path in frame_images:
        mask_path = Path(mask_dir) / image_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"could not read mask {mask_path}")
        tile = _overlay(_read_rgb(image_path), mask)
        h, w = tile.shape[:2]
        tile_height = round(tile_width * h / w)
        tile = cv2.resize(tile, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        cam = IMAGE_NAME.match(image_path.name)[2]
        scale = tile_width / 512
        cv2.putText(
            tile, f"cam{cam}", (round(8 * scale), round(28 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(1, round(2 * scale)),
        )
        tiles.append(tile)

    th, tw = tiles[0].shape[:2]
    sheet = np.zeros((rows * th, columns * tw, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        sheet[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = tile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return path


def generate_masks(
    capture_dir,
    output_dir,
    segment,
    batch_size=4,
    contact_sheet_frame=None,
    contact_sheet_columns=None,
):
    """Writes one foreground mask per capture image into `output_dir`.

    Masks are single-channel PNGs with the image's filename and raw size. If
    `contact_sheet_frame` is given, also writes an overlay contact sheet for that
    frame to `output_dir/contact_sheets/frame{FFFFFF}.png` and returns its path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _capture_images(capture_dir)
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        class_maps = segment([_read_rgb(p) for p in batch])
        for path, class_map in zip(batch, class_maps):
            if not cv2.imwrite(str(output_dir / path.name), foreground_mask(class_map)):
                raise IOError(f"could not write mask {output_dir / path.name}")
        print(f"{min(start + batch_size, len(paths))}/{len(paths)} images", flush=True)

    if contact_sheet_frame is None:
        return None
    return write_contact_sheet(
        capture_dir,
        output_dir,
        contact_sheet_frame,
        output_dir / "contact_sheets" / f"frame{contact_sheet_frame:06d}.png",
        columns=contact_sheet_columns,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "assets/sapiens/checkpoints/bfloat16/seg/checkpoints/sapiens_1b"
    / "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_bfloat16.pt2"
)


def load_model(checkpoint, device, dtype):
    """Loads a Sapiens lite checkpoint (torch.export .pt2, or TorchScript)."""
    import torch

    if "_torchscript" in Path(checkpoint).name:
        return torch.jit.load(str(checkpoint)).to(device), torch.float32
    model = torch.export.load(str(checkpoint)).module()
    return model.to(device=device, dtype=dtype), dtype


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("capture_dir", type=Path, help="capture folder containing images/")
    parser.add_argument("output_dir", type=Path, help="folder to write the masks to")
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Sapiens segmentation checkpoint (default: the repo's sapiens_1b bfloat16 .pt2)",
    )
    parser.add_argument("--device", default="cuda:0", help="torch device (default: cuda:0)")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16",
        help="inference dtype for .pt2 checkpoints (default: bfloat16)",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="images per forward pass")
    parser.add_argument(
        "--shape", type=int, nargs=2, default=[1024, 768], metavar=("H", "W"),
        help="model input size; images are letterboxed into it (default: 1024 768)",
    )
    parser.add_argument(
        "--contact-sheet-frame", type=int, default=None, metavar="N",
        help="also write a contact sheet of mask overlays for frame N (all cameras)",
    )
    args = parser.parse_args(argv)

    import torch

    model, dtype = load_model(args.checkpoint, args.device, getattr(torch, args.dtype))
    segment = SapiensSegmenter(model, input_shape=args.shape, device=args.device, dtype=dtype)
    sheet = generate_masks(
        args.capture_dir,
        args.output_dir,
        segment,
        batch_size=args.batch_size,
        contact_sheet_frame=args.contact_sheet_frame,
    )
    print(f"masks written to {args.output_dir}")
    if sheet is not None:
        print(f"contact sheet written to {sheet}")


if __name__ == "__main__":
    main()
