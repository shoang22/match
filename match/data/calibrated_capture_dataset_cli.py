"""Debug CLI for the calibrated-capture dataset: loads one sample and saves a visualization.

Run from the repository root, for example:

    python match/data/calibrated_capture_dataset_cli.py \
        --root "/path/to/background 1" --mask-dir /path/to/masks --out demos/calibrated_sample.png

Rows per camera: the resized full image with the head-crop box (green) and the
projected head center (red cross); the cropped image with the mask overlaid;
the mask alone. The camera-to-slot assignment is logged when the dataset is built.
"""
import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from match.data.calibrated_capture_dataset import CalibratedMultiCaptureDataset  # noqa: E402
from match.utils import geo_util  # noqa: E402

# The pretrained MATCH model's 12 target camera angles [az, el] in degrees.
PRETRAINED_CAMERA_ANGLES = [[0, 0], [0, -15], [20, 0], [20, 36], [20, -15], [-20, 0], [-20, 36],
                            [-20, -15], [40, -5], [40, 25], [-40, -5], [-40, 25]]


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', required=True, help='capture folder, or a folder of capture folders')
    parser.add_argument('--mask-dir', required=True, help='foreground masks, named like the images')
    parser.add_argument('--out', default='demos/calibrated_capture_sample.png')
    parser.add_argument('--index', type=int, default=0, help='sample index')
    parser.add_argument('--height', type=int, default=786,
                        help="resize target of the (Ava-aspect) image; default: the pretrained model's Ava size")
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--crop-height', type=int, default=640)
    parser.add_argument('--crop-width', type=int, default=512)
    parser.add_argument('--crop-offset-y', type=int, default=-128,
                        help='as in configs/match/save_camera_sweeps_calibrated.gin')
    parser.add_argument('--crop-offset-x', type=int, default=0)
    parser.add_argument('--no-crop', action='store_true')
    parser.add_argument('--scene-center', type=float, nargs=3, default=None,
                        help='head center in calibration units (default: optical-axes intersection)')
    parser.add_argument('--all-cameras', action='store_true',
                        help='use every camera instead of the pretrained camera-angle slots')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    dataset = CalibratedMultiCaptureDataset(
        root_path=args.root, mask_directory=args.mask_dir, height=args.height, width=args.width,
        camera_angles=None if args.all_cameras else PRETRAINED_CAMERA_ANGLES,
        head_crop=not args.no_crop, head_crop_height=args.crop_height, head_crop_width=args.crop_width,
        head_crop_offset_x=args.crop_offset_x, head_crop_offset_y=args.crop_offset_y,
        scene_center=args.scene_center, training=None, require_verts=False, require_segmentation=False,
        max_captures=1,
    )
    sample = dataset[args.index]
    single = dataset.single_capture_datasets[dataset.captures[sample['idindex']]]
    visualize(sample, single, args.out)
    print(f'Wrote {args.out}')


def visualize(sample, dataset, out_path):
    images, masks = sample['image'].numpy(), sample['mask'].numpy()
    V, _, H, W = images.shape
    head = sample['headpose'][:3, 3][None, None]
    head_px = geo_util.project_points_to_screen(head, sample['C2W'][None], sample['fxfycxcy'][None], H=H, W=W)
    head_px = head_px[0, :, 0, :2].numpy()  # (V, 2), pixel corner at 0

    fig, axes = plt.subplots(3, V, figsize=(3 * V, 9), squeeze=False)
    for v in range(V):
        camera = str(int(sample['camera'][v]))
        l, t, r, b = sample['bboxs'][v].tolist()
        full = cv2.resize(dataset.load_image(sample['frame'], camera), (dataset.width, dataset.height),
                          interpolation=cv2.INTER_LINEAR)
        hx, hy = head_px[v] + [l, t] - 0.5  # imshow puts pixel centers at integers
        axes[0, v].imshow(full)
        axes[0, v].add_patch(plt.Rectangle((l - 0.5, t - 0.5), r - l, b - t, fill=False, color='lime', lw=1.5))
        axes[0, v].plot(hx, hy, 'r+', ms=12, mew=2)
        axes[0, v].set_title(f'cam {camera} (slot {v})')
        axes[1, v].imshow(images[v].transpose(1, 2, 0))
        axes[1, v].imshow(masks[v, 0], cmap='gray', alpha=0.5, vmin=0, vmax=1)
        axes[1, v].plot(head_px[v, 0] - 0.5, head_px[v, 1] - 0.5, 'r+', ms=12, mew=2)
        axes[1, v].set_title('crop + mask')
        axes[2, v].imshow(masks[v, 0], cmap='gray', vmin=0, vmax=1)
        axes[2, v].set_title('mask')
    for ax in axes.flat:
        ax.axis('off')
    fig.suptitle(f'{sample["subject"]} / {sample["sequence"]} / frame {sample["frame"]}')
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


if __name__ == '__main__':
    main()
