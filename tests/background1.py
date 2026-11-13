"""Facts about the `background 1` capture (12 cameras, 30 frames, outdoor) that several tests check.

Independent of the code under test: frame IDs from the capture's image names (images/frame{FFFFFF}_cam{CCC}.png),
camera slots from the dataset's slot-assignment log in the README ("camera-angle slot assignment"), the TEMPEH output
layout from configs/tempeh/save_predictions_background1.yml.
"""
from tce049 import REPO_ROOT

CAPTURE = 'background 1'  # the capture folder's name: subject, sequence and segment ID
CAPTURE_DIR = REPO_ROOT / 'data/captures' / CAPTURE
TEMPEH_DIR = REPO_ROOT / 'data/tempeh_predictions/background_1/calibrated' / CAPTURE / CAPTURE

FRAMES = [f'{k:06d}' for k in range(30)]
CAMERAS = [str(c) for c in range(12)]
# The camera filling each of the pretrained model's 12 camera-angle slots, in slot order. Every camera is an input.
INPUT_CAMERA_IDS = ['0', '9', '1', '7', '4', '2', '8', '6', '3', '10', '5', '11']
FRONTAL_INPUT_CAMERA = '0'  # slot [az 0, el 0]
# The frame rate isn't recorded with the capture; the configs time the videos at 10 fps (30 frames = 3 s).
CAPTURE_FPS = 10
