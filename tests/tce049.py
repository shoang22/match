"""Facts about the TCE049 capture and the SEN_a_good_morrow_to_you_my_boy segment that several tests check.

Independent of the code under test: frame IDs come from the 4TB frame list, camera IDs and angles from
camera_calibration.json (see configs/match/eval_sequence_SEN_a_good_morrow.gin).
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

AVA_ROOT = REPO_ROOT / 'data/ava-256'
DECODER_DIR = AVA_ROOT / '20230807--0839--TCE049' / 'decoder'
ASSETS = REPO_ROOT / 'assets'

SEGMENT = 'SEN_a_good_morrow_to_you_my_boy'
# The 4TB tier keeps every 12th recorded frame (0.4 s apart at 30 fps): 32885, 32897, ..., 33353.
SEGMENT_FRAMES = [str(32885 + 12 * k) for k in range(40)]

# The camera the dataset picks for each input angle of the pretrained uv_res_1024 config, in angle order.
INPUT_CAMERA_IDS = ['401875', '401919', '402020', '402012', '401920', '401965',
                    '401630', '401916', '401860', '402038', '401966', '401969']

# Held-out cameras with their [az, el] in degrees, in config order.
HELDOUT_CAMERAS = {
    '402040': [7.7, 24.0],   # interpolating; closest to frontal, so the fixed novel viewpoint
    '401905': [-31.0, 4.7],  # interpolating
    '402031': [75.6, -1.5],  # extrapolating
}
FIXED_VIEW_CAMERA = '402040'
