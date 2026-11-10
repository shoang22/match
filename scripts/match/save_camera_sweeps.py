from torch._tensor import Tensor
from typing import Any


import argparse 
from collections import defaultdict
from collections.abc import Sequence
import os
from absl import app
from absl import flags
from absl import logging as absl_logging
import accelerate
import einops
import gin
import mediapy as media
import numpy as np
import pudb
import tensorflow as tf
import torch
import tqdm
from match.utils import data_util, file_util, general_util
from match.options import Options
from match import models, data
from match.utils import general_util, render_util, file_util, gin_util, data_util, geo_util
from match.utils.camera_sweep_util import frames_to_video, generate_camera_sweep_extrinsics, sweep_intrinsics
from pyvirtualdisplay import Display
from PIL import Image
from match.runner import MatchRunner
import json
from torchvision.utils import make_grid, save_image



@gin.configurable()
def save_camera_sweeps(runner: MatchRunner, out_dir: str = None, nframes: int=200, fps:int=24, angle_range_hor=[-40,40], angle_range_vert=[-15, 36], n_samples:int=-1, store_config_files: list[str] = None) -> None:
  """saving prediction results as camera sweeps
  """
  opt = runner.opt
  

  if out_dir is None:
    out_dir = runner.exp_dir / "camera_sweeps" / f"iter_{runner.pretrained_model_iter:08d}"
  out_dir = file_util.Path(out_dir)
  runner.logger.info(f"Saving camera sweeps to [{out_dir}]\n")
  
  out_dir.mkdir(parents=True, exist_ok=True)
  # saving configs
  for i in range(len(store_config_files)):
    try:
      file_util.copy(store_config_files[i], out_dir / f"config_{i}.gin", True)
    except Exception as e:
      absl_logging.info(
          f"Failed to copy gin config from [{store_config_files[i]}] to"
          f" [{out_dir / f'config_{i}.gin'}]\n"
      )
      pass

  # split val datasets across processes
  process_idx = int(os.environ.get('CONDOR_Process', '0'))
  world_size = int(os.environ.get('CONDOR_WORLD_SIZE', '1'))
  runner.split_val_sets_across_processes(process_idx=process_idx, world_size=world_size)
  runner.init_val_loaders()
  runner.accelerate_prepare()
  runner.eval()

  file_util.makedirs(out_dir, exist_ok=True)
  sample_counter = 0
  with torch.autocast('cuda', dtype=torch.float16):
    with torch.no_grad():
      for dataset_name, loader in zip(runner.dataset_names_val, runner.val_loaders):
        runner.logger.info(f"Processing dataset [{dataset_name}]...")
        for j, batch in enumerate(tqdm.tqdm(loader, desc=f"Dataset [{dataset_name}]")):          
          batch = data_util.MatchBatch(batch)
          batch.to_device(runner.device)
          B, H, W = batch.B, batch.H, batch.W
          outputs = runner.forward_image(
                batch,
                return_masked_gaussians=True,
            )
          masked_gaussian_parameters = outputs['masked_gaussians']

          ###
          # rendering camera sweep:
          ###
          
          # center gaussians
          masked_gaussian_parameters['xyz'] -= masked_gaussian_parameters['xyz'].mean(dim=-2, keepdim=True)

          # get camera parameters:
          fxfycxcy_sweep = sweep_intrinsics(batch['fxfycxcy'])
          C2W_sweep = generate_camera_sweep_extrinsics(
            vert_range_deg=angle_range_vert,   # (min, max) vertical tilt (x-axis rotation)
            horiz_range_deg=angle_range_hor, # (min, max) horizontal pan (y-axis rotation)
            expand_periods = 0.5,
            reduce_periods = 0.5,
            full_periods = 1,
            num_steps=nframes,
          )
          C2W_sweep = torch.from_numpy(C2W_sweep).float().cuda()

          for i_cam in tqdm.tqdm(range(len(C2W_sweep)), desc='Render Sweep', leave=False):
            C2W = C2W_sweep[i_cam][None, None].repeat(B, 1, 1, 1)
            fxfycxcy = fxfycxcy_sweep.repeat(B, 1, 1)
            render_img = runner.render_gaussians(
                  masked_gaussian_parameters,
                  C2W,
                  fxfycxcy,
                  height=H,
                  width=W,
                  bg_color=(1.0, 1.0, 1.0),
              )["image"]
            for i_b in range(B):
              out_path_pred = out_dir / f'{sample_counter+i_b:06d}_sweep_frames/{i_cam:03d}.jpg'
              out_path_pred.parent.mkdir(parents=True, exist_ok=True)
              save_image(render_img[i_b,0], out_path_pred)

          # writing images
          for i_b in range(B):
            out_path_pred = out_dir / f'{sample_counter+i_b:06d}_pred.jpg'
            out_path_input = out_dir / f'{sample_counter+i_b:06d}_input.jpg'
            out_path_pred.parent.mkdir(parents=True, exist_ok=True)
            out_path_input.parent.mkdir(parents=True, exist_ok=True)
            save_image(make_grid(outputs['image'][i_b], nrow=4),out_path_pred)
            save_image(make_grid(batch['image'][i_b], nrow=4),out_path_input)

          # writing video
          for i_b in range(B):
            sweep_frames_dir = out_dir / f'{sample_counter+i_b:06d}_sweep_frames'
            sweep_path = out_dir / f'{sample_counter+i_b:06d}_sweep.mp4'
            sweep_path.parent.mkdir(parents=True, exist_ok=True)
            frames_to_video(sweep_frames_dir, sweep_path, fps=fps)
          sample_counter += B  


def main(argv: list[str]|None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gin_configs", action="append", default=[])
    parser.add_argument("--gin_bindings", action="append", default=[])
    FLAGS = parser.parse_args(argv)

    # explicitly initialize the warm pool as the first line of main!
    gin.parse_config_files_and_bindings(
        config_files=FLAGS.gin_configs, bindings=None, skip_unknown=True
    )
    opt = Options()
    runner = MatchRunner(opt)
    save_camera_sweeps(runner, store_config_files=FLAGS.gin_configs)
    runner.graceful_exit(0)



if __name__ == "__main__":
  main()
