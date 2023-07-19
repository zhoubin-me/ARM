from typing import List, Any

import torch

from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, HistogramSummary, ImageSummary
from tensorboard.plugins.mesh import summary_v2 as mesh_summary
from einops import rearrange
from dataclasses import dataclass
import numpy as np
from kornia.geometry.depth import depth_to_3d

@dataclass
class PointCloud:
    rgb: Any = None
    pcd: Any = None
    pcd_depth: Any = None
    name: str = "pcd"

class PreprocessAgent(Agent):

    def __init__(self,
                 pose_agent: Agent):
        self._pose_agent = pose_agent
        self._depth_near = 0.01
        self._depth_far = 4.5

    def build(self, training: bool, device: torch.device = None):
        self._pose_agent.build(training, device)

    def _norm_rgb_(self, x):
        return (x.float() / 255.0) * 2.0 - 1.0

    def calc_pcd_from_depth(self, replay_sample: dict) -> dict:
        came_name = 'wrist'
        # pcd = replay_sample['wrist_point_cloud'][:, 0]
        depth = replay_sample[f'{came_name}_depth'][:, 0]
        camera_intr = replay_sample[f'{came_name}_camera_intrinsics'][:, 0]
        camera_extr = replay_sample[f'{came_name}_camera_extrinsics'][:, 0]

        pcd_depth = depth_to_3d(depth, camera_intr)
        b, _, h, w = pcd_depth.shape
        pcd_depth = rearrange(pcd_depth, 'b c h w -> b c (h w)')
        pcd_depth = torch.cat((pcd_depth, torch.ones(b, 1, h * w).to(depth)), dim=1)
        pcd_depth = torch.bmm(torch.inverse(camera_extr), pcd_depth)
        pcd_depth = rearrange(pcd_depth[:, :3], 'b c (h w) -> b h w c', h=h, w=w)
        replay_sample[f'{came_name}_point_cloud'] = pcd_depth.unsqueeze(1).float()

        if 'tp1' in replay_sample:
            depth_tp1 = replay_sample[f'{came_name}_depth_tp1'][:, 0]
            camera_intr_tp1 = replay_sample[f'{came_name}_camera_intrinsics_tp1'][:, 0]
            camera_extr_tp1 = replay_sample[f'{came_name}_camera_extrinsics_tp1'][:, 0]

            pcd_depth_tp1 = depth_to_3d(depth_tp1, camera_intr_tp1)
            pcd_depth_tp1 = rearrange(pcd_depth_tp1, 'b c h w -> b c (h w)')
            pcd_depth_tp1 = torch.cat((pcd_depth_tp1, torch.ones(b, 1, h * w).to(depth)), dim=1)
            pcd_depth_tp1 = torch.bmm(torch.inverse(camera_extr_tp1), pcd_depth_tp1)
            pcd_depth_tp1 = rearrange(pcd_depth_tp1[:, :3], 'b c (h w) -> b h w c', h=h, w=w)
            replay_sample[f'{came_name}_point_cloud_tp1'] = pcd_depth_tp1.unsqueeze(1).float()
            
        return replay_sample
    
    def update(self, step: int, replay_sample: dict) -> dict:
        # Samples are (B, N, ...) where N is number of buffers/tasks. This is a single task setup, so 0 index.
        replay_sample = {k: v[:, 0] for k, v in replay_sample.items()}
        replay_sample = self.calc_pcd_from_depth(replay_sample)

        for k, v in replay_sample.items():
            if 'rgb' in k:
                replay_sample[k] = self._norm_rgb_(v)
        self._replay_sample = replay_sample
        return self._pose_agent.update(step, replay_sample)

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        # observation = {k: torch.tensor(v) for k, v in observation.items()}
        observation = self.calc_pcd_from_depth(observation)
        for k, v in observation.items():
            if 'rgb' in k:
                observation[k] = self._norm_rgb_(v)
        act_res = self._pose_agent.act(step, observation, deterministic)
        act_res.replay_elements.update({'demo': False})
        return act_res

    def update_summaries(self) -> List[Summary]:
        prefix = 'inputs'
        demo_f = self._replay_sample['demo'].float()
        demo_proportion = demo_f.mean()
        tile = lambda x: torch.squeeze(
            torch.cat(x.split(1, dim=1), dim=-1), dim=1)
        for cam_name in ['front', 'wrist']:
            cam_name_rgb = cam_name + '_rgb'
            if cam_name_rgb not in self._replay_sample: continue
            rgb = self._replay_sample[f'{cam_name}_rgb'][0][0]
            rgb = ((rgb + 1.0) / 2.0 * 255.0).to(torch.uint8)
            pcd = self._replay_sample[f'{cam_name}_point_cloud'][0][0]
            depth = self._replay_sample[f'{cam_name}_depth'][0][:1]

            camera_intr = self._replay_sample[f'{cam_name}_camera_intrinsics'][0]
            # camera_extr = self._replay_sample[f'{cam_name}_camera_extrinsics'][0]
            pcd_depth = depth_to_3d(depth, camera_intr, normalize_points=False).squeeze(0)
            rgb = rearrange(rgb, 'c h w -> (h w) c')
            pcd = rearrange(pcd, 'c h w -> (h w) c')
            pcd_depth = rearrange(pcd_depth, 'c h w -> (h w) c')
            break

        sums = [
            PointCloud(rgb=rgb, pcd=pcd, pcd_depth=pcd_depth, name=f"{prefix}/point_cloud"),
            ScalarSummary('%s/demo_proportion' % prefix, demo_proportion),
            HistogramSummary('%s/low_dim_state' % prefix,
                    self._replay_sample['low_dim_state']),
            HistogramSummary('%s/low_dim_state_tp1' % prefix,
                    self._replay_sample['low_dim_state_tp1']),
            ScalarSummary('%s/low_dim_state_mean' % prefix,
                    self._replay_sample['low_dim_state'].mean()),
            ScalarSummary('%s/low_dim_state_min' % prefix,
                    self._replay_sample['low_dim_state'].min()),
            ScalarSummary('%s/low_dim_state_max' % prefix,
                    self._replay_sample['low_dim_state'].max()),
            ScalarSummary('%s/timeouts' % prefix,
                    self._replay_sample['timeout'].float().mean()),
        ]

        for k, v in self._replay_sample.items():
            if 'rgb' in k or 'point_cloud' in k:
                if 'rgb' in k:
                    # Convert back to 0 - 1
                    v = (v + 1.0) / 2.0
                sums.append(ImageSummary('%s/%s' % (prefix, k), tile(v)))

        if 'sampling_probabilities' in self._replay_sample:
            sums.extend([
                HistogramSummary('replay/priority',
                                 self._replay_sample['sampling_probabilities']),
            ])
        sums.extend(self._pose_agent.update_summaries())
        return sums

    def act_summaries(self) -> List[Summary]:
        return self._pose_agent.act_summaries()

    def load_weights(self, savedir: str):
        self._pose_agent.load_weights(savedir)

    def save_weights(self, savedir: str):
        self._pose_agent.save_weights(savedir)

    def reset(self) -> None:
        self._pose_agent.reset()

