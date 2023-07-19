from typing import List, Any

import torch

from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, HistogramSummary, ImageSummary
from tensorboard.plugins.mesh import summary_v2 as mesh_summary
from einops import rearrange
from dataclasses import dataclass

@dataclass
class PointCloud:
    rgb: Any = None
    pcd: Any = None
    name: str = "pcd"

class PreprocessAgent(Agent):

    def __init__(self,
                 pose_agent: Agent):
        self._pose_agent = pose_agent

    def build(self, training: bool, device: torch.device = None):
        self._pose_agent.build(training, device)

    def _norm_rgb_(self, x):
        return (x.float() / 255.0) * 2.0 - 1.0

    def update(self, step: int, replay_sample: dict) -> dict:
        # Samples are (B, N, ...) where N is number of buffers/tasks. This is a single task setup, so 0 index.
        replay_sample = {k: v[:, 0] for k, v in replay_sample.items()}
        for k, v in replay_sample.items():
            if 'rgb' in k:
                replay_sample[k] = self._norm_rgb_(v)
        self._replay_sample = replay_sample
        return self._pose_agent.update(step, replay_sample)

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        # observation = {k: torch.tensor(v) for k, v in observation.items()}
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
        rgb = self._replay_sample['front_rgb'][0][0]
        rgb = (rgb + 1.0) / 2.0
        pcd = self._replay_sample['front_point_cloud'][0][0]
        rgb = rearrange(rgb, 'c h w -> (h w) c')
        pcd = rearrange(pcd, 'c h w -> (h w) c')
        sums = [
            PointCloud(rgb=rgb, pcd=pcd, name=f"{prefix}/point_cloud"),
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

