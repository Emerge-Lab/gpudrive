from collections import deque
from typing import Dict

import torch

MetricName = str


class MetricsLogger:
    def __init__(self, cfg, max_len=1000):
        self.max_len = max_len
        self.metrics: Dict[MetricName, deque] = {}
        # TODO(ev) fix and make attributes
        self.ep_reward = torch.zeros((cfg["num_sims"], cfg["num_agents"]), dtype=torch.float32, device=cfg["device"])

    def step(self, ret_info):
        # go through all the done episodes in the step
        # and store which player won
        stats = {}
        self.ep_reward += ret_info.reward * ret_info.valids

        if torch.any(ret_info.done):
            stats["ep_rew"] = self.ep_reward[ret_info.done].sum().cpu().item()
            self.ep_reward[ret_info.done] = 0
            stats["num_episodes"] = ret_info.done.sum().cpu().item()
            # stats["on_goal"] = ret_info.on_goal[ret_info.done].sum().cpu().item()
            # stats["colls"] = ret_info.colls[ret_info.done].sum().cpu().item()
            # stats['episode_len'] = ret_info.episode_len[ret_info.done].sum().cpu().item()

        for k, v in stats.items():
            if k not in self.metrics:
                self.metrics[k] = deque(maxlen=self.max_len)
            self.metrics[k].append(v)

    def get_stats(self):
        stats = {}
        # TODO(ev) handle the invalid keys more elegantly
        for k, v in self.metrics.items():
            if k == "num_episodes":
                stats[k] = sum(self.metrics[k])
            else:
                stats[k] = sum(self.metrics[k]) / sum(self.metrics["num_episodes"])
        return stats