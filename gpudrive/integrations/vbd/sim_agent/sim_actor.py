import torch
from torch import nn
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm
from gpudrive.integrations.vbd.model.VBD import VBD
from gpudrive.integrations.vbd.sim_agent.utils import *
from gpudrive.integrations.vbd.model.model_utils import roll_out
from gpudrive.integrations.vbd.sim_agent.guidance_metrics import *


class VBDTest(VBD):
    def __init__(
        self,
        cfg: dict,
        early_stop: int = 0,
        skip: int = 1,
        reward_func: nn.Module = None,
        guidance_iter: int = 5,
        guidance_end: int = 1,
        guidance_start: int = 99,
        gradient_scale: float = 1.0,
        scale_grad_by_std: bool = True,
        guide_mode: str = "waymo",
    ):
        """
        Initializes the SimActor object.
        Args:
            cfg (dict): Configuration dictionary.
            early_stop (int, optional): Early stop parameter. Defaults to 0.
            skip (int, optional): Skip diffusion step. Defaults to 1.
            reward_func (nn.Module, optional): Reward function module. Defaults to None.
            guidance_iter (int, optional): Guidance iteration parameter. Defaults to 5.
            guidance_end (int, optional): Guidance end parameter. Defaults to 1.
            guidance_start (int, optional): Guidance start parameter. Defaults to 99.
            gradient_scale (float, optional): Gradient scale parameter. Defaults to 1.0.
            scale_grad_by_std (bool, optional): Flag to scale gradient by standard deviation. Defaults to True.
            guide_mode (str, optional): Guidance mode. Defaults to 'waymo'.
        """
        super().__init__(cfg)

        # Parameters for the denoiser sampling
        self.early_stop = early_stop
        self.skip = skip

        # Parameters for the guidance function
        self.reward_func = reward_func
        self.guidance_iter = guidance_iter
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.gradient_scale = gradient_scale
        self.scale_grad_by_std = scale_grad_by_std

        if guide_mode == "waymo":
            self.guidance_func = self.waymo_guidance
        elif guide_mode == "ctg":
            self.guidance_func = self.ctg_guidance

    def reset_agent_length(self, _agents_len) -> None:
        """
        Resets the number of the agent.

        Args:
            _agents_len (int): The new length of the agent.

        Returns:
            None
        """

        self._agents_len = _agents_len
        if self.predictor is not None:
            self.predictor.reset_agent_length(_agents_len)

        if self.denoiser is not None:
            self.denoiser.reset_agent_length(_agents_len)

    ################### Testing Setup ###################
    def inference_predictor(self, batch) -> Dict[str, torch.Tensor]:
        """
        Perform inference using the predictor model.

        Args:
            batch: The input batch for inference.

        Returns:
            The output of the predictor model.
        """
        if self.predictor is None:
            raise RuntimeError("Predictor is not defined")

        batch = self.batch_to_device(batch, self.device)
        encoder_outputs = self.encoder(batch)
        goal_outputs = self.forward_predictor(encoder_outputs)

        return goal_outputs

    ################### Guidance ###################
    def ctg_guidance(
        self, x_t: torch.Tensor, c: dict, t: int, **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, np.ndarray]]:
        """
        Performs guidance for the simulation actor using the method proposed by Controllable Traffic Generation.
        Args:
            x_t (torch.Tensor): The input tensor representing the normalized actions.
            c (dict): The dictionary containing the encoder outputs.
            t (int): The current timestep.
            **kwargs: Additional keyword arguments.
        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, np.ndarray]]: A tuple containing the denoiser output, the previous actions, and the guidance history.
        Raises:
            None
        """

        mu_guide = []
        traj_guide = []
        reward_guide = []
        grad_guide = []

        # Denoise and step
        denoiser_output = self.forward_denoiser(
            encoder_outputs=c,
            noised_actions_normalized=x_t,
            diffusion_step=t,
        )

        x_0 = denoiser_output["denoised_actions_normalized"]

        mu = self.noise_scheduler.q_mean(
            model_output=x_0,
            timestep=t,
            sample=x_t,
        )
        mu_guide.append(mu.detach().cpu().numpy())
        std = self.noise_scheduler.q_variance(t) ** 0.5
        mu = mu.detach()

        with torch.enable_grad():
            mu.requires_grad_()

            if self.scale_grad_by_std:
                lr = std * self.gradient_scale
            else:
                lr = self.gradient_scale

            optimizer = torch.optim.Adam([mu], lr=lr)

            for _ in range(self.guidance_iter):
                optimizer.zero_grad()
                noised_actions = self.unnormalize_actions(mu)
                currnet_states = c["agents"][:, : self._agents_len, -1]
                noised_trajs = roll_out(
                    currnet_states,
                    noised_actions,
                    action_len=self.denoiser._action_len,
                    global_frame=True,
                )

                total_rewards = 0
                for reward_func in self.reward_func:
                    reward = reward_func(
                        traj_pred=noised_trajs,
                        action_pred=noised_actions,
                        c=c,
                        **kwargs
                    )
                    total_rewards += reward.sum()

                cost = -1.0 * total_rewards
                cost.backward()
                grad = mu.grad
                optimizer.step()

                mu_guide.append(mu.detach().cpu().numpy())
                traj_guide.append(noised_trajs.detach().cpu().numpy())
                reward_guide.append(-cost.detach().cpu().numpy())
                grad_guide.append(grad.detach().cpu().numpy())

        mu = mu.detach()
        noise = torch.randn(mu.shape).type_as(mu)
        x_t_prev = mu + noise * std

        guide_history = {
            "t_guide": np.array(t),
            "mu_guide": np.stack(mu_guide, axis=0),
            "traj_guide": np.stack(traj_guide, axis=0),
            "reward_guide": np.stack(reward_guide, axis=0),
            "grad_guide": np.stack(grad_guide, axis=0),
        }

        return denoiser_output, x_t_prev, guide_history

    def waymo_guidance(
        self, x_t: torch.Tensor, c: dict, t: int, **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, np.ndarray]]:
        """
        Perform guidance for the given input based on Motion Diffuser paper.
        Args:
            x_t (torch.Tensor): The input tensor.
            c (dict): The dictionary of encoder outputs.
            t (int): The diffusion step.
            **kwargs: Additional keyword arguments.
        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, np.ndarray]]: A tuple containing:
                - denoiser_output (Dict[str, torch.Tensor]): The output of the denoiser.
                - x_t_prev (torch.Tensor): The previous input tensor.
                - guide_history (Dict[str, np.ndarray]): The history of guidance.
        Raises:
            None
        """

        # Save History
        mu_guide = []
        x_0_guide = []
        traj_guide = []
        reward_guide = []
        grad_guide = []

        # Denoise and step
        denoiser_output = self.forward_denoiser(
            encoder_outputs=c,
            noised_actions_normalized=x_t,
            diffusion_step=t,
        )

        x_0 = denoiser_output["denoised_actions_normalized"]

        mu = self.noise_scheduler.q_mean(
            model_output=x_0,
            timestep=t,
            sample=x_t,
        ).detach()

        mu_guide.append(mu.detach().cpu().numpy())
        std = self.noise_scheduler.q_variance(t) ** 0.5

        for _ in range(self.guidance_iter):
            with torch.enable_grad():
                mu.requires_grad_()

                guidance_denoiser_output = self.forward_denoiser(
                    encoder_outputs=c,
                    noised_actions_normalized=mu,
                    diffusion_step=t - 1,
                )

                traj_pred = guidance_denoiser_output["denoised_trajs"]
                action_pred = guidance_denoiser_output["denoised_actions"]
                action_pred_normalized = guidance_denoiser_output[
                    "denoised_actions_normalized"
                ]

                total_rewards = 0
                for reward_func in self.reward_func:
                    reward = reward_func(
                        traj_pred=traj_pred,
                        action_pred=action_pred,
                        action_normalized=action_pred_normalized,
                        c=c,
                        **kwargs
                    )
                    total_rewards += reward.sum()
                print("round: ", _, "reward: ", total_rewards.item())
                grad = torch.autograd.grad([total_rewards], [mu])[0]

                if self.scale_grad_by_std:
                    grad = grad * (std)

                mu = mu.detach() + grad.detach() * self.gradient_scale

                mu_guide.append(mu.detach().cpu().numpy())
                x_0_guide.append(action_pred_normalized.detach().cpu().numpy())
                traj_guide.append(traj_pred.detach().cpu().numpy())
                reward_guide.append(reward.detach().cpu().numpy())
                grad_guide.append(grad.detach().cpu().numpy())

        noise = torch.randn(mu.shape).type_as(mu)
        x_t_prev = mu + noise * std

        guide_history = {
            "t_guide": np.array(t),
            "mu_guide": np.stack(mu_guide, axis=0),
            "x_0_guide": np.stack(x_0_guide, axis=0),
            "traj_guide": np.stack(traj_guide, axis=0),
            "reward_guide": np.stack(reward_guide, axis=0),
            "grad_guide": np.stack(grad_guide, axis=0),
        }

        return denoiser_output, x_t_prev, guide_history

    def ibr_guidance(
        self,
        x_t: torch.Tensor,
        c: dict,
        t: int,
        ego_idx: int,
        adv_idx: int,
        other_idx: list = None,
        ego_iter: int = 5,
        adv_iter=5,
        t_react: int = 81,
        roadgraph_points=None,
        adv_use_ctg: bool = False,
        ego_use_ctg: bool = False,
        **kwargs
    ):

        """
        Perform Iterative Best Response (IBR) guidance for a simulated actor.
        Args:
            x_t (torch.Tensor): The input tensor representing the current state.
            c (dict): A dictionary containing encoder outputs.
            t (int): The current time step.
            ego_idx (int): The index of the ego agent.
            adv_idx (int): The index of the adversary agent.
            other_idx (list, optional): A list of indices of other agents. Defaults to None.
            ego_iter (int, optional): The number of iterations for the ego agent. Defaults to 5.
            adv_iter (int, optional): The number of iterations for the adversary agent. Defaults to 5.
            t_react (int, optional): The time step at which the agent starts reacting. Defaults to 81.
            roadgraph_points (None, optional): Points representing the road graph. Defaults to None.
            adv_use_ctg (bool, optional): Whether to use the cost-to-go (CTG) method for the adversary agent. Defaults to False.
            ego_use_ctg (bool, optional): Whether to use the cost-to-go (CTG) method for the ego agent. Defaults to False.
            **kwargs: Additional keyword arguments.
        Returns:
            Tuple: A tuple containing the denoiser output, the previous state, and the guidance history.

        vgd.guidance_iter = 5
        vgd.guidance_start = 99
        vgd.guidance_end = 1
        vgd.gradient_scale = 0.1
        vgd.scale_grad_by_std = True
        """
        # Save History
        mu_guide = []
        x_0_guide = []
        traj_guide = []
        reward_guide = []
        grad_guide = []

        # Denoise and step
        denoiser_output = self.forward_denoiser(
            encoder_outputs=c,
            noised_actions_normalized=x_t,
            diffusion_step=t,
        )

        x_0 = denoiser_output["denoised_actions_normalized"]

        mu = self.noise_scheduler.q_mean(
            model_output=x_0,
            timestep=t,
            sample=x_t,
        )

        mu_guide.append(mu.detach().cpu().numpy())
        std = self.noise_scheduler.q_variance(t) ** 0.5

        for iter_round in range(self.guidance_iter):
            for iter in range(ego_iter + adv_iter):
                with torch.enable_grad():
                    mu = mu.detach()
                    mu.requires_grad_()
                    if (iter < adv_iter and adv_use_ctg) or (
                        iter >= adv_iter and ego_use_ctg
                    ):
                        method = "ctg"
                        noised_actions = self.unnormalize_actions(mu)
                        currnet_states = c["agents"][:, : self._agents_len, -1]
                        traj_pred = roll_out(
                            currnet_states,
                            noised_actions,
                            action_len=self.denoiser._action_len,
                            global_frame=True,
                        )
                        gradient_scale = 0.1 * self.gradient_scale
                    else:
                        method = "waymo"
                        guidance_denoiser_output = self.forward_denoiser(
                            encoder_outputs=c,
                            noised_actions_normalized=mu,
                            diffusion_step=t - 1,
                        )
                        traj_pred = guidance_denoiser_output["denoised_trajs"]
                        gradient_scale = self.gradient_scale

                    if iter < adv_iter:
                        pursue_reward = -OverlapReward(
                            saturate=False,
                        ).forward(
                            traj_pred=traj_pred,
                            c=c,
                            aoi=[adv_idx, ego_idx],
                            **kwargs
                        )  # [B, A, T, A]
                        pursue_reward = torch.max(
                            pursue_reward[:, 0, :, 1], dim=-1
                        )[
                            0
                        ]  # [B, T]

                        # Positive reward for stay on road
                        onroad_reward = OnroadReward(weight=2).forward(
                            traj_pred=traj_pred,
                            c=c,
                            aoi=[adv_idx],
                            roadgraph_points=roadgraph_points,
                        )  # [B, 1, T]

                        onroad_reward = torch.mean(
                            onroad_reward, dim=-1
                        )  # [B]

                        # Only care about getting close to adv
                        grad_mask = torch.zeros_like(mu)  # [B, A, T, D]
                        grad_mask[:, adv_idx, :t_react, :] = 1

                        # Do a softmin over time
                        reward = pursue_reward + onroad_reward  # [B]
                        print(
                            method,
                            " Adv round: ",
                            iter_round,
                            ", step: ",
                            iter,
                            "pursue reward: ",
                            round(pursue_reward.mean().item(), 3),
                            "onroad_reward: ",
                            round(onroad_reward.mean().item(), 3),
                        )
                    else:
                        if other_idx is None:
                            aoi = None
                            ego_i = ego_idx
                            adv_i = adv_idx
                        else:
                            aoi = [adv_idx, ego_idx] + other_idx
                            ego_i = 1
                            adv_i = 0

                        offset = 0.5
                        # Ego Plays second
                        evasion_reward = OverlapReward(
                            offset=offset,
                            weight=1.0,
                            saturate=True,
                        ).forward(
                            traj_pred=traj_pred, c=c, aoi=aoi, **kwargs
                        )  # [B, A, T, A]

                        evasion_reward[
                            :, adv_i, :t_react, :
                        ] = 100  # adv Ignore collision before t_react
                        evasion_reward = evasion_reward.reshape(
                            *evasion_reward.shape[:2], -1
                        )  # [B, A, T*A]
                        evasion_reward_min = torch.min(evasion_reward, dim=-1)[
                            0
                        ]  # [B, A]
                        onroad_reward = OnroadReward().forward(
                            traj_pred=traj_pred,
                            c=c,
                            aoi=aoi,
                            roadgraph_points=roadgraph_points,
                        )  # [B, A, T]
                        onroad_reward = torch.mean(
                            onroad_reward, dim=-1
                        )  # [B, A]
                        # onroad_reward = torch.min(onroad_reward, dim=-1)[0]

                        grad_mask = torch.ones_like(mu)  # [B, A, T, D]
                        grad_mask[:, adv_idx, :t_react, :] = 0

                        # Do a softmin over time and other agents
                        reward = onroad_reward + evasion_reward_min * 15

                        print(
                            method,
                            " ego round: ",
                            iter_round,
                            ", step: ",
                            iter - adv_iter,
                            "ego evasion reward: ",
                            round(
                                evasion_reward_min[0, ego_i].item() + offset, 3
                            ),
                            "adv evasion reward: ",
                            round(
                                evasion_reward_min[0, adv_i].item() + offset, 3
                            ),
                            "onroad reward: ",
                            round(onroad_reward[0, ego_i].item(), 3),
                        )

                    grad = torch.autograd.grad([reward.sum()], [mu])[0]
                    grad = grad * grad_mask
                    if self.scale_grad_by_std:
                        grad = grad * (std)
                    mu = mu.detach() + grad.detach() * gradient_scale
                    # Clip the mu
                    mu = torch.clamp(mu, -self._action_max, self._action_max)

                    mu_guide.append(mu.detach().cpu().numpy())
                    # x_0_guide.append(guidance_denoiser_output['denoised_actions_normalized'].detach().cpu().numpy())
                    traj_guide.append(traj_pred.detach().cpu().numpy())
                    # reward_guide.append(reward.detach().cpu().numpy())
                    grad_guide.append(grad.detach().cpu().numpy())

        noise = torch.randn(mu.shape).type_as(mu)
        x_t_prev = mu + noise * std

        guide_history = {
            "t_guide": np.array(t),
            "mu_guide": np.stack(mu_guide, axis=0),
            # 'x_0_guide': np.stack(x_0_guide, axis=0),
            "traj_guide": np.stack(traj_guide, axis=0),
            # 'reward_guide': np.stack(reward_guide, axis=0),
            "grad_guide": np.stack(grad_guide, axis=0),
        }

        return denoiser_output, x_t_prev, guide_history

    ################### Denoising ###################
    def step_denoiser(
        self, x_t: torch.Tensor, c: dict, t: int, global_frame: bool = True
    ):
        """
        Perform a denoising step to sample x_{t-1} ~ P[x_{t-1} | x_t, D(x_t, c, t)].

        Args:
            x_t (torch.Tensor): The input tensor representing the current state. Shape: (num_batch, num_agent, num_action, action_dim)
            c (dict): The conditional variable dictionary.
            t (int): The number of diffusion steps.

        Returns:
            denoiser_output (dict): The denoiser outputs.
            x_t_prev (torch.Tensor): The tensor representing the previous noised action. Shape: (num_batch, num_agent, num_action, action_dim)
        """

        if self.denoiser is None:
            raise RuntimeError("Denoiser is not defined")

        # Denoise to reconstruct x_0 ~ D(x_t, c, t)
        denoiser_output = self.forward_denoiser(
            encoder_outputs=c,
            noised_actions_normalized=x_t,
            diffusion_step=t,
            global_frame=global_frame,
        )

        x_0 = denoiser_output["denoised_actions_normalized"]

        # Step to sample from P(x_t-1 | x_t, x_0)
        x_t_prev = self.noise_scheduler.step(
            model_output=x_0,
            timestep=t,
            sample=x_t,
        )

        return denoiser_output, x_t_prev

    @torch.no_grad()
    def sample_denoiser(
        self,
        batch,
        num_samples=1,
        x_t=None,
        use_tqdm=True,
        global_frame=True,
        **kwargs
    ):
        """
        Perform denoising inference on the given batch of data.

        Args:
            batch (dict): The input batch of data.
            guidance_func (callable, optional): A callable function that provides guidance for denoising. Defaults to None.
            early_stop (int, optional): The index of the step at which denoising should stop. Defaults to 0.
            skip (int, optional): The number of steps to skip between denoising iterations. Defaults to 1.
            **kwargs: Additional keyword arguments for guidance.
        Returns:
            dict: The denoising outputs, including the history of noised action normalization.

        """
        # Encode the scene
        batch = self.batch_to_device(batch, self.device)
        encoder_outputs = self.encoder(batch)

        if num_samples > 1:
            encoder_outputs = duplicate_batch(encoder_outputs, num_samples)

        agents_history = encoder_outputs["agents"]
        num_batch, num_agent = agents_history.shape[:2]
        num_step = self._future_len // self._action_len
        action_dim = 2

        diffusion_steps = list(
            reversed(
                range(
                    self.early_stop, self.noise_scheduler.num_steps, self.skip
                )
            )
        )

        # History
        x_t_history = []
        denoiser_output_history = []
        guide_history = []

        # Inital X_T
        if x_t is None:
            x_t = torch.randn(
                num_batch, num_agent, num_step, action_dim, device=self.device
            )
        else:
            x_t = x_t.to(self.device)

        if use_tqdm:
            diffusion_steps = tqdm(
                diffusion_steps, total=len(diffusion_steps), desc="Diffusion"
            )

        for t in diffusion_steps:
            x_t_history.append(x_t.detach().cpu().numpy())

            if (
                t <= self.guidance_start
                and t >= self.guidance_end
                and self.guidance_func is not None
                and self.reward_func is not None
            ):

                denoiser_output, x_t, guide = self.guidance_func(
                    x_t=x_t, c=encoder_outputs, t=t, **kwargs
                )
                guide_history.append(guide)
            else:
                denoiser_output, x_t = self.step_denoiser(
                    x_t=x_t,
                    c=encoder_outputs,
                    t=t,
                    global_frame=global_frame,
                )
                guide = None

            denoiser_output_history.append(
                torch_dict_to_numpy(denoiser_output)
            )

        denoiser_output["history"] = {
            "x_t_history": np.stack(x_t_history, axis=0),
            "denoiser_output_history": stack_dict(denoiser_output_history),
            "guide_history": stack_dict(guide_history),
        }

        return denoiser_output
