import torch

# import lightning.pytorch as pl
import pytorch_lightning as pl

from .modules import Encoder, Denoiser, GoalPredictor
from .utils import DDPM_Sampler
from .model_utils import (
    inverse_kinematics,
    roll_out,
    batch_transform_trajs_to_global_frame,
)
from torch.nn.functional import smooth_l1_loss, cross_entropy


class VBD(pl.LightningModule):
    """
    Versertile prior-guided diffusion model.
    """

    def __init__(
        self,
        cfg: dict,
    ):
        """
        Initialize the VPD model.

        Args:
            cfg (dict): Configuration parameters for the model.
        """
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self._future_len = cfg["future_len"]
        self._agents_len = cfg["agents_len"]
        self._action_len = cfg["action_len"]
        self._diffusion_steps = cfg["diffusion_steps"]
        self._encoder_layers = cfg["encoder_layers"]
        self._action_mean = cfg["action_mean"]
        self._action_std = cfg["action_std"]

        self._train_encoder = cfg.get("train_encoder", True)
        self._train_denoiser = cfg.get("train_denoiser", True)
        self._train_predictor = cfg.get("train_predictor", True)
        self._with_predictor = cfg.get("with_predictor", True)

        self.encoder = Encoder(self._encoder_layers)

        self.denoiser = Denoiser(
            future_len=self._future_len,
            action_len=self._action_len,
            agents_len=self._agents_len,
            steps=self._diffusion_steps,
        )
        if self._with_predictor:
            self.predictor = GoalPredictor(
                future_len=self._future_len,
                agents_len=self._agents_len,
                action_len=self._action_len,
            )
        else:
            self.predictor = None
            self._train_predictor = False

        self.noise_scheduler = DDPM_Sampler(steps=self._diffusion_steps)

        self.register_buffer("action_mean", torch.tensor(self._action_mean))
        self.register_buffer("action_std", torch.tensor(self._action_std))

    ################### Training Setup ###################
    def configure_optimizers(self):
        """
        This function is called by Lightning to create the optimizer and learning rate scheduler.
        """
        if not self._train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if not self._train_denoiser:
            for param in self.denoiser.parameters():
                param.requires_grad = False
        if self._with_predictor and (not self._train_predictor):
            for param in self.predictor.parameters():
                param.requires_grad = False

        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        assert len(params_to_update) > 0, "No parameters to update"

        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )

        lr_warmpup_step = self.cfg["lr_warmup_step"]
        lr_step_freq = self.cfg["lr_step_freq"]
        lr_step_gamma = self.cfg["lr_step_gamma"]

        def lr_update(step, warmup_step, step_size, gamma):
            if step < warmup_step:
                # warm up lr
                lr_scale = 1 - (warmup_step - step) / warmup_step * 0.95
            else:
                n = (step - warmup_step) // step_size
                lr_scale = gamma**n

            if lr_scale < 1e-2:
                lr_scale = 1e-2
            elif lr_scale > 1:
                lr_scale = 1

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_update(
                step,
                lr_warmpup_step,
                lr_step_freq,
                lr_step_gamma,
            ),
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, inputs, noised_actions_normalized, diffusion_step):
        """
        Forward pass of the VPD model.

        Args:
            inputs: Input data.
            noised_actions: noised actions.
            diffusion_step: Diffusion step.

        Returns:
            output_dict: Dictionary containing the model outputs.
        """
        # Encode scene
        output_dict = {}
        encoder_outputs = self.encoder(inputs)

        if self._train_denoiser:
            denoiser_outputs = self.forward_denoiser(
                encoder_outputs, noised_actions_normalized, diffusion_step
            )
            output_dict.update(denoiser_outputs)

        if self._train_predictor:
            predictor_outputs = self.forward_predictor(encoder_outputs)
            output_dict.update(predictor_outputs)

        return output_dict

    def forward_denoiser(
        self,
        encoder_outputs,
        noised_actions_normalized,
        diffusion_step,
        global_frame=True,
    ):
        """
        Forward pass of the denoiser module.

        Args:
            encoder_outputs: Outputs from the encoder module.
            noised_actions: noised actions.
            diffusion_step: Diffusion step.

        Returns:
            denoiser_outputs: Dictionary containing the denoiser outputs.
        """
        noised_actions = self.unnormalize_actions(noised_actions_normalized)
        denoised_actions_normalized = self.denoiser(
            encoder_outputs, noised_actions, diffusion_step
        )
        current_states = encoder_outputs["agents"][:, : self._agents_len, -1]
        assert (
            encoder_outputs["agents"].shape[1] >= self._agents_len
        ), "Too many agents to consider"

        # Roll out
        denoised_actions = self.unnormalize_actions(
            denoised_actions_normalized
        )
        denoised_trajs = roll_out(
            current_states,
            denoised_actions,
            action_len=self.denoiser._action_len,
            global_frame=global_frame,
        )

        return {
            "denoised_actions_normalized": denoised_actions_normalized,
            "denoised_actions": denoised_actions,
            "denoised_trajs": denoised_trajs,
        }

    def forward_predictor(self, encoder_outputs):
        """
        Forward pass of the predictor module.

        Args:
            encoder_outputs: Outputs from the encoder module.

        Returns:
            predictor_outputs: Dictionary containing the predictor outputs.
        """
        # Predict goal
        goal_actions_normalized, goal_scores = self.predictor(encoder_outputs)

        current_states = encoder_outputs["agents"][:, : self._agents_len, -1]
        assert (
            encoder_outputs["agents"].shape[1] >= self._agents_len
        ), "Too many agents to consider"

        # Roll out
        goal_actions = self.unnormalize_actions(goal_actions_normalized)
        goal_trajs = roll_out(
            current_states[:, :, None, :],
            goal_actions,
            action_len=self.predictor._action_len,
            global_frame=True,
        )

        return {
            "goal_actions_normalized": goal_actions_normalized,
            "goal_actions": goal_actions,
            "goal_scores": goal_scores,
            "goal_trajs": goal_trajs,
        }

    def forward_and_get_loss(self, batch, prefix="", debug=False):
        """
        Forward pass of the model and compute the loss.

        Args:
            batch: Input batch.
            prefix: Prefix for the loss keys.
            debug: Flag to enable debug mode.

        Returns:
            total_loss: Total loss.
            log_dict: Dictionary containing the loss values.
            debug_outputs: Dictionary containing debug outputs.
        """
        # data inputs
        agents_future = batch["agents_future"][:, : self._agents_len]

        # TODO: Investigate why this to NAN
        # agents_future_valid = batch['agents_future_valid'][:, :self._agents_len]
        agents_future_valid = torch.ne(agents_future.sum(-1), 0)
        agents_interested = batch["agents_interested"][:, : self._agents_len]
        anchors = batch["anchors"][:, : self._agents_len]

        # get actions from trajectory
        gt_actions, gt_actions_valid = inverse_kinematics(
            agents_future,
            agents_future_valid,
            dt=0.1,
            action_len=self._action_len,
        )

        gt_actions_normalized = self.normalize_actions(gt_actions)
        B, A, T, D = gt_actions_normalized.shape

        log_dict = {}
        debug_outputs = {}
        total_loss = 0

        ############## Run Encoder ##############
        encoder_outputs = self.encoder(batch)

        ############### Denoise #################
        if self._train_denoiser:
            # sample noise
            noise = torch.randn(B * A, T, D).type_as(agents_future)

            diffusion_steps = (
                torch.randint(
                    0,
                    self.noise_scheduler.num_steps,
                    (B,),
                    device=agents_future.device,
                )
                .long()
                .unsqueeze(-1)
                .repeat(1, A)
            )

            # noise the input
            noised_action_normalized = self.noise_scheduler.add_noise(
                gt_actions_normalized.reshape(B * A, T, D),
                noise,
                diffusion_steps.reshape(B * A),
            ).reshape(B, A, T, D)

            denoise_outputs = self.forward_denoiser(
                encoder_outputs, noised_action_normalized, diffusion_steps
            )
            debug_outputs.update(denoise_outputs)

            # Get Loss
            denoised_trajs = denoise_outputs["denoised_trajs"]

            state_loss_mean, yaw_loss_mean = self.denoise_loss(
                denoised_trajs,
                agents_future,
                agents_future_valid,
                agents_interested,
            )

            denoise_ade, denoise_fde = self.calculate_metrics_denoise(
                denoised_trajs,
                agents_future,
                agents_future_valid,
                agents_interested,
                8,
            )

            denoise_loss = state_loss_mean + yaw_loss_mean
            total_loss += denoise_loss

            log_dict.update(
                {
                    prefix + "state_loss": state_loss_mean.item(),
                    prefix + "yaw_loss": yaw_loss_mean.item(),
                    prefix + "denoise_ADE": denoise_ade,
                    prefix + "denoise_FDE": denoise_fde,
                }
            )

        ############### Behavior Prior Prediction #################
        if self._train_predictor:
            goal_outputs = self.forward_predictor(encoder_outputs)
            debug_outputs.update(goal_outputs)

            # get loss
            goal_scores = goal_outputs["goal_scores"]
            goal_trajs = goal_outputs["goal_trajs"]

            goal_loss_mean, score_loss_mean = self.goal_loss(
                goal_trajs,
                goal_scores,
                agents_future,
                agents_future_valid,
                anchors,
                agents_interested,
            )

            pred_loss = goal_loss_mean + 0.05 * score_loss_mean
            total_loss += 50 * pred_loss  #!ZZX: 0.5 is the original weight

            pred_ade, pred_fde = self.calculate_metrics_predict(
                goal_trajs,
                agents_future,
                agents_future_valid,
                agents_interested,
                8,
            )

            log_dict.update(
                {
                    prefix + "goal_loss": goal_loss_mean.item(),
                    prefix + "score_loss": score_loss_mean.item(),
                    prefix + "pred_ADE": pred_ade,
                    prefix + "pred_FDE": pred_fde,
                }
            )

        log_dict[prefix + "loss"] = total_loss.item()

        if debug:
            return total_loss, log_dict, debug_outputs
        else:
            return total_loss, log_dict

    def training_step(self, batch, batch_idx):
        """
        Training step of the model.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            loss: Loss value.
        """
        # print("******************* training_step")

        # Add random mask to the history for dropout
        history_original = batch["agents_history"]
        B, A, T, C = history_original.shape
        history_mask = torch.rand(B, A) < 0.5
        history_mask = history_mask[:, :, None, None].repeat(1, 1, T, C)
        history_mask[:, :, -1, :] = 1
        history_mask = history_mask.type_as(history_original)
        batch["agents_history"] = history_original * history_mask

        loss, log_dict = self.forward_and_get_loss(batch, prefix="train/")
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.

        Args:
            batch: Input batch.
            batch_idx: Batch index.
        """
        loss, log_dict = self.forward_and_get_loss(batch, prefix="val/")
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss

    ################### Loss function ###################
    def denoise_loss(
        self,
        denoised_trajs,
        agents_future,
        agents_future_valid,
        agents_interested,
    ):
        """
        Calculates the denoise loss for the denoised actions and trajectories.

        Args:
            denoised_actions_normalized (torch.Tensor): Normalized denoised actions tensor of shape [B, A, T, C].
            denoised_trajs (torch.Tensor): Denoised trajectories tensor of shape [B, A, T, C].
            agents_future (torch.Tensor): Future agent positions tensor of shape [B, A, T, 3].
            agents_future_valid (torch.Tensor): Future agent validity tensor of shape [B, A, T].
            gt_actions_normalized (torch.Tensor): Normalized ground truth actions tensor of shape [B, A, T, C].
            gt_actions_valid (torch.Tensor): Ground truth actions validity tensor of shape [B, A, T].
            agents_interested (torch.Tensor): Interested agents tensor of shape [B, A].

        Returns:
            state_loss_mean (torch.Tensor): Mean state loss.
            yaw_loss_mean (torch.Tensor): Mean yaw loss.
            action_loss_mean (torch.Tensor): Mean action loss.
        """

        agents_future = agents_future[..., 1:, :3]
        future_mask = agents_future_valid[..., 1:] * (
            agents_interested[..., None] > 0
        )

        # Calculate State Loss
        # [B, A, T]
        state_loss = smooth_l1_loss(
            denoised_trajs[..., :2], agents_future[..., :2], reduction="none"
        ).sum(-1)
        yaw_error = denoised_trajs[..., 2] - agents_future[..., 2]
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
        yaw_loss = torch.abs(yaw_error)

        # Filter out the invalid state
        state_loss = state_loss * future_mask
        yaw_loss = yaw_loss * future_mask

        # Calculate the mean loss
        state_loss_mean = state_loss.sum() / future_mask.sum()
        yaw_loss_mean = yaw_loss.sum() / future_mask.sum()

        return state_loss_mean, yaw_loss_mean

    def goal_loss(
        self,
        trajs,
        scores,
        agents_future,
        agents_future_valid,
        anchors,
        agents_interested,
    ):
        """
        Calculates the loss for trajectory prediction.

        Args:
            trajs (torch.Tensor): Tensor of shape [B*A, Q, T, 3] representing predicted trajectories.
            scores (torch.Tensor): Tensor of shape [B*A, Q] representing predicted scores.
            agents_future (torch.Tensor): Tensor of shape [B, A, T, 3] representing future agent states.
            agents_future_valid (torch.Tensor): Tensor of shape [B, A, T] representing validity of future agent states.
            anchors (torch.Tensor): Tensor of shape [B, A, Q, 2] representing anchor points.
            agents_interested (torch.Tensor): Tensor of shape [B, A] representing interest in agents.

        Returns:
            traj_loss_mean (torch.Tensor): Mean trajectory loss.
            score_loss_mean (torch.Tensor): Mean score loss.
        """
        # Convert Anchor to Global Frame
        current_states = agents_future[:, :, 0, :3]
        anchors_global = batch_transform_trajs_to_global_frame(
            anchors, current_states
        )
        num_batch, num_agents, num_query, _ = anchors_global.shape

        # Get Mask
        traj_mask = agents_future_valid[..., 1:] * (
            agents_interested[..., None] > 0
        )  # [B, A, T]

        # Flatten batch and agents
        goal_gt = agents_future[:, :, -1:, :2].flatten(0, 1)  # [B*A, 1, 2]
        trajs_gt = agents_future[:, :, 1:, :3].flatten(0, 1)  # [B*A, T, 3]
        trajs = trajs.flatten(0, 1)[..., :3]  # [B*A, Q, T, 3]
        anchors_global = anchors_global.flatten(0, 1)  # [B*A, Q, 2]

        # Find the closest anchor
        idx_anchor = torch.argmin(
            torch.norm(anchors_global - goal_gt, dim=-1), dim=-1
        )  # [B*A,]

        # For agents that do not have valid end point, use the minADE
        dist = torch.norm(
            trajs[:, :, :, :2] - trajs_gt[:, None, :, :2], dim=-1
        )  # [B*A, Q, T]
        dist = dist * traj_mask.flatten(0, 1)[:, None, :]  # [B*A, Q, T]
        idx = torch.argmin(dist.mean(-1), dim=-1)  # [B*A,]

        # Select trajectory
        idx = torch.where(
            agents_future_valid[..., -1].flatten(0, 1), idx_anchor, idx
        )
        trajs_select = trajs[
            torch.arange(num_batch * num_agents), idx
        ]  # [B*A, T, 3]

        # Calculate the trajectory loss
        traj_loss = smooth_l1_loss(
            trajs_select, trajs_gt, reduction="none"
        ).sum(
            -1
        )  # [B*A, T]
        traj_loss = traj_loss * traj_mask.flatten(0, 1)  # [B*A, T]

        # Calculate the score loss
        scores = scores.flatten(0, 1)  # [B*A, Q]
        score_loss = cross_entropy(scores, idx, reduction="none")  # [B*A]
        score_loss = score_loss * (
            agents_interested.flatten(0, 1) > 0
        )  # [B*A]

        # Calculate the mean loss
        traj_loss_mean = traj_loss.sum() / traj_mask.sum()
        score_loss_mean = score_loss.sum() / (agents_interested > 0).sum()

        return traj_loss_mean, score_loss_mean

    @torch.no_grad()
    def calculate_metrics_denoise(
        self,
        denoised_trajs,
        agents_future,
        agents_future_valid,
        agents_interested,
        top_k=None,
    ):
        """
        Calculates the denoising metrics for the predicted trajectories.

        Args:
            denoised_trajs (torch.Tensor): Denoised trajectories of shape [B, A, T, 2].
            agents_future (torch.Tensor): Ground truth future trajectories of agents of shape [B, A, T, 2].
            agents_future_valid (torch.Tensor): Validity mask for future trajectories of agents of shape [B, A, T].
            agents_interested (torch.Tensor): Interest mask for agents of shape [B, A].
            top_k (int, optional): Number of top agents to consider. Defaults to None.

        Returns:
            Tuple[float, float]: A tuple containing the denoising ADE (Average Displacement Error) and FDE (Final Displacement Error).
        """

        if not top_k:
            top_k = self._agents_len

        pred_traj = denoised_trajs[:, :top_k, :, :2]  # [B, A, T, 2]
        gt = agents_future[:, :top_k, 1:, :2]  # [B, A, T, 2]
        gt_mask = (
            agents_future_valid[:, :top_k, 1:]
            & (agents_interested[:, :top_k, None] > 0)
        ).bool()  # [B, A, T]

        denoise_mse = torch.norm(pred_traj - gt, dim=-1)
        denoise_ADE = denoise_mse[gt_mask].mean()
        denoise_FDE = denoise_mse[..., -1][gt_mask[..., -1]].mean()

        return denoise_ADE.item(), denoise_FDE.item()

    @torch.no_grad()
    def calculate_metrics_predict(
        self,
        goal_trajs,
        agents_future,
        agents_future_valid,
        agents_interested,
        top_k=None,
    ):
        """
        Calculates the metrics for predicting goal trajectories.

        Args:
            goal_trajs (torch.Tensor): Tensor of shape [B, A, Q, T, 2] representing the goal trajectories.
            agents_future (torch.Tensor): Tensor of shape [B, A, T, 2] representing the future trajectories of agents.
            agents_future_valid (torch.Tensor): Tensor of shape [B, A, T] representing the validity of future trajectories.
            agents_interested (torch.Tensor): Tensor of shape [B, A] representing the interest level of agents.
            top_k (int, optional): The number of top agents to consider. Defaults to None.

        Returns:
            tuple: A tuple containing the goal Average Displacement Error (ADE) and goal Final Displacement Error (FDE).
        """

        if not top_k:
            top_k = self._agents_len
        goal_trajs = goal_trajs[:, :top_k, :, :, :2]  # [B, A, Q, T, 2]
        gt = agents_future[:, :top_k, 1:, :2]  # [B, A, T, 2]
        gt_mask = (
            agents_future_valid[:, :top_k, 1:]
            & (agents_interested[:, :top_k, None] > 0)
        ).bool()  # [B, A, T]

        goal_mse = torch.norm(
            goal_trajs - gt[:, :, None, :, :], dim=-1
        )  # [B, A, Q, T]
        goal_mse = goal_mse * gt_mask[..., None, :]  # [B, A, Q, T]
        best_idx = torch.argmin(goal_mse.sum(-1), dim=-1)

        best_goal_mse = goal_mse[
            torch.arange(goal_mse.shape[0])[:, None],
            torch.arange(goal_mse.shape[1])[None, :],
            best_idx,
        ]

        goal_ADE = best_goal_mse.sum() / gt_mask.sum()
        goal_FDE = best_goal_mse[..., -1].sum() / gt_mask[..., -1].sum()

        return goal_ADE.item(), goal_FDE.item()

    ################### Helper Functions ##############
    def batch_to_device(self, input_dict: dict, device: torch.device):
        """
        Move the tensors in the input dictionary to the specified device.

        Args:
            input_dict (dict): A dictionary containing tensors to be moved.
            device (torch.device): The target device to move the tensors to.

        Returns:
            dict: The input dictionary with tensors moved to the specified device.
        """
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(device)

        return input_dict

    def normalize_actions(self, actions: torch.Tensor):
        """
        Normalize the given actions using the mean and standard deviation.

        Args:
            actions : The actions to be normalized.

        Returns:
            The normalized actions.
        """
        return (actions - self.action_mean) / self.action_std

    def unnormalize_actions(self, actions: torch.Tensor):
        """
        Unnormalize the given actions using the stored action standard deviation and mean.

        Args:
            actions: The normalized actions to be unnormalized.

        Returns:
             The unnormalized actions.
        """
        return actions * self.action_std + self.action_mean
