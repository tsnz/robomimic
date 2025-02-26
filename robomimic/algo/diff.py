"""
Implementation of Diff.
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from robomimic.models.conditional_unet1d import ConditionalUnet1D
from robomimic.models.obs_nets import MultiStepObservationGroupEncoder
from robomimic.models.obs_core import VisualCore
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

from typing import Callable

from copy import deepcopy


@register_algo_factory_func("diff")
def algo_config_to_class(algo_config):
    return Diff, {}


class Diff(PolicyAlgo):
    def __init__(
        self, algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device
    ):
        # current diffusion net configuration only allows predicting actions in multiple of 4s
        assert algo_config.pred_horizon % 4 == 0, (
            "current diffusion net only allows for prediction horizons which are divisible by 4"
        )

        super().__init__(
            algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device
        )

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """

        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_key_shapes)

        self.nets["encoder"] = MultiStepObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            feature_activation=None,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
                self.obs_config.encoder
            ),
            obs_horizon=self.algo_config.obs_horizon,
        )

        # replace bn with gn if ema is used for performance reasons
        if self.algo_config.use_ema:
            for net in self.nets["encoder"].nets["obs"].obs_nets.values():
                if isinstance(net, VisualCore):
                    net.backbone = self._replace_bn_with_gn(net.backbone)

        obs_shape = self.nets["encoder"].output_shape()[0]

        self.nets["policy"] = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.algo_config.obs_horizon * obs_shape,
        )

        self.nets = self.nets.float().to(self.device)

        if self.algo_config.use_ema:
            self.ema = EMAModel(parameters=self.nets.parameters(), power=0.75)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.algo_config.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )

    def _replace_submodules(
        self,
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module],
    ) -> nn.Module:
        """
        Replace all submodules selected by the predicate with
        the output of func.

        predicate: Return true if the module is to be replaced.
        func: Return new module to use.
        """
        if predicate(root_module):
            return func(root_module)

        bn_list = [
            k.split(".")
            for k, m in root_module.named_modules(remove_duplicate=True)
            if predicate(m)
        ]
        for *parent, k in bn_list:
            parent_module = root_module
            if len(parent) > 0:
                parent_module = root_module.get_submodule(".".join(parent))
            if isinstance(parent_module, nn.Sequential):
                src_module = parent_module[int(k)]
            else:
                src_module = getattr(parent_module, k)
            tgt_module = func(src_module)
            if isinstance(parent_module, nn.Sequential):
                parent_module[int(k)] = tgt_module
            else:
                setattr(parent_module, k, tgt_module)
        # verify that all modules are replaced
        bn_list = [
            k.split(".")
            for k, m in root_module.named_modules(remove_duplicate=True)
            if predicate(m)
        ]
        assert len(bn_list) == 0
        return root_module

    def _replace_bn_with_gn(
        self, root_module: nn.Module, features_per_group: int = 16
    ) -> nn.Module:
        """
        Relace all BatchNorm layers with GroupNorm.
        """
        self._replace_submodules(
            root_module=root_module,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=x.num_features // features_per_group,
                num_channels=x.num_features,
            ),
        )
        return root_module

    def _prepare_obs(self, obs):
        for key in obs:
            obs[key] = self._apply_horizon(obs[key], self.algo_config.obs_horizon)

        return obs

    def _apply_horizon(self, vals, horizon):
        vals = vals[:, :horizon, :]
        return vals

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        batch["obs"] = self._prepare_obs(batch["obs"])
        batch["actions"] = self._apply_horizon(
            batch["actions"], self.algo_config.pred_horizon
        )

        return TensorUtils.to_float(TensorUtils.to_device(batch, self.device))

    def postprocess_batch_for_training(self, batch, obs_normalization_stats):
        """
        Does some operations (like channel swap, uint8 to float conversion, normalization)
        after @process_batch_for_training is called, in order to ensure these operations
        take place on GPU.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader. Assumed to be on the device where
                training will occur (after @process_batch_for_training
                is called)

            obs_normalization_stats (dict or None): if provided, this should map observation
                keys to dicts with a "mean" and "std" of shape (1, ...) where ... is the
                default shape for the observation.

        Returns:
            batch (dict): postproceesed batch
        """
        nbatch = super(Diff, self).postprocess_batch_for_training(
            batch, obs_normalization_stats
        )
        return nbatch

    def _diff_lr_scheduler_from_optim_params(
        self, net_optim_params, optimizer, num_training_steps
    ):
        """ """
        name = net_optim_params["scheduler"]["name"]
        num_warmup_steps = net_optim_params["scheduler"]["num_warmup_steps"]
        return get_scheduler(
            name=name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _create_optimizers(self):
        """
        Creates optimizers using @self.optim_params and places them into @self.optimizers.
        """
        self.optimizers = dict()
        self.lr_schedulers = dict()

        self.optimizers["policy"] = TorchUtils.optimizer_from_optim_params(
            net_optim_params=self.optim_params["policy"],
            net=self.nets,
        )

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(Diff, self).train_on_batch(batch, epoch, validate=validate)

            # sample noise to add to actions
            noise = torch.randn(batch["actions"].shape, device=self.device)

            obs = self.nets["encoder"](obs=batch["obs"])

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (obs.shape[0],),
                device=self.device,
            ).long()

            # add noise to the clean images according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                batch["actions"], noise, timesteps
            )

            # predict the noise residual
            noise_pred = self.nets["policy"](noisy_actions, timesteps, global_cond=obs)
            # L2 loss
            loss = nn.functional.mse_loss(noise_pred, noise)

            info["predictions"] = TensorUtils.detach(noise_pred)
            info["losses"] = TensorUtils.detach(loss)

            if not validate:
                loss.backward()
                self.optimizers["policy"].step()
                self.optimizers["policy"].zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                self.lr_schedulers["policy"].step()
                if self.algo_config.use_ema:
                    self.ema.step(self.nets.parameters())

            return info

    def on_epoch_end(self, epoch):
        """
        Don't step lr_scheduler at end of epoch, since it is stepped in batch
        """
        return

    def on_train_end(self):
        if self.algo_config.use_ema:
            self.ema.copy_to(self.nets.parameters())

    def pre_model_save(self):
        # move ema params so it is saved instead of current net params
        if self.algo_config.use_ema:
            self._w_nets = self.nets
            self.nets = deepcopy(self.nets)
            self.ema.copy_to(self.nets.parameters())

    def post_model_save(self):
        # move back current working params
        if self.algo_config.use_ema:
            self.nets = self._w_nets
            self._w_nets = None

    def pre_run_prep(self, train_data, val_data):
        """
        Called before training to adapt settings depending on train and validation data
        """
        if self.global_config.experiment.epoch_every_n_steps is None:
            num_steps = len(train_data)
        else:
            num_steps = self.global_config.experiment.epoch_every_n_steps

        self.lr_schedulers["policy"] = self._diff_lr_scheduler_from_optim_params(
            net_optim_params=self.optim_params["policy"],
            optimizer=self.optimizers["policy"],
            num_training_steps=num_steps * self.global_config.train.num_epochs,
        )
        return

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        # bring list of observations into correct format
        obs = dict()
        for key in obs_dict[0].keys():
            obs[key] = torch.zeros(
                [1] + [self.algo_config.obs_horizon] + self.obs_key_shapes[key],
                device=self.device,
            )

        for i, ob in enumerate(obs_dict):
            for key, val in ob.items():
                obs[key][0][i] = val

        obs = self._prepare_obs(obs)

        pred_nets = self.nets

        if self.algo_config.use_ema:
            pred_nets = deepcopy(self.nets)
            self.ema.copy_to(pred_nets.parameters())

        obs_cond = pred_nets["encoder"](obs=obs)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (1, self.algo_config.pred_horizon, self.ac_dim), device=self.device
        )
        naction = noisy_action

        # init scheduler
        self.noise_scheduler.set_timesteps(self.algo_config.num_diffusion_iters)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = pred_nets["policy"](
                sample=naction, timestep=k, global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

        return naction

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(Diff, self).log_info(info)
        log["Loss"] = info["losses"].item()
        return log
