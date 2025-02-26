"""
Config for Diffusion.
"""

from robomimic.config.base_config import BaseConfig


class Diff_BCConfig(BaseConfig):
    ALGO_NAME = "diff"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.learning_rate.initial = (
            1e-4  # policy learning rate
        )
        self.algo.optim_params.policy.regularization.L2 = (
            1e-6  # L2 regularization strength
        )
        self.algo.optim_params.policy.scheduler.name = "cosine"
        self.algo.optim_params.policy.scheduler.num_warmup_steps = 500

        self.algo.pred_horizon = 16
        self.algo.obs_horizon = 2
        self.algo.action_horizon = 8

        self.algo.num_diffusion_iters = 100

        self.algo.use_ema = False

    def experiment_config(self):
        """
        Update from subclass to set paper defaults for gym envs.
        """
        super(Diff_BCConfig, self).experiment_config()

        self.experiment.name = "diff_test"

        # epoch definitions - if not None, set an epoch to be this many gradient steps, else the full dataset size will be used
        self.experiment.epoch_every_n_steps = 100
        self.experiment.validation_epoch_every_n_steps = 10

        # rollout settings
        self.experiment.rollout.enabled = True
        self.experiment.rollout.n = 3
        self.experiment.rollout.horizon = 50
        self.experiment.rollout.rate = 10

    def train_config(self):
        """
        Update from subclass to set paper defaults for gym envs.
        """
        super(Diff_BCConfig, self).train_config()

        # update to normalize observations
        self.train.hdf5_normalize_obs = True
        self.train.hdf5_load_next_obs = False

        # increase batch size to 256
        self.train.batch_size = 256
        self.train.num_epochs = 150

        # tain data settings
        self.train.pad_seq_length = True
        self.train.seq_length = self.algo.pred_horizon
        self.train.pad_frame_stack = False

    def observation_config(self):
        """
        Update from superclass to use flat observations from gym envs.
        """
        super(Diff_BCConfig, self).observation_config()

        # specify low-dim observations for agent
        self.observation.modalities.obs.low_dim = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ]
