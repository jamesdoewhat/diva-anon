import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import os
import time

import gym
import numpy as np
import torch
import threading
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import gc

from qd_metarl.algorithms.a2c import A2C
from qd_metarl.algorithms.online_storage import DictOnlineStorage, OnlineStorage
from qd_metarl.algorithms.ppo import PPO
from qd_metarl.exploration.exploration_bonus import ExplorationBonus
from qd_metarl.models.policy import Policy
from qd_metarl.utils import evaluation as utl_eval
from qd_metarl.utils import env_utils as utl
from qd_metarl.utils.loggers import TBLogger, WandBLogger
from qd_metarl.utils import maze_utils as mze
from qd_metarl.models.vae import VaribadVAE
from qd_metarl.models.ensemble_vae import VaribadEnsembleVAE
from qd_metarl.utils.torch_utils import DeviceConfig

from scipy.stats import entropy
from qd_metarl.utils.schedulers import LinearSchedule, ConstantSchedule
from qd_metarl.utils.viz_utils import plot_trial_data
from qd_metarl.utils.plr_utils import get_plr_args_dict
from qd_metarl.qd.qd_module import QDModule


class MetaLearner:
    """
    Meta-Learner class with the main training loop for VariBAD.
    """
    def __init__(self, args, envs, plr_components, archive_dims=None):
        self.args = args
        self.envs = envs
        self.plr_level_sampler, self.plr_level_store = plr_components
        self.archive_dims = archive_dims
        utl.seed(self.args.seed, self.args.deterministic_execution)

        # Set environment-specific variables
        self.use_beta_distribution = 'Racing' in args.env_name

        # Calculate number of updates and keep count of frames/iterations
        self.num_updates = int(args.num_frames) // args.policy_num_steps // args.num_processes
        self.frames = 0
        self.iter_idx = -1
        print(f'num_updates (num_frames / policy_num_steps / num_processes): {self.num_updates}')
        print(f'policy_num_steps: {args.policy_num_steps}')
        print(f'num_processes: {args.num_processes}')

        # Initialise logger
        logger_class = WandBLogger if self.args.use_wb else TBLogger
        self.logger = logger_class(self.args, self.args.wb_label)

        if self.args.use_plr:
            # If we use a temperature schedule, we need to initialize it
            if (self.args.plr_level_replay_temperature_start
                    == self.args.plr_level_replay_temperature_end):
                self.plr_temperature_scheduler = ConstantSchedule(
                    self.args.plr_level_replay_temperature_start)
            else:
                self.plr_temperature_scheduler = LinearSchedule(
                    start=self.args.plr_level_replay_temperature_start,
                    end=self.args.plr_level_replay_temperature_end,
                    steps=self.num_updates)
        
        self.plr_lock = threading.Lock() if self.args.use_plr else None

        assert not self.args.single_task_mode, 'Single task mode not supported'
        self.train_tasks = None

        # Calculate what the maximum length of the trajectories is
        self.args.max_trajectory_len = self.envs._max_trial_steps
        self.args.max_trajectory_len *= self.args.trials_per_episode
        print('trials_per_episode:', self.args.trials_per_episode)
        print('max_trial_steps:', self.envs._max_trial_steps)
        print('max_trajectory_len:', self.args.max_trajectory_len)

        # Get policy input dimensions
        if isinstance(self.envs.observation_space, gym.spaces.Dict):
            # This will be the case if observation space is Dict
            state_keys = list(self.envs.observation_space.spaces.keys())
            self.args.state_dim = {
                key: self.envs.observation_space.spaces[key].shape 
                for key in state_keys}
            self.args.state_feature_extractor = \
                lambda i, o, a: mze.MazeFeatureExtractor(
                i, o, a, relevant_keys=self.args.state_relevant_keys)
            self.online_storage_class = DictOnlineStorage
            self.args.state_is_dict = True
            self.args.state_dtype = torch.float32
            self.args.state_is_image = False
        elif isinstance(self.envs.observation_space, gym.spaces.Box):
            # TODO: VariBAD believes that the state is flat, which is not the
            # case for Racing---it's a 2D image. Fix.
            self.args.state_is_dict = False
            self.online_storage_class = OnlineStorage
            if len(self.envs.observation_space.shape) == 3:
                # We assume that the state is an image
                self.args.state_dim = self.envs.observation_space.shape
                self.args.state_feature_extractor = utl.FeatureExtractorConv
                self.args.state_dtype = torch.uint8
                self.args.state_is_image = True
            elif len(self.envs.observation_space.shape) == 1:
                # We assume that the state is flat
                self.args.state_dim = self.envs.observation_space.shape[0]
                self.args.state_feature_extractor = utl.FeatureExtractor
                self.args.state_dtype = torch.float32
                self.args.state_is_image = False
        else: 
            raise NotImplementedError
        self.args.task_dim = self.envs.task_dim
        self.args.belief_dim = self.envs.belief_dim
        self.args.num_states = self.envs.num_states
        # Get policy output (action) dimensions
        self.args.action_space = self.envs.action_space
        if isinstance(self.envs.action_space, gym.spaces.discrete.Discrete):
            self.args.action_dim = 1
        else:
            self.args.action_dim = self.envs.action_space.shape[0]

        # Initialise VAE 
        if self.args.vae_use_ensemble:
            self.vae = VaribadEnsembleVAE(
                self.args, 
                self.logger, 
                lambda: self.iter_idx,
                ensemble_size=self.args.ensemble_size)
        else:
            self.vae = VaribadVAE(
                self.args, 
                self.logger, 
                lambda: self.iter_idx)
        
        # Initialise reward bonus
        if self.args.add_exploration_bonus:
            self.intrinsic_reward = ExplorationBonus(
                args=self.args,
                logger=self.logger,
                dim_state=self.args.state_dim,
                encoder=self.vae.encoder,
                rollout_storage=self.vae.rollout_storage)
        else:
            self.intrinsic_reward = None

        # Initialise policy
        self.policy = self.initialise_policy()
        self.policy_storage = self.initialise_policy_storage()

        # Initialise QD components
        if (self.args.use_qd or 
            self.args.use_plr and self.args.plr_env_generator == 'gen'):
            self._init_qd_module()
        else:
            self.qd_module = None

        # For PLR-gen, we need to set the bounds of the genotype so we can
        # randomly generate them outside of QD code
        if self.plr_level_store is not None:
            self.envs.set_genotype_bounds_info(
                self.qd_module.lower_bounds, self.qd_module.upper_bounds, self.qd_module.solution_dim, self.envs.size, self.envs.gt_type)

    def initialise_policy_storage(self, num_steps=None, num_processes=None):
        if num_steps is None: 
            num_steps = self.args.policy_num_steps
        if num_processes is None:
            num_processes = self.args.num_processes
        return self.online_storage_class(
            args=self.args,
            model=self.policy.actor_critic,
            num_steps=num_steps,
            num_processes=num_processes,
            state_dim=self.args.state_dim,
            latent_dim=self.args.latent_dim,
            belief_dim=self.args.belief_dim,
            task_dim=self.args.task_dim,
            action_space=self.args.action_space,
            hidden_size=self.args.encoder_gru_hidden_size,
            normalise_rewards=self.args.norm_rew_for_policy,
            add_exploration_bonus=self.args.add_exploration_bonus,
            intrinsic_reward=self.intrinsic_reward,
            use_popart=self.args.use_popart)

    def initialise_policy(self):
        # Initialise policy network
        policy_net = Policy(
            args=self.args,
            pass_state_to_policy=self.args.pass_state_to_policy,
            pass_latent_to_policy=self.args.pass_latent_to_policy,
            pass_belief_to_policy=self.args.pass_belief_to_policy,
            pass_task_to_policy=self.args.pass_task_to_policy,
            dim_state=self.args.state_dim,
            dim_latent=self.args.latent_dim * 2,
            dim_belief=self.args.belief_dim,
            dim_task=self.args.task_dim,
            hidden_layers=self.args.policy_layers,
            activation_function=self.args.policy_activation_function,
            policy_initialisation=self.args.policy_initialisation,
            action_space=self.envs.action_space,
            init_std=self.args.policy_init_std,
            state_feature_extractor=self.args.state_feature_extractor,
            state_is_dict=self.args.state_is_dict,
            use_popart=self.args.use_popart,
            use_beta=self.use_beta_distribution,
        ).to(DeviceConfig.DEVICE)

        # Args and kwargs shared by all policy optimisers
        common_args = [self.args, policy_net, self.args.policy_value_loss_coef, 
                       self.args.policy_entropy_coef]
        common_kwargs = {
            'policy_optimiser': self.args.policy_optimiser,
            'policy_anneal_lr': self.args.policy_anneal_lr,
            'train_steps': self.num_updates,
            'optimiser_vae': self.vae.optimiser_vae,
            'lr': self.args.lr_policy,
            'eps': self.args.policy_eps}
        a2c_kwargs = common_args
        ppo_kwargs = {
            'ppo_epoch': self.args.ppo_num_epochs,
            'num_mini_batch': self.args.ppo_num_minibatch,
            'use_clipped_value_loss': self.args.ppo_use_clipped_value_loss,
            'clip_param': self.args.ppo_clip_param,
            'use_huber_loss': self.args.ppo_use_huberloss}
        ppo_kwargs.update(common_kwargs)
        
        # Initialise policy optimiser
        if self.args.policy == 'a2c':
            policy = A2C(*common_args, **a2c_kwargs)
        elif self.args.policy == 'ppo':
            policy = PPO(*common_args, **ppo_kwargs)
        else:
            raise NotImplementedError

        return policy
    
    def _init_qd_module(self):
        """ Initialize QD module. """
        self.qd_module = QDModule(self.args, self)
    
    def kickstart_training(self):
        """ Prepare for training """
        self.start_time = time.time()
        print('\nKickstarting training...')

        self.level_renderings_recent = dict()
        self.genotype_counts_all = defaultdict(lambda: 0)

        # Warm start
        if self.args.use_qd:
            # Fill the archive with initial solutions
            if not self.qd_module.skip_warm_start:
                if self.args.qd_use_two_stage_ws:
                    print('Performing 2-stage warm start...')
                    self.qd_module.fill_archive(stage=1, two_stage=True)
                    self.qd_module.prepare_second_ws_stage()
                    self.qd_module.fill_archive(stage=2, two_stage=True)
                else:
                    print('Performing 1-stage warm start...')
                    self.qd_module.fill_archive()
        
        # Reset environments
        if self.args.use_qd and not self.args.qd_use_plr_for_training:
            # Get initial genotype samples from the archive for each env.
            print('Sampling initial genotypes from QD archive...')
            genotypes = self.qd_module.sample_from_archive(self.args.num_processes)
            # Reset each environment and set tasks to the sampled genotypes.
            prev_state, belief, task, level_seeds = utl.reset_env(self.envs, self.args, task=genotypes)
            if self.args.use_plr:
                self.plr_iter_idx = 0
                level_seeds = level_seeds.unsqueeze(-1)
        elif self.args.use_plr:
            print('Sampling initial level seeds using PLR...')
            self.plr_iter_idx = 0
            prev_state, belief, task, level_seeds = utl.reset_env(self.envs, self.args)
            level_seeds = level_seeds.unsqueeze(-1)
        else:
            print('Randomly sampling initial environments...')
            prev_state, belief, task, level_seeds = utl.reset_env(self.envs, self.args)

        # End if only doing warm start
        if self.args.qd_warm_start_only:
            print('Ending because warm start complete!')
            # Print (1) number of genotypes in archive and (2) percentage of archive filled
            print('Number of genotypes in archive: ', self.qd_module.archive._stats.num_elites)
            print('Percentage of archive filled: ', self.qd_module.archive._stats.num_elites / self.qd_module.archive._cells)
            self.envs.close()
            self._prev_state, self._belief, self._task, self._level_seeds = \
                None, None, None, None
            return        
        if self.args.qd_no_sim_objective and hasattr(self.qd_module, 'qd_envs'):
            # Close QD environments since we're done with them
            try:
                self.qd_module.qd_envs.close()
            except:
                pass

        # Store initial genotypes
        if self.args.qd_supported_env:
            init_genotypes = self.envs.genotype
            assert init_genotypes is not None
            init_genotypes_processed = [
                QDModule.process_genotype_static(g, self.args.qd_emitter_type) 
                for g in init_genotypes]
        else:
            init_genotypes = None
            init_genotypes_processed = None
        
        if self.args.qd_supported_env:
            for i in range(self.args.num_processes):
                self.genotype_counts_all[init_genotypes_processed[i]] += 1

        # Store init level renderings
        if self.args.eval_save_video and self.args.qd_supported_env:
            init_level_renderings = self.envs.level_rendering
            for i in range(self.args.num_processes):
                self.level_renderings_recent[init_genotypes_processed[i]] = \
                    init_level_renderings[i]
            
        # Insert initial observation / embeddings to rollout storage
        if self.args.state_is_dict:
            for k in prev_state.keys():
                self.policy_storage.prev_state[k][0].copy_(prev_state[k])
        else:
            self.policy_storage.prev_state[0].copy_(prev_state)

        # Log once before training
        with torch.no_grad():
            self.eval_and_log(first_log=True)

        self.intrinsic_reward_is_pretrained = False

        self._prev_state, self._belief, self._task, self._level_seeds = \
            prev_state, belief, task, level_seeds

    def _reset_done_environments(self):
        """
        Check if we need to reset environmets based on self._done_indices and
        if so, reset the environments.
        """
        # Check if we need to reset environments
        if len(self._done_indices) == 0:
            return
        # Reset environments that are done
        st1 = time.time()
        if self.args.use_qd and not self.args.qd_use_plr_for_training:
            print('Using QD to sample new genotypes for training!')
            # If using QD, sample new genotypes from archive
            genotypes = self.qd_module.sample_from_archive(len(self._done_indices))
            self._next_state, self._belief, self._task, self._level_seeds = utl.reset_env(
                self.envs, self.args, indices=self._done_indices, 
                state=self._next_state, task=genotypes)
            if self.args.use_plr:
                self._level_seeds = self._level_seeds.unsqueeze(-1)
            if self.args.qd_supported_env:
                genotypes = self.envs.genotype
        else:
            # Otherwise, we just trust that the envs reset
            # themselves properly
            self._next_state, self._belief, self._task, self._level_seeds = utl.reset_env(
                self.envs, self.args, indices=self._done_indices, 
                state=self._next_state)
            if self.args.use_plr:
                print('Using PLR to sample new training levels!')
                self._level_seeds = self._level_seeds.unsqueeze(-1)
            else:
                print('Randomly sampling new training levels!')
            if self.args.qd_supported_env:
                genotypes = self.envs.genotype

        # We only store these values if our environment supports genotypes
        if self.args.qd_supported_env:
            # Store genotypes
            for i in self._done_indices:
                self.genotype_counts_all[QDModule.process_genotype_static(
                    genotypes[i], qd_emitter_type=self.args.qd_emitter_type)] += 1
            # Store level renderings
            if self.args.eval_save_video:
                level_renderings = self.envs.level_rendering
            else:
                level_renderings = [None for _ in range(self.args.num_processes)]
            for i in self._done_indices:
                self.level_renderings_recent[QDModule.process_genotype_static(
                    genotypes[i], qd_emitter_type=self.args.qd_emitter_type)] = \
                    level_renderings[i]
        et1 = time.time(); self.sts_reset_env.append(et1-st1)

    def _train_policy_step(self, step):
        # Sample actions from policy
        st1 = time.time()
        with torch.no_grad():
            self._value, self._action = utl.select_action(
                args=self.args, policy=self.policy, state=self._prev_state,
                belief=self._belief, task=self._task, deterministic=False,
                latent_sample=self._latent_sample, latent_mean=self._latent_mean,
                latent_logvar=self._latent_logvar,
            )
        et1 = time.time(); self.sts_select_action.append(et1-st1)

        # Take step in the environment
        st1 = time.time()
        [self._next_state, self._belief, self._task], rewards, \
            done, infos = utl.env_step(self.envs, self._action, self.args)
        et1 = time.time(); self.sts_env_step.append(et1-st1)
        
        # Gather information from step
        if len(rewards) == 2:
            rew_raw, rew_normalised = rewards  # vector norm wrapper
        else: 
            rew_raw = rewards
            rew_normalised = rewards

        # NOTE: we're only logging values for the first environment,
        # for the sake of efficiency
        for k, v in infos[0].items():
            if 'time/' in k:
                self.sts_env_step_infos[k].append(v)
            if 'env/' in k:
                # Want to specify these are values for the training env
                self.sts_env_step_infos['train-' + k].append(v)
        done = torch.from_numpy(np.array(done, dtype=int)).to(DeviceConfig.DEVICE).float().view((-1, 1))
        masks_done = torch.FloatTensor(  # mask for trial ends
            [[0.0] if done_ else [1.0] for done_ in done]).to(DeviceConfig.DEVICE)
        bad_masks = torch.FloatTensor(  # trial ended because time limit was reached
            [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos]).to(DeviceConfig.DEVICE)
        # TODO: Look into how DCD code uses this. Is it relevant to us at all?
        cliffhanger_masks = torch.FloatTensor(
            [[0.0] if 'cliffhanger' in info.keys() else [1.0] for info in infos]).to(DeviceConfig.DEVICE)            

        # Compute next embedding (for next loop and/or value prediction bootstrap)
        st1 = time.time()
        with torch.no_grad():
            self._latent_sample, self._latent_mean, self._latent_logvar, self._hidden_state = \
                utl.update_encoding(encoder=self.vae.encoder,
                    next_obs=self._next_state, action=self._action, reward=rew_raw,
                    done=done, hidden_state=self._hidden_state)
        et1 = time.time(); self.sts_update_encoding.append(et1-st1)

        # Before resetting, update the embedding and add to vae buffer
        # (last state might include useful task info)
        st1 = time.time()
        if not (self.args.disable_decoder and 
                self.args.disable_kl_term):
            if isinstance(self._prev_state, dict):
                prev_state_clone = {k: v.clone() for k, v in self._prev_state.items()}
                next_state_clone = {k: v.clone() for k, v in self._next_state.items()}
            else:
                prev_state_clone = self._prev_state.clone()
                next_state_clone = self._next_state.clone()
            self.vae.rollout_storage.insert(prev_state_clone,
                self._action.detach().clone(), next_state_clone,
                rew_raw.clone(), done.clone(),
                (self._task.clone() if self._task is not None else None))
        et1 = time.time(); self.sts_vae_insert.append(et1-st1)

        # Add new observation to intrinsic reward
        if self.args.add_exploration_bonus:
            beliefs = torch.cat((self._latent_mean, self._latent_logvar), dim=-1)
            self.intrinsic_reward.add(
                self._next_state, beliefs, self._action.detach())

        # Add the obs before reset to the policy storage
        if isinstance(self._next_state, dict):
            for k in self._next_state.keys():
                self.policy_storage.next_state[k][step] = self._next_state[k].clone()
        else:
            self.policy_storage.next_state[step] = self._next_state.clone()

        # Reset environments that are done
        self._done_indices = np.argwhere(done.cpu().flatten()).flatten()
        self._reset_done_environments()

        st1 = time.time()
        # Add experience to policy buffer
        self.policy_storage.insert(state=self._next_state, belief=self._belief,
            task=self._task, actions=self._action, rewards_raw=rew_raw,
            rewards_normalised=rew_normalised, value_preds=self._value,
            masks=masks_done, bad_masks=bad_masks, 
            cliffhanger_masks=cliffhanger_masks, done=done,
            hidden_states=self._hidden_state.squeeze(0),
            latent_sample=self._latent_sample, latent_mean=self._latent_mean,
            latent_logvar=self._latent_logvar, level_seeds=self._level_seeds
        )
        et1 = time.time(); self.sts_policy_insert.append(et1-st1)
        self._prev_state = self._next_state
        self.frames += self.args.num_processes

    def _rnd_pretrain_step(self):
        # compute returns once - this will normalise the RND inputs!
        next_value = self.get_value(
            state=self._next_state,
            belief=self._belief,
            task=self._task,
            latent_sample=self._latent_sample,
            latent_mean=self._latent_mean,
            latent_logvar=self._latent_logvar)
        self.policy_storage.compute_returns(
            next_value,
            self.args.policy_use_gae,
            self.args.policy_gamma,
            self.args.policy_tau,
            use_proper_time_limits=self.args.use_proper_time_limits,
            vae=self.vae)

        self.intrinsic_reward.update(
            self.args.num_frames, self.iter_idx, log=False
        )  # (calling with max num of frames to init all networks)
        self.intrinsic_reward_is_pretrained = True

    def _plr_update(self):
        """ Update PLR weights """
        print('Updating PLR weights...')
        st = time.time()
        # NOTE: self.policy_storage.compute_returns already called in
        # update() above, so policy_storage has correct returns for PLR

        # Update level sampler (NOTE: currently we always update)
        if self.plr_level_sampler and self.args.precollect_len <= self.frames:
            # NOTE: We can't update level sampler until we've updated the 
            # agent, since before_update needs to be called to set
            # storage.action_log_dist, which is used by the following method
            if not self.args.use_qd:
                # We do not updated level sampler with online experience if
                # we are using QD.
                self.plr_level_sampler.update_with_rollouts(self.policy_storage)
            if self.plr_iter_idx > 5:
                self.plr_log()
            self.plr_iter_idx += 1

        # Clean up after update
        self.policy_storage.after_update()
        if self.plr_level_sampler and not self.args.use_qd:
            self.plr_level_sampler.after_update()
        
        et = time.time()
        self.logger.add('time/plr_update', et-st, self.iter_idx)

    def _vae_pretrain_update(self, p):
        self.vae.compute_vae_loss(
            update=True, 
            pretrain_index=self.iter_idx * 
                           self.args.num_vae_updates_per_pretrain + p)

    def _standard_update_step(self):
        st = time.time()
        # Check if we'll be doing any update steps at all
        if self.args.precollect_len <= self.frames:
            # Check if we are pre-training the VAE
            if self.args.pretrain_len > self.iter_idx:
                for p in range(self.args.num_vae_updates_per_pretrain):
                    self._vae_pretrain_update(p)
            # Otherwise do the normal update (policy + vae)
            else:
                self._train_stats = self.update()
                with torch.no_grad():
                    self.eval_and_log(first_log=False)
        et = time.time()
        self.logger.add('time/meta_rl_update', et-st, self.iter_idx)

        # --- INTRINSIC REWARD MODEL UPDATE
        if self.args.add_exploration_bonus:
            if self.iter_idx % self.args.rnd_update_frequency == 0:
                self.intrinsic_reward.update(self.frames, self.iter_idx)

        # --- PLR UPDATE ---
        if self.plr_level_sampler and (not self.args.use_qd or self.args.qd_use_plr_for_training):
            # NOTE: we do not perform plr update for QD unless we using it for 
            # sampling levels during training (instead of sampling from the archive)
            self._plr_update()

        #  --- QD EVAL AND ARCHIVE UPDATE ---

        st = time.time()
        if (self.args.use_qd and
            (self.iter_idx + 1) % self.args.qd_update_interval == 0 and
            not self.args.qd_no_sim_objective):
            # TODO(costales): maybe init subprocesses for envs here so that
            # we don't need to create new subprocesses for each update
            for uidx in range(self.args.qd_updates_per_iter):
                if self.args.qd_async_updates:
                    self.qd_module.launch_update()
                else:
                    self.qd_module.update()
                # Update iteration index
                self.qd_module.iter_idx += 1
        et = time.time()
        self.logger.add('time/qd_update', et-st, self.iter_idx)

        if self.plr_level_sampler:
            self.plr_level_sampler.temperature = self.plr_temperature_scheduler()
            self.logger.add('plr/temperature', self.plr_level_sampler.temperature, self.iter_idx)

        # Increment steps since last updated for all archive solutions
        if self.args.use_qd:
            self.qd_module.archive.increment_sslu()
            self.logger.add('qd/max_sslu', self.qd_module.archive.get_max_sslu(), self.iter_idx)

    def train(self):
        """ Main Meta-Training loop """
        # Kickstart training
        # input('B4 kickstart_training... (PRESS ENTER)')
        self.kickstart_training()
        if self.args.qd_warm_start_only:
            self.close()
            return
    
        # input('B4 loop... (PRESS ENTER)')
        # Training loop
        print('Training commences...')
        for self.iter_idx in range(self.num_updates):
            # Log iteration number
            print('-'*10, 'Iteration {}; Frames {}'.format(self.iter_idx, self.frames), '-'*10)
            
            # First, re-compute the hidden states given the current rollouts 
            # (since the VAE might've changed)
            st = time.time()
            with torch.no_grad():
                self.encode_running_trajectory()
            et = time.time(); self.logger.add('time/encode_running_trajectory', et-st, self.iter_idx)

            # Make sure we emptied buffers
            assert len(self.policy_storage.latent_mean) == 0  
            # Add initial hidden state we just computed to the policy storage
            self.policy_storage.hidden_states[0].copy_(self._hidden_state)
            self.policy_storage.latent_samples.append(self._latent_sample.clone())
            self.policy_storage.latent_mean.append(self._latent_mean.clone())
            self.policy_storage.latent_logvar.append(self._latent_logvar.clone())

            # Stats logging (timing and environment metrics)
            self.sts_select_action, self.sts_env_step, \
                self.sts_update_encoding, self.sts_vae_insert, \
                self.sts_reset_env, self.sts_policy_insert = [], [], [], [], [], []
            self.sts_env_step_infos = defaultdict(list)
            
            # Rollout policies for a few steps
            st = time.time()
            for step in range(self.args.policy_num_steps):
                self._train_policy_step(step)
            et = time.time(); self.logger.add('time/rollout_policies', et-st, self.iter_idx)
            
            # Log timing states for policy rollouts
            self.logger.add('time/rollout_policies_select_action', sum(self.sts_select_action), self.iter_idx)
            self.logger.add('time/rollout_policies_env_step', sum(self.sts_env_step), self.iter_idx)
            self.logger.add('time/rollout_policies_update_encoding', sum(self.sts_update_encoding), self.iter_idx)
            self.logger.add('time/rollout_policies_vae_insert', sum(self.sts_vae_insert), self.iter_idx)
            self.logger.add('time/rollout_policies_reset_env', sum(self.sts_reset_env), self.iter_idx)
            self.logger.add('time/rollout_policies_policy_insert', sum(self.sts_policy_insert), self.iter_idx)
            
            # Process step infos and log
            for k, v in self.sts_env_step_infos.items():
                agg_fn = sum if 'time/' in k else np.mean  # Sum times; mean others
                self.logger.add(k, agg_fn(v), self.iter_idx)

            # Log episodic rewards for completed environments
            if len(self._done_indices) > 0:
                episodic_rewards_raw = self.policy_storage.rewards_raw.sum(axis=0)[self._done_indices].cpu().numpy().mean()
                episodic_rewards_normalised = self.policy_storage.rewards_normalised.sum(axis=0)[self._done_indices].cpu().numpy().mean()
                self.logger.add('train/episodic_rewards_raw', episodic_rewards_raw, self.iter_idx)
                self.logger.add('train/episodic_rewards_normalised', episodic_rewards_normalised, self.iter_idx)

            # Check if we still need to fill the VAE buffer more
            if (len(self.vae.rollout_storage) == 0 and not self.args.size_vae_buffer == 0) or \
                    (self.args.precollect_len > self.frames):
                print('Not updating yet because; filling up the VAE buffer.')
                self.policy_storage.after_update()
                continue

            # Update
            if self.args.add_exploration_bonus and not self.intrinsic_reward_is_pretrained:
                # Pretrain RND model once to bring it on right scale
                self._rnd_pretrain_step()
            else:
                # Standard update step
                self._standard_update_step()

            # clean up after update
            self.policy_storage.after_update()

        print('The end! Closing envs...')
        self.close()

    def encode_running_trajectory(self):
        """
        (Re-)Encodes (for each process) the entire current trajectory.
        Returns sample/mean/logvar and hidden state (if applicable) for the 
        current timestep.
        """

        # For each process, get the current batch 
        # (zero-padded obs/act/rew + length indicators)
        prev_obs, next_obs, act, rew, lens = \
            self.vae.rollout_storage.get_running_batch()
        del prev_obs  # don't need this

        # Get embedding - will return (1+sequence_len) * batch * input_size 
        # -- includes the prior!
        all_latent_samples, all_latent_means, \
        all_latent_logvars, all_hidden_states = self.vae.encoder(
            actions=act, states=next_obs, rewards=rew, 
            hidden_state=None,return_prior=True)

        # Get the embedding / hidden state of the current time step 
        # (need to do this since we zero-padded)
        latent_sample = (torch.stack([all_latent_samples[lens[i]][i] 
                                     for i in range(len(lens))])).to(DeviceConfig.DEVICE)
        latent_mean = (torch.stack([all_latent_means[lens[i]][i] 
                                     for i in range(len(lens))])).to(DeviceConfig.DEVICE)
        latent_logvar = (torch.stack([all_latent_logvars[lens[i]][i] 
                                     for i in range(len(lens))])).to(DeviceConfig.DEVICE)
        hidden_state = (torch.stack([all_hidden_states[lens[i]][i] 
                                     for i in range(len(lens))])).to(DeviceConfig.DEVICE)

        self._latent_sample = latent_sample
        self._latent_mean = latent_mean
        self._latent_logvar = latent_logvar
        self._hidden_state = hidden_state

    def get_value(self, state, belief, task, latent_sample, 
                  latent_mean, latent_logvar):
        latent = utl.get_latent_for_policy(
            self.args, latent_sample=latent_sample, latent_mean=latent_mean, 
            latent_logvar=latent_logvar)
        return self.policy.actor_critic.get_value(state=state, belief=belief, 
            task=task, latent=latent).detach()

    def update(self):
        """
        Meta-update.
        Here the policy is updated for good average performance across tasks.
        """
        # Update policy (if we are not pre-training, have enough data in the 
        # vae buffer, and are not at iteration 0)
        if self.iter_idx >= self.args.pretrain_len and self.iter_idx > 0:

            # Bootstrap next value prediction
            with torch.no_grad():
                next_value = self.get_value(state=self._prev_state, 
                    belief=self._belief, task=self._task, latent_sample=self._latent_sample,
                    latent_mean=self._latent_mean, latent_logvar=self._latent_logvar)

            # Compute returns for current rollouts
            self.policy_storage.compute_returns(
                next_value, 
                self.args.policy_use_gae, 
                self.args.policy_gamma,
                self.args.policy_tau, 
                use_proper_time_limits=self.args.use_proper_time_limits,
                vae=self.vae)

            # Update agent (this will also call the VAE update!)
            policy_train_stats = self.policy.update(
                policy_storage=self.policy_storage,
                encoder=self.vae.encoder,
                rlloss_through_encoder=self.args.rlloss_through_encoder,
                compute_vae_loss=self.vae.compute_vae_loss)
        else:
            policy_train_stats = 0, 0, 0, 0
            # Pre-train the VAE
            if self.iter_idx < self.args.pretrain_len:
                self.vae.compute_vae_loss(update=True)

        return policy_train_stats
    
    ###########                     Logging                         ###########
    
    def plr_log(self):
        """ Logging for PLR. """
        # For readability
        la, ii = self.logger.add, self.iter_idx
        try: 
            sample_weights = self.plr_level_sampler.sample_weights()
        except FloatingPointError: 
            sample_weights = None
            print('Caught floating point error when computing sample weights.')
        except ZeroDivisionError:
            sample_weights = None
            print('Caught zero division error when computing sample weights.')
        if sample_weights is not None:
            # Log num non-zero sample weights
            perc_nonzero = np.count_nonzero(sample_weights) / len(sample_weights)
            la('plr/perc_nonzero_sample_weights', perc_nonzero, ii)
            # Log entropy of non-zero portion of sample weights
            la('plr/entropy_nonzero_sample_weights', entropy(sample_weights[sample_weights != 0]), ii)
            la('plr/num_seeds', len(sample_weights), ii)

    def eval_and_log(self, first_log=False):
        """ Logging. """
        if first_log:
            run_stats = None
            train_stats = None
        else:
            run_stats = [self._action, self.policy_storage.action_log_probs, self._value]
            train_stats = self._train_stats

        # For readability
        la, ii, fr = self.logger.add, self.iter_idx, self.frames

        log_video_this_iter = (
            (self.iter_idx + 1) % self.args.vis_interval == 0 
             and self.args.eval_save_video and self.iter_idx + 1 > 0)

        # --- Evaluate policy ----
        if ((self.iter_idx + 1) % self.args.eval_interval == 0
            and not self.args.skip_eval):
            print('Evaluation...')
            ret_rms = (self.envs.venv.ret_rms if self.args.norm_rew_for_policy 
                       else None)
            ret = utl_eval.evaluate(
                args=self.args,
                policy=self.policy, 
                ret_rms=ret_rms, 
                encoder=self.vae.encoder,
                iter_idx=self.iter_idx, 
                tasks=self.train_tasks,
                create_video=log_video_this_iter,
                vae=self.vae,
                intrinsic_reward=self.intrinsic_reward)

            (returns_per_trial, sparse_returns_per_trial, 
             dense_returns_per_trial, returns_bonus_per_trial,
             returns_bonus_state_per_trial, returns_bonus_belief_per_trial,
             returns_bonus_hyperstate_per_trial,
             returns_bonus_vae_loss_per_trial,
             success_per_trial, video_buffer,
             trial_latent_means, trial_latent_logvars, trial_events,
             final_metrics, eval_sts_env_step_infos) = ret

            print('Post-evaluation...')

            # Log the misc eval stats
            for k, v in eval_sts_env_step_infos.items():
                la(k, np.mean(v), self.iter_idx)

            # Log the return avg/std across tasks (=processes)
            returns_avg = returns_per_trial.mean(dim=0)
            returns_std = returns_per_trial.std(dim=0)
            if success_per_trial is not None:
                success_avg = success_per_trial.mean(dim=0)
            sparse_returns_avg = sparse_returns_per_trial.mean(dim=0)
            dense_returns_avg = dense_returns_per_trial.mean(dim=0)
            returns_bonus_avg = returns_bonus_per_trial.mean(dim=0)
            returns_bonus_state_avg = returns_bonus_state_per_trial.mean(dim=0)
            returns_bonus_belief_avg = returns_bonus_belief_per_trial.mean(dim=0)
            returns_bonus_hyperstate_avg = returns_bonus_hyperstate_per_trial.mean(dim=0)
            returns_bonus_vae_loss_avg = returns_bonus_vae_loss_per_trial.mean(dim=0)
            for k, v in final_metrics.items():
                # Take mean over first dimension
                final_metrics[k] = v.mean(dim=0)

            la('return_avg_per_iter/diff', returns_avg[-1] - returns_avg[0], ii)
            la('return_avg_per_frame/diff', returns_avg[-1] - returns_avg[0], fr)
            if success_per_trial is not None:
                la('success_avg_per_iter/diff', success_avg[-1] - success_avg[0], ii)
                la('success_avg_per_frame/diff', success_avg[-1] - success_avg[0], fr)
            for metric_name, v in final_metrics.items():
                # NOTE: -2 because there's still an extra buffer slot at the
                # end from eval---we removed these for return and success, 
                # but not final metrics
                la(f'{metric_name}/diff', v[-2] - v[0], ii)

            for k in range(len(returns_avg)):
                la('return_avg_per_iter/trial_{}'.format(k + 1), returns_avg[k], ii)
                la('return_avg_per_frame/trial_{}'.format(k + 1), returns_avg[k], fr)
                la('return_std_per_iter/trial_{}'.format(k + 1), returns_std[k], ii)
                la('return_std_per_frame/trial_{}'.format(k + 1), returns_std[k], fr)
                if success_per_trial is not None:
                    la('success_avg_per_iter/trial_{}'.format(k + 1), success_avg[k], ii)
                    la('success_avg_per_frame/trial_{}'.format(k + 1), success_avg[k], fr)
                la('sparse_return_avg_per_iter/trial_{}'.format(k + 1), sparse_returns_avg[k], ii)
                la('sparse_return_avg_per_frame/trial_{}'.format(k + 1), sparse_returns_avg[k], fr)
                la('dense_return_avg_per_iter/trial_{}'.format(k + 1), dense_returns_avg[k], ii)
                la('dense_return_avg_per_frame/trial_{}'.format(k + 1), dense_returns_avg[k], fr)
                la('returns_bonus_avg_per_iter/trial_{}'.format(k + 1), returns_bonus_avg[k], ii)
                la('returns_bonus_avg_per_frame/trial_{}'.format(k + 1), returns_bonus_avg[k], fr)
                
                for metric_name, v in final_metrics.items():
                    la(f'{metric_name}/trial_{k+1}', v[k], ii)

                if self.args.add_exploration_bonus:
                    # individual bonuses: states
                    if self.args.exploration_bonus_state:
                        la('returns_bonus_state_avg_per_iter/trial_{}'.format(k + 1), returns_bonus_state_avg[k], ii)
                        la('returns_bonus_state_avg_per_frame/trial_{}'.format(k + 1), returns_bonus_state_avg[k], fr)
                    # individual bonuses: beliefs
                    if self.args.exploration_bonus_belief:
                        la('returns_bonus_belief_avg_per_iter/trial_{}'.format(k + 1), returns_bonus_belief_avg[k], ii)
                        la('returns_bonus_belief_avg_per_frame/trial_{}'.format(k + 1), returns_bonus_belief_avg[k], fr)
                    # individual bonuses: hyperstates
                    if self.args.exploration_bonus_hyperstate:
                        la('returns_bonus_hyperstate_avg_per_iter/trial_{}'.format(k + 1), returns_bonus_hyperstate_avg[k], ii)
                        la('returns_bonus_hyperstate_avg_per_frame/trial_{}'.format(k + 1), returns_bonus_hyperstate_avg[k], fr)
                    # individual bonuses: vae loss
                    if self.args.exploration_bonus_vae_error:
                        la('returns_bonus_vae_loss_avg_per_iter/trial_{}'.format(k + 1), returns_bonus_vae_loss_avg[k], ii)
                        la('returns_bonus_vae_loss_avg_per_frame/trial_{}'.format(k + 1), returns_bonus_vae_loss_avg[k], fr)

            print(f"Updates {self.iter_idx}, "
                  f"Frames {self.frames}, "
                  f"FPS {int(self.frames / (time.time() - self.start_time))}, "
                  f"\n Mean return (train): {returns_avg[-1].item()} \n")
            
            # Log video
            if log_video_this_iter:
                print('iter idx:', self.iter_idx)
                video_name = utl.generate_video_name(ii+1)
                self.logger.add_video(f'eval_videos/{video_name}', 
                                      video_buffer, ii+1, 
                                      fps=self.args.video_fps)
                
            # Log latent means and logvars plots
            if trial_latent_means is not None:
                for k in range(len(trial_latent_means)): 
                    img = plot_trial_data(
                        trial_latent_means,
                        trial_latent_logvars,
                        trial_events,
                        trial_num=k,
                        process_num=0, 
                        ensemble_size=self.args.ensemble_size 
                                    if self.args.vae_use_ensemble else None)
                    img = utl.stitch_images([img], n=1)
                    image_name = 'trial_{}.png'.format(k+1)
                    self.logger.add_image(f'latent_vals_over_trials/{image_name}',
                                          img, ii+1)
            
            # Log number of genotypes encountered total
            la('train/num_genotypes_total', len(self.genotype_counts_all), ii)
            # Log renderings
            if self.args.eval_save_video: 
                images_to_stitch = list(self.level_renderings_recent.values())[:5*5]
                if len(images_to_stitch) > 0:
                    image = utl.stitch_images(images_to_stitch, n=5)
                    image_name = 'level_rendering.png'
                    self.logger.add_image(f'train_levels/{image_name}', image, ii+1)
            
            # Clear renderings
            self.level_renderings_recent = dict()

            # After every evaluation, we want to clear memory
            plt.close('all')
            gc.collect()

        # --- Save models ---
        if (self.iter_idx + 1) % self.args.save_interval == 0:
            save_path = os.path.join(self.logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            idx_labels = ['']
            if self.args.save_intermediate_models:
                idx_labels.append(int(self.iter_idx))
            # TODO: look at OG varibad/hyperx repo for their buggy model saving code

        # --- Log some other things ---
        if (((self.iter_idx + 1) % self.args.log_interval == 0) and
            (train_stats is not None)):
            # Log environment stats (?)
            if isinstance(self.policy_storage.prev_state, dict):
                for k in self.policy_storage.prev_state.keys():
                    la(f'environment/state_max_{k}', self.policy_storage.prev_state[k].max(), ii)
                    la(f'environment/state_min_{k}', self.policy_storage.prev_state[k].min(), ii)
            else:
                la('environment/state_max', self.policy_storage.prev_state.max(), ii)
                la('environment/state_min', self.policy_storage.prev_state.min(), ii)
            # Log reward stats
            la('environment/rew_max', self.policy_storage.rewards_raw.max(), ii)
            la('environment/rew_min', self.policy_storage.rewards_raw.min(), ii)
            # Log policy stats
            la('policy_losses/value_loss', train_stats[0], ii)
            la('policy_losses/action_loss', train_stats[1], ii)
            la('policy_losses/dist_entropy', train_stats[2], ii)
            la('policy_losses/sum', train_stats[3], ii)

            la('policy/action', run_stats[0][0].float().mean(), ii)
            if hasattr(self.policy.actor_critic, 'logstd'):
                action_logstd = self.policy.actor_critic.dist.logstd.mean()
                la('policy/action_logstd', action_logstd, ii)
            la('policy/action_logprob', run_stats[1].mean(), ii)
            la('policy/value', run_stats[2].mean(), ii)
            # Log encoder stats
            la('encoder/latent_mean', torch.cat(self.policy_storage.latent_mean).mean(), ii)
            la('encoder/latent_logvar', torch.cat(self.policy_storage.latent_logvar).mean(), ii)
            # Log average weights and gradients of applicable models 
            for [model, name] in [
                [self.policy.actor_critic, 'policy'],
                [self.vae.encoder, 'encoder'],
                [self.vae.reward_decoder, 'reward_decoder'],
                [self.vae.state_decoder, 'state_transition_decoder'],
                [self.vae.task_decoder, 'task_decoder']
            ]:
                # For each model...
                if model is not None:
                    # If it exists...
                    # Log the mean weights of the model
                    param_list = list(model.parameters())
                    param_mean = np.mean([param_list[i].data.cpu().numpy().mean() for i in range(len(param_list))])
                    la('weights/mean_{}'.format(name), param_mean, ii)
                    param_abs_mean = np.mean([np.abs(param_list[i].data.cpu().numpy()).mean() for i in range(len(param_list))])
                    la('weights/mean-abs_{}'.format(name), param_abs_mean, ii)
                    # If policy, we also log standard deviation
                    if name == 'policy':
                        la('weights/policy_std', param_list[0].data.mean(), ii)
                    # Log mean gradient information
                    if param_list[0].grad is not None:
                        try:
                            grad_list = [param_list[i].grad.cpu().numpy() for i in range(len(param_list))]
                            # Mean gradients
                            param_grad_mean = np.mean([grad_list[i].mean() for i in range(len(grad_list))])
                            la('gradients/mean_{}'.format(name), param_grad_mean, ii)
                            # Mean of absolute value of gradients
                            param_grad_abs_mean = np.mean([np.abs(grad_list[i]).mean() for i in range(len(grad_list))])
                            la('gradients/mean-abs_{}'.format(name), param_grad_abs_mean, ii)
                            # Max of absolute value of gradients
                            param_grad_abs_max = np.max([np.abs(grad_list[i]).max() for i in range(len(grad_list))])
                            la('gradients/max-abs_{}'.format(name), param_grad_abs_max, ii)
                        except:
                            pass
    
    def close(self):
        self.envs.close()
        self.logger.close()