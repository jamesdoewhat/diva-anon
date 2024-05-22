import warnings

import gym
import numpy as np
# import torch
from torch.nn import functional as F
from torch.distributions.normal import Normal
import torch
import torch.nn as nn
from torch.optim import Adam

from qd_metarl.models.decoder import StateTransitionDecoder, RewardDecoder, TaskDecoder
from qd_metarl.models.encoder import RNNEncoder
from qd_metarl.utils.env_utils import get_task_dim, get_num_tasks, shape
from qd_metarl.utils.storage_vae import DictRolloutStorageVAE, RolloutStorageVAE
from qd_metarl.utils.torch_utils import DeviceConfig


class VaribadVAE:
    """
    VAE of VariBAD:
    - has an encoder and decoder
    - can compute the ELBO loss
    - can update the VAE (encoder+decoder)
    """

    def __init__(self, args, logger, get_iter_idx):

        self.args = args
        self.logger = logger
        self.get_iter_idx = get_iter_idx
        self.task_dim = (get_task_dim(self.args) if self.args.decode_task 
                         else None)
        self.num_tasks = (get_num_tasks(self.args) if self.args.decode_task 
                          else None)

        # initialise the encoder
        self.encoder = self.initialise_encoder()

        # initialise the decoders (returns None for unused decoders)
        self.state_decoder, self.reward_decoder, self.task_decoder = \
            self.initialise_decoder()

        # initialise rollout storage for the VAE update
        # (this differs from the data that the on-policy RL algorithm uses)
        if isinstance(self.args.state_dim, dict):
            RSVAE = DictRolloutStorageVAE
        else:
            RSVAE = RolloutStorageVAE
        self.rollout_storage = RSVAE(
            num_processes=self.args.num_processes,
            max_trajectory_len=self.args.max_trajectory_len,
            zero_pad=True,
            max_num_rollouts=self.args.size_vae_buffer,
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            vae_buffer_add_thresh=self.args.vae_buffer_add_thresh,
            task_dim=self.task_dim,
            state_dtype=self.args.state_dtype
        )

        # initalise optimiser for the encoder and decoders
        decoder_params = []
        if not self.args.disable_decoder:
            if self.args.decode_reward:
                decoder_params.extend(self.reward_decoder.parameters())
            if self.args.decode_state:
                decoder_params.extend(self.state_decoder.parameters())
            if self.args.decode_task:
                decoder_params.extend(self.task_decoder.parameters())
        self.optimiser_vae = Adam(
            [*self.encoder.parameters(), *decoder_params], lr=self.args.lr_vae)

    def initialise_encoder(self):
        """ Initialises and returns an RNN encoder """
        encoder = RNNEncoder(
            args=self.args,
            layers_before_gru=self.args.encoder_layers_before_gru,
            hidden_size=self.args.encoder_gru_hidden_size,
            layers_after_gru=self.args.encoder_layers_after_gru,
            latent_dim=self.args.latent_dim,
            action_dim=self.args.action_dim,
            action_embed_dim=self.args.action_embedding_size,
            state_dim=self.args.state_dim,
            state_embed_dim=self.args.state_embedding_size,
            reward_size=1,
            reward_embed_size=self.args.reward_embedding_size,
            state_feature_extractor=self.args.state_feature_extractor,
            state_is_image=self.args.state_is_image,
        ).to(DeviceConfig.DEVICE)
        return encoder

    def initialise_decoder(self):
        """
        Initialises and returns the (state/reward/task) decoder as specified 
        in self.args
        """

        if self.args.disable_decoder:
            return None, None, None

        latent_dim = self.args.latent_dim
        # if we don't sample embeddings for the decoder, we feed in 
        # mean & variance
        if self.args.disable_stochasticity_in_latent:
            latent_dim *= 2

        # initialise state decoder for VAE
        if self.args.decode_state:
            state_decoder = StateTransitionDecoder(
                args=self.args,
                layers=self.args.state_decoder_layers,
                latent_dim=latent_dim,
                action_dim=self.args.action_dim,
                action_embed_dim=self.args.action_embedding_size,
                state_dim=self.args.state_dim,
                state_embed_dim=self.args.state_embedding_size,
                pred_type=self.args.state_pred_type,
                state_feature_extractor=self.args.state_feature_extractor
            ).to(DeviceConfig.DEVICE)
        else:
            state_decoder = None

        # initialise reward decoder for VAE
        if self.args.decode_reward:
            reward_decoder = RewardDecoder(
                args=self.args,
                layers=self.args.reward_decoder_layers,
                latent_dim=latent_dim,
                state_dim=self.args.state_dim,
                state_embed_dim=self.args.state_embedding_size,
                action_dim=self.args.action_dim,
                action_embed_dim=self.args.action_embedding_size,
                num_states=self.args.num_states,
                multi_head=self.args.multihead_for_reward,
                pred_type=self.args.rew_pred_type,
                input_prev_state=self.args.input_prev_state,
                input_action=self.args.input_action,
                state_feature_extractor=self.args.state_feature_extractor
            ).to(DeviceConfig.DEVICE)
        else:
            reward_decoder = None

        # initialise task decoder for VAE
        if self.args.decode_task:
            assert self.task_dim != 0
            task_decoder = TaskDecoder(
                latent_dim=latent_dim,
                layers=self.args.task_decoder_layers,
                task_dim=self.task_dim,
                num_tasks=self.num_tasks,
                pred_type=self.args.task_pred_type,
            ).to(DeviceConfig.DEVICE)
        else:
            task_decoder = None

        return state_decoder, reward_decoder, task_decoder

    def compute_state_reconstruction_loss(
            self, latent, prev_obs, next_obs, action, return_predictions=False):
        """
        Compute state reconstruction loss. (No reduction of loss along batch 
        dimension is done here; sum/avg has to be done outside)

        TODO(rsc): Explain in which cases this function is used.
        """

        state_pred = self.state_decoder(latent, prev_obs, action)

        if self.args.state_pred_type == 'deterministic':
            loss_state = (state_pred - next_obs).pow(2).mean(dim=-1)
        elif self.args.state_pred_type == 'gaussian':  # TODO: untested!
            state_pred_mean = state_pred[:, :state_pred.shape[1] // 2]
            state_pred_std = torch.exp(
                0.5 * state_pred[:, state_pred.shape[1] // 2:])
            m = Normal(state_pred_mean, 
                                                  state_pred_std)
            loss_state = -m.log_prob(next_obs).mean(dim=-1)
        else:
            raise NotImplementedError

        if return_predictions:
            return loss_state, state_pred
        else:
            return loss_state

    def compute_rew_reconstruction_loss(self, latent, prev_obs, next_obs, 
                                        action, reward, 
                                        return_predictions=False):
        """
        Compute reward reconstruction loss. (No reduction of loss along batch 
        dimension is done here; sum/avg has to be done outside)
        
        TODO(rsc): Explain the difference between multi-head reward prediction 
        and single-head prediction in the docstring (currently in the comments).
        """
        if self.args.multihead_for_reward:
            # Use multiple heads per reward pred (i.e. per state). 
            # NOTE: This means that we need to be able to enumnerate all states
            # and num_states should therefore be small. Otherwise we will have
            # too many heads, and we are better off sending in the state as 
            # input instead of creating a separate head for each.

            # Use reward decoder to predict reward from encoded latent
            rew_pred = self.reward_decoder(latent, None)
            if self.args.rew_pred_type == 'categorical':
                # Categorical reward
                rew_pred = F.softmax(rew_pred, dim=-1)
            elif self.args.rew_pred_type == 'bernoulli':
                # Bernoulli reward
                rew_pred = torch.sigmoid(rew_pred)

            # Create environment so we can access task information
            env = gym.make(self.args.env_name)
            
            # next_obs.shape = [10, 60, 2] for gridworld; num_steps = 15
            # next_obs.shape = [10, 400, 75] for MazeEnv; num_steps = 100
            # [task_batch_size, num_steps*4 (why?), obs_dim(s)]
            # In utils/evaluation.py, next_obs[1] is called 'traj_len'

            # Use next observations to get the ID of the task we will have 
            # completed by landing in that state (?)
            state_indices = env.task_to_id(next_obs).to(DeviceConfig.DEVICE)

            # Ensure state indices and reward predictions have matching 
            # dimensions; if state_indices has one fewer, we unsqueeze (i.e.
            # adding dimension at the end)
            if state_indices.dim() +1 == rew_pred.dim():
                state_indices = state_indices.unsqueeze(-1)
            elif state_indices.dim() == rew_pred.dim():
                pass
            else: 
                # They should either already match, or we should match by
                # unsqueezing once.
                raise ValueError
            # NOTE: state_indices is now shape:
            #                       [task_batch_size, num_steps, 1]

            # Gather values along final axis (because dim=-1), using task/state
            # IDs as indicies. 
            # NOTE: rew_pred is currently of shape:
            #                       [task_batch_size, num_steps*4, num_states]
            # We use state_indices (which index the states) to choose which
            # reward prediction to use for each item in the batch
            rew_pred = rew_pred.gather(dim=-1, index=state_indices)
            # New shape of rew_pred: 
            #                       [task_batch_size, num_steps*4, 1]
            
            # Depending on the reward prediction type, compute the loss between
            # the reward predictions and the targets
            rew_target = (reward == 1).float()
            if self.args.rew_pred_type == 'deterministic':  # TODO: untested!
                loss_rew = (rew_pred - reward).pow(2).mean(dim=-1)
            elif self.args.rew_pred_type in ['categorical', 'bernoulli']:
                loss_rew = F.binary_cross_entropy(rew_pred, rew_target, 
                                                  reduction='none').mean(dim=-1)
            else:
                raise NotImplementedError
        else:
            # Use one head per reward pred
            # NOTE: This is better when we have too many states to create a new
            # head for each, and we're better of sending in the state as 
            # input.
            rew_pred = self.reward_decoder(latent, next_obs, prev_obs, 
                                           action.float())
            if self.args.rew_pred_type == 'bernoulli':  # TODO: untested!
                rew_pred = torch.sigmoid(rew_pred)
                rew_target = (reward == 1).float()  # TODO: necessary?
                loss_rew = F.binary_cross_entropy(rew_pred, rew_target, 
                                                  reduction='none').mean(dim=-1)
            elif self.args.rew_pred_type == 'deterministic':
                loss_rew = (rew_pred - reward).pow(2).mean(dim=-1)
            else:
                raise NotImplementedError

        if return_predictions:
            return loss_rew, rew_pred
        else:
            return loss_rew

    def compute_task_reconstruction_loss(self, latent, task, 
                                         return_predictions=False):
        """
        Compute task reconstruction loss. (No reduction of loss along batch 
        dimension is done here; sum/avg has to be done outside)

        TODO(rsc): Explain when this function is used--it's not used for Varibad!
        Or even the belief_oracle... so who uses it?
        """
        raise NotImplementedError  # TODO: need to pass env kwargs to gym.make

        task_pred = self.task_decoder(latent)
        if self.args.task_pred_type == 'task_id':
            # Create environment so we can get task information
            env = gym.make(self.args.env_name)
            task_target = env.task_to_id(task).to(DeviceConfig.DEVICE)
            # expand along first axis (number of ELBO terms)
            task_target = task_target.expand(task_pred.shape[:-1]).reshape(-1)
            loss_task = F.cross_entropy(task_pred.view(-1, task_pred.shape[-1]),
                                        task_target, reduction='none').view(
                                        task_pred.shape[:-1])
        elif self.args.task_pred_type == 'task_description':
            loss_task = (task_pred - task).pow(2).mean(dim=-1)
        else:
            raise NotImplementedError

        if return_predictions:
            return loss_task, task_pred
        else:
            return loss_task

    def compute_kl_loss(self, latent_mean, latent_logvar, elbo_indices):
        # -- KL divergence
        if self.args.kl_to_gauss_prior:
            kl_divergences = (- 0.5 * (1 + latent_logvar - latent_mean.pow(2) - 
                                       latent_logvar.exp()).sum(dim=-1))
        else:
            gauss_dim = latent_mean.shape[-1]
            # add the gaussian prior
            all_means = torch.cat((torch.zeros(
                1, *latent_mean.shape[1:]).to(DeviceConfig.DEVICE), latent_mean))
            all_logvars = torch.cat((torch.zeros(
                1, *latent_logvar.shape[1:]).to(DeviceConfig.DEVICE), latent_logvar))
            # https://arxiv.org/pdf/1811.09975.pdf
            # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + 
            #                       (m-mu)^T S^-1 (m-mu)))
            mu = all_means[1:]
            m = all_means[:-1]
            logE = all_logvars[1:]
            logS = all_logvars[:-1]
            kl_divergences = (
                0.5 * (torch.sum(logS, dim=-1) - torch.sum(logE, dim=-1) - 
                       gauss_dim + torch.sum(1 / torch.exp(logS) * 
                       torch.exp(logE), dim=-1) + ((m - mu) / torch.exp(logS) *
                       (m - mu)).sum(dim=-1)))

        # returns, for each ELBO_t term, one KL (so H+1 kl's)
        if elbo_indices is not None:
            batchsize = kl_divergences.shape[-1]
            task_indices = torch.arange(batchsize).repeat(
                self.args.vae_subsample_elbos)
            kl_divergences = kl_divergences[elbo_indices, task_indices].reshape(
                (self.args.vae_subsample_elbos, batchsize))

        return kl_divergences

    def compute_loss(self, latent_mean, latent_logvar, vae_prev_obs, 
                     vae_next_obs, vae_actions,
                     vae_rewards, vae_tasks, trajectory_lens):
        """
        Computes the VAE loss for the given data.
        Batches everything together and therefore needs all trajectories to be 
        of the same length.
        (Important because we need to separate ELBOs and decoding terms so 
        can't collapse those dimensions)
        """
        num_unique_trajectory_lens = len(np.unique(trajectory_lens))

        assert ((num_unique_trajectory_lens == 1) or 
                (self.args.vae_subsample_elbos and 
                 self.args.vae_subsample_decodes))
        assert not self.args.decode_only_past

        # cut down the batch to the longest trajectory length
        # this way we can preserve the structure
        # but we will waste some computation on zero-padded trajectories that 
        # are shorter than max_traj_len
        max_traj_len = np.max(trajectory_lens)
        latent_mean = latent_mean[:max_traj_len + 1]
        latent_logvar = latent_logvar[:max_traj_len + 1]
        if isinstance(vae_prev_obs, dict):
            for key in vae_prev_obs.keys():
                vae_prev_obs[key] = vae_prev_obs[key][:max_traj_len]
                vae_next_obs[key] = vae_next_obs[key][:max_traj_len]
        else:
            vae_prev_obs = vae_prev_obs[:max_traj_len]
            vae_next_obs = vae_next_obs[:max_traj_len]
        vae_actions = vae_actions[:max_traj_len]
        vae_rewards = vae_rewards[:max_traj_len]

        # take one sample for each ELBO term
        if not self.args.disable_stochasticity_in_latent:
            latent_samples = self.encoder._sample_gaussian(latent_mean, 
                                                           latent_logvar)
        else:
            latent_samples = torch.cat((latent_mean, latent_logvar), dim=-1)

        num_elbos = latent_samples.shape[0]
        if isinstance(vae_prev_obs, dict):
            num_decodes = vae_prev_obs[list(vae_prev_obs.keys())[0]].shape[0]
        else: 
            num_decodes = vae_prev_obs.shape[0]
        batchsize = latent_samples.shape[1]  # number of trajectories

        # subsample elbo terms
        #   shape before: num_elbos * batchsize * dim
        #   shape after: vae_subsample_elbos * batchsize * dim
        if self.args.vae_subsample_elbos is not None:
            # randomly choose which elbo's to subsample
            if num_unique_trajectory_lens == 1:
                elbo_indices = torch.LongTensor(
                    self.args.vae_subsample_elbos * batchsize
                    ).random_(0, num_elbos)  # select diff elbos for each task
            else:
                # if we have different trajectory lengths, subsample elbo 
                # indices separately up to their maximum possible encoding 
                # length; only allow duplicates if the sample size would be
                # larger than the number of samples
                elbo_indices = np.concatenate(
                    [np.random.choice(
                        range(0, t + 1), self.args.vae_subsample_elbos,
                        replace=self.args.vae_subsample_elbos > (t+1)) 
                     for t in trajectory_lens])
                if max_traj_len < self.args.vae_subsample_elbos:
                    warnings.warn('The required number of ELBOs is larger than '
                                  'the shortest trajectory, '
                                  'so there will be duplicates in your batch.'
                                  'To avoid this use --split-batches-by-elbo '
                                  'or --split-batches-by-task.')
            task_indices = torch.arange(batchsize).repeat(
                self.args.vae_subsample_elbos)  # for selection mask
            latent_samples = \
                latent_samples[elbo_indices, task_indices, :].reshape(
                    (self.args.vae_subsample_elbos, batchsize, -1))
            num_elbos = latent_samples.shape[0]
        else:
            elbo_indices = None

        # expand the state/rew/action inputs to the decoder 
        # (to match size of latents)
        # shape will be: [num tasks in batch] x [num elbos] x 
        #                [len trajectory (reconstrution loss)] x [dimension]
        if isinstance(vae_prev_obs, dict):
            dec_prev_obs = dict()
            dec_next_obs = dict()
            for key in vae_prev_obs.keys():
                dec_prev_obs[key] = vae_prev_obs[key].unsqueeze(0).expand(
                    (num_elbos, *vae_prev_obs[key].shape))
                dec_next_obs[key] = vae_next_obs[key].unsqueeze(0).expand(
                    (num_elbos, *vae_next_obs[key].shape))
        else:
            dec_prev_obs = vae_prev_obs.unsqueeze(0).expand(
                (num_elbos, *vae_prev_obs.shape))
            dec_next_obs = vae_next_obs.unsqueeze(0).expand(
                (num_elbos, *vae_next_obs.shape))
        dec_actions = vae_actions.unsqueeze(0).expand(
            (num_elbos, *vae_actions.shape))
        dec_rewards = vae_rewards.unsqueeze(0).expand(
            (num_elbos, *vae_rewards.shape))

        # subsample reconstruction terms
        if self.args.vae_subsample_decodes is not None:
            # shape before: vae_subsample_elbos * num_decodes * batchsize * dim
            # shape after: vae_subsample_elbos * vae_subsample_decodes * 
            #              batchsize * dim
            # (Note that this will always have duplicates given how we 
            # set up the code)
            indices0 = torch.arange(num_elbos).repeat(
                self.args.vae_subsample_decodes * batchsize)
            if num_unique_trajectory_lens == 1:
                indices1 = torch.LongTensor(
                    num_elbos * self.args.vae_subsample_decodes * 
                    batchsize).random_(0, num_decodes)
            else:
                indices1 = np.concatenate(
                    [np.random.choice(range(0, t), num_elbos *
                     self.args.vae_subsample_decodes, replace=True) 
                     for t in trajectory_lens])
            indices2 = torch.arange(batchsize).repeat(
                num_elbos * self.args.vae_subsample_decodes)

            if isinstance(dec_prev_obs, dict):
                dec_prev_obs = {
                    key: dec_prev_obs[key][indices0, indices1, indices2, :].reshape(
                        (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
                        for key in dec_prev_obs.keys()}
                dec_next_obs = {
                    key: dec_next_obs[key][indices0, indices1, indices2, :].reshape(
                        (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
                        for key in dec_next_obs.keys()}
            else:
                dec_prev_obs = \
                    dec_prev_obs[indices0, indices1, indices2, :].reshape(
                        (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
                dec_next_obs = \
                    dec_next_obs[indices0, indices1, indices2, :].reshape(
                        (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
            dec_actions = \
                dec_actions[indices0, indices1, indices2, :].reshape(
                    (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
            dec_rewards = \
                dec_rewards[indices0, indices1, indices2, :].reshape(
                    (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
            
            if isinstance(dec_prev_obs, dict):
                num_decodes = dec_prev_obs[list(dec_prev_obs.keys())[0]].shape[1]
            else:
                num_decodes = dec_prev_obs.shape[1]

        # expand the latent (to match the number of state/rew/action inputs 
        # to the decoder)
        # shape will be: [num tasks in batch] x [num elbos] x 
        #                [len trajectory (reconstrution loss)] x [dimension]
        dec_embedding = latent_samples.unsqueeze(0).expand(
            (num_decodes, *latent_samples.shape)).transpose(1, 0)

        if self.args.decode_reward:
            # compute reconstruction loss for this trajectory (for each 
            # timestep that was encoded, decode everything and sum it up)
            # shape: [num_elbo_terms] x [num_reconstruction_terms] x 
            #        [num_trajectories]
            rew_reconstruction_loss = self.compute_rew_reconstruction_loss(
                dec_embedding, dec_prev_obs, dec_next_obs,
                dec_actions, dec_rewards)
            # avg/sum across individual ELBO terms
            if self.args.vae_avg_elbo_terms:
                rew_reconstruction_loss = rew_reconstruction_loss.mean(dim=0)
            else:
                rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=0)
            # avg/sum across individual reconstruction terms
            if self.args.vae_avg_reconstruction_terms:
                rew_reconstruction_loss = rew_reconstruction_loss.mean(dim=0)
            else:
                rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=0)
            # average across tasks
            rew_reconstruction_loss = rew_reconstruction_loss.mean()
        else:
            rew_reconstruction_loss = 0

        if self.args.decode_state:
            state_reconstruction_loss = self.compute_state_reconstruction_loss(
                dec_embedding, dec_prev_obs, dec_next_obs, dec_actions)
            # avg/sum across individual ELBO terms
            if self.args.vae_avg_elbo_terms:
                state_reconstruction_loss = \
                    state_reconstruction_loss.mean(dim=0)
            else:
                state_reconstruction_loss = \
                    state_reconstruction_loss.sum(dim=0)
            # avg/sum across individual reconstruction terms
            if self.args.vae_avg_reconstruction_terms:
                state_reconstruction_loss = \
                    state_reconstruction_loss.mean(dim=0)
            else:
                state_reconstruction_loss = \
                    state_reconstruction_loss.sum(dim=0)
            # average across tasks
            state_reconstruction_loss = state_reconstruction_loss.mean()
        else:
            state_reconstruction_loss = 0

        if self.args.decode_task:
            task_reconstruction_loss = self.compute_task_reconstruction_loss(
                latent_samples, vae_tasks)
            # avg/sum across individual ELBO terms
            if self.args.vae_avg_elbo_terms:
                task_reconstruction_loss = task_reconstruction_loss.mean(dim=0)
            else:
                task_reconstruction_loss = task_reconstruction_loss.sum(dim=0)
            # sum the elbos, average across tasks
            task_reconstruction_loss = \
                task_reconstruction_loss.sum(dim=0).mean()
        else:
            task_reconstruction_loss = 0

        if not self.args.disable_kl_term:
            # compute the KL term for each ELBO term of the current trajectory
            # shape: [num_elbo_terms] x [num_trajectories]
            kl_loss = self.compute_kl_loss(latent_mean, latent_logvar, 
                                           elbo_indices)
            # avg/sum the elbos
            if self.args.vae_avg_elbo_terms:
                kl_loss = kl_loss.mean(dim=0)
            else:
                kl_loss = kl_loss.sum(dim=0)
            # average across tasks
            kl_loss = kl_loss.sum(dim=0).mean()
        else:
            kl_loss = 0

        return (rew_reconstruction_loss, state_reconstruction_loss, 
                task_reconstruction_loss, kl_loss)

    def compute_loss_split_batches_by_elbo(
            self, latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, 
            vae_actions, vae_rewards, vae_tasks, trajectory_lens):
        """
        Loop over the elvo_t terms to compute losses per t.
        Saves some memory if batch sizes are very large,
        or if trajectory lengths are different, or if we decode only the past.
        """

        rew_reconstruction_loss = []
        state_reconstruction_loss = []
        task_reconstruction_loss = []

        assert len(np.unique(trajectory_lens)) == 1
        n_horizon = np.unique(trajectory_lens)[0]
        n_elbos = latent_mean.shape[0]  # includes the prior

        # for each elbo term (including one for the prior)...
        for idx_elbo in range(n_elbos):

            # get the embedding values (size: traj_length+1 * latent_dim; 
            # the +1 is for the prior)
            curr_means = latent_mean[idx_elbo]
            curr_logvars = latent_logvar[idx_elbo]

            # take one sample for each task
            if not self.args.disable_stochasticity_in_latent:
                curr_samples = self.encoder._sample_gaussian(curr_means, 
                                                             curr_logvars)
            else:
                curr_samples = torch.cat((latent_mean, latent_logvar))

            # if the size of what we decode is always the same, we can speed 
            # up creating the batches
            if not self.args.decode_only_past:

                # expand the latent to match the (x, y) pairs of the decoder
                dec_embedding = curr_samples.unsqueeze(0).expand(
                    (n_horizon, *curr_samples.shape))
                dec_embedding_task = curr_samples

                dec_prev_obs = vae_prev_obs
                dec_next_obs = vae_next_obs
                dec_actions = vae_actions
                dec_rewards = vae_rewards

            # otherwise, we unfortunately have to loop!
            # loop through the lengths we are feeding into the encoder for 
            # that trajectory (starting with prior)
            # (these are the different ELBO_t terms)
            else:

                # get the index until which we want to decode
                # (i.e. eithe runtil curr timestep or entire trajectory 
                # including future)
                if self.args.decode_only_past:
                    dec_from = 0
                    dec_until = idx_elbo
                else:
                    dec_from = 0
                    dec_until = n_horizon

                if dec_from == dec_until:
                    continue

                # (1) ... get the latent sample after feeding in some data 
                # (determined by len_encoder) & expand (to number of outputs)
                # num latent samples x embedding size
                dec_embedding = curr_samples.unsqueeze(0).expand(
                    dec_until - dec_from, *curr_samples.shape)
                dec_embedding_task = curr_samples
                # (2) ... get the predictions for the trajectory until the
                # timestep we're interested in
                dec_prev_obs = vae_prev_obs[dec_from:dec_until]
                dec_next_obs = vae_next_obs[dec_from:dec_until]
                dec_actions = vae_actions[dec_from:dec_until]
                dec_rewards = vae_rewards[dec_from:dec_until]

            if self.args.decode_reward:
                # compute reconstruction loss for this trajectory (for each 
                # timestep that was encoded, decode everything and sum it up)
                # size: if all trajectories are of same length 
                # [num_elbo_terms x num_reconstruction_terms], otherwise it's 
                # flattened into one
                rrc = self.compute_rew_reconstruction_loss(
                    dec_embedding, dec_prev_obs, dec_next_obs, dec_actions,
                    dec_rewards)
                # sum up the reconstruction terms; average over tasks
                rrc = rrc.sum(dim=0).mean()
                rew_reconstruction_loss.append(rrc)

            if self.args.decode_state:
                src = self.compute_state_reconstruction_loss(
                    dec_embedding, dec_prev_obs, dec_next_obs, dec_actions)
                # sum up the reconstruction terms; average over tasks
                src = src.sum(dim=0).mean()
                state_reconstruction_loss.append(src)

            if self.args.decode_task:
                trc = self.compute_task_reconstruction_loss(
                    dec_embedding_task, vae_tasks)
                # average across tasks
                trc = trc.mean()
                task_reconstruction_loss.append(trc)

        # sum the ELBO_t terms
        if self.args.decode_reward:
            rew_reconstruction_loss = torch.stack(rew_reconstruction_loss)
            rew_reconstruction_loss = rew_reconstruction_loss.sum()
        else:
            rew_reconstruction_loss = 0

        if self.args.decode_state:
            state_reconstruction_loss = torch.stack(state_reconstruction_loss)
            state_reconstruction_loss = state_reconstruction_loss.sum()
        else:
            state_reconstruction_loss = 0

        if self.args.decode_task:
            task_reconstruction_loss = torch.stack(task_reconstruction_loss)
            task_reconstruction_loss = task_reconstruction_loss.sum()
        else:
            task_reconstruction_loss = 0

        if not self.args.disable_kl_term:
            # compute the KL term for each ELBO term of the current trajectory
            kl_loss = self.compute_kl_loss(latent_mean, latent_logvar, None)
            # sum the elbos, average across tasks
            kl_loss = kl_loss.sum(dim=0).mean()
        else:
            kl_loss = 0

        return (rew_reconstruction_loss, state_reconstruction_loss, 
                task_reconstruction_loss, kl_loss)

    def compute_vae_loss(self, update=False, pretrain_index=None):
        """ Returns the VAE loss """

        if not self.rollout_storage.ready_for_update():
            return 0

        if self.args.disable_decoder and self.args.disable_kl_term:
            return 0

        # get a mini-batch
        vae_prev_obs, vae_next_obs, vae_actions, vae_rewards, vae_tasks, \
        trajectory_lens = self.rollout_storage.get_batch(
            batchsize=self.args.vae_batch_num_trajs)
        # vae_prev_obs will be of size: max trajectory len x num trajectories 
        #                               x dimension of observations

        # pass through encoder (outputs will be: 
        # (max_traj_len+1) x number of rollouts x latent_dim 
        # -- includes the prior!)
        _, latent_mean, latent_logvar, _ = self.encoder(
            actions=vae_actions,
            states=vae_next_obs,
            rewards=vae_rewards,
            hidden_state=None,
            return_prior=True,
            detach_every=self.args.tbptt_stepsize 
                if hasattr(self.args, 'tbptt_stepsize') else None,
        )

        if self.args.split_batches_by_task:
            raise NotImplementedError
            losses = self.compute_loss_split_batches_by_task(
                latent_mean, latent_logvar, vae_prev_obs, vae_next_obs,
                vae_actions, vae_rewards, vae_tasks, trajectory_lens, 
                len_encoder)
        elif self.args.split_batches_by_elbo:
            losses = self.compute_loss_split_batches_by_elbo(
                latent_mean, latent_logvar, vae_prev_obs, vae_next_obs,
                vae_actions, vae_rewards, vae_tasks, trajectory_lens)
        else:
            losses = self.compute_loss(
                latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, 
                vae_actions, vae_rewards, vae_tasks, trajectory_lens)
        (rew_reconstruction_loss, state_reconstruction_loss, 
         task_reconstruction_loss, kl_loss) = losses

        # VAE loss = KL loss + reward reconstruction + state transition 
        # reconstruction
        # take average (this is the expectation over p(M))
        loss = (self.args.rew_loss_coeff * rew_reconstruction_loss +
                self.args.state_loss_coeff * state_reconstruction_loss +
                self.args.task_loss_coeff * task_reconstruction_loss +
                self.args.kl_weight * kl_loss).mean()

        # make sure we can compute gradients
        if not self.args.disable_kl_term:
            assert kl_loss.requires_grad
        if self.args.decode_reward:
            assert rew_reconstruction_loss.requires_grad
        if self.args.decode_state:
            assert state_reconstruction_loss.requires_grad
        if self.args.decode_task:
            assert task_reconstruction_loss.requires_grad

        # overall loss
        elbo_loss = loss.mean()

        if update:
            self.optimiser_vae.zero_grad()
            elbo_loss.backward()
            # clip gradients
            if self.args.encoder_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.encoder.parameters(), 
                                         self.args.encoder_max_grad_norm)
            if self.args.decoder_max_grad_norm is not None:
                if self.args.decode_reward:
                    nn.utils.clip_grad_norm_(self.reward_decoder.parameters(), 
                                             self.args.decoder_max_grad_norm)
                if self.args.decode_state:
                    nn.utils.clip_grad_norm_(self.state_decoder.parameters(), 
                                             self.args.decoder_max_grad_norm)
                if self.args.decode_task:
                    nn.utils.clip_grad_norm_(self.task_decoder.parameters(), 
                                             self.args.decoder_max_grad_norm)
            # update
            self.optimiser_vae.step()

        self.log(elbo_loss, rew_reconstruction_loss, state_reconstruction_loss, 
                 task_reconstruction_loss, kl_loss, pretrain_index)


        return elbo_loss

    def log(self, elbo_loss, rew_reconstruction_loss, state_reconstruction_loss, 
            task_reconstruction_loss, kl_loss, pretrain_index=None):

        if pretrain_index is None:
            curr_iter_idx = self.get_iter_idx()
        else:
            curr_iter_idx = (- self.args.pretrain_len * 
                             self.args.num_vae_updates_per_pretrain + 
                             pretrain_index)

        if curr_iter_idx % self.args.log_interval == 0:

            if self.args.decode_reward:
                self.logger.add('vae_losses/reward_reconstr_err', 
                                rew_reconstruction_loss.mean(), curr_iter_idx)
            if self.args.decode_state:
                self.logger.add('vae_losses/state_reconstr_err', 
                                state_reconstruction_loss.mean(), curr_iter_idx)
            if self.args.decode_task:
                self.logger.add('vae_losses/task_reconstr_err', 
                                task_reconstruction_loss.mean(), curr_iter_idx)

            if not self.args.disable_kl_term:
                self.logger.add('vae_losses/kl', kl_loss.mean(), curr_iter_idx)
            self.logger.add('vae_losses/sum', elbo_loss, curr_iter_idx)
