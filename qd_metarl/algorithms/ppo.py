import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import traceback

from qd_metarl.utils import env_utils as utl


class PPO:
    def __init__(self,
                 args,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 policy_optimiser,
                 policy_anneal_lr,
                 train_steps,
                 optimiser_vae=None,
                 lr=None,
                 clip_param=0.2,
                 ppo_epoch=5,
                 num_mini_batch=5,
                 eps=None,
                 use_huber_loss=True,
                 use_clipped_value_loss=True,
                 ):
        self.args = args

        # the model
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_huber_loss = use_huber_loss

        # optimiser
        if policy_optimiser == 'adam':
            self.optimiser = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        elif policy_optimiser == 'rmsprop':
            self.optimiser = optim.RMSprop(actor_critic.parameters(), lr=lr, eps=eps, alpha=0.99)
        self.optimiser_vae = optimiser_vae

        self.lr_scheduler_policy = None
        self.lr_scheduler_encoder = None
        if policy_anneal_lr:
            lam = lambda f: 1 - f / train_steps
            self.lr_scheduler_policy = optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda=lam)
            if hasattr(self.args, 'rlloss_through_encoder') and self.args.rlloss_through_encoder:
                self.lr_scheduler_encoder = optim.lr_scheduler.LambdaLR(self.optimiser_vae, lr_lambda=lam)

    def update(self,
               policy_storage,
               encoder=None,  # variBAD encoder
               rlloss_through_encoder=False,  # whether or not to backprop RL loss through encoder
               compute_vae_loss=None  # function that can compute the VAE loss
               ):
        
        if policy_storage.use_popart:
            value_preds = policy_storage.denorm_value_preds
        else:
            value_preds = policy_storage.value_preds


        # -- get action values --
        advantages = policy_storage.returns[:-1] - value_preds[:-1]
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # if this is true, we will update the VAE at every PPO update
        # otherwise, we update it after we update the policy
        if rlloss_through_encoder:
            # recompute embeddings (to build computation graph)
            utl.recompute_embeddings(policy_storage, 
                                     encoder, 
                                     sample=False, 
                                     update_idx=0,
                                     detach_every=self.args.tbptt_stepsize 
                                     if hasattr(self.args, 'tbptt_stepsize') 
                                     else None)

        # update the normalisation parameters of policy inputs before updating
        self.actor_critic.update_rms(args=self.args, policy_storage=policy_storage)

        # call this to make sure that the action_log_probs are computed
        # (needs to be done right here because of some caching thing when normalising actions)
        policy_storage.before_update(self.actor_critic)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        loss_epoch = 0
        for e in range(self.ppo_epoch):

            data_generator = policy_storage.feed_forward_generator(advantages, self.num_mini_batch)
            for sample in data_generator:

                state_batch, belief_batch, task_batch, \
                actions_batch, latent_sample_batch, latent_mean_batch, latent_logvar_batch, value_preds_batch, \
                return_batch, old_action_log_probs_batch, adv_targ = sample

                if not rlloss_through_encoder:
                    # TODO: implement util functions that will do dict logic
                    # automatically, so code is cleaner
                    if isinstance(state_batch, dict):
                        for key in state_batch:
                            state_batch[key] = state_batch[key].detach()
                    else:
                        state_batch = state_batch.detach()
                    if latent_sample_batch is not None:
                        latent_sample_batch = latent_sample_batch.detach()
                        latent_mean_batch = latent_mean_batch.detach()
                        latent_logvar_batch = latent_logvar_batch.detach()

                latent_batch = utl.get_latent_for_policy(
                    args=self.args, latent_sample=latent_sample_batch,
                    latent_mean=latent_mean_batch, latent_logvar=latent_logvar_batch)

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, action_log_dist, dist_entropy = \
                    self.actor_critic.evaluate_actions(
                        state=state_batch, latent=latent_batch,
                        belief=belief_batch, task=task_batch,
                        action=actions_batch)
                
                og_log_prob_diff = action_log_probs - old_action_log_probs_batch
                log_prob_diff = torch.clamp(og_log_prob_diff, min=-20, max=20)
                ratio = torch.exp(log_prob_diff)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if policy_storage.use_popart:
                    self.actor_critic.popart.update(return_batch)
                    return_batch = self.actor_critic.popart.normalize(return_batch)

                if self.use_huber_loss and self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = F.smooth_l1_loss(values, return_batch, reduction='none')
                    value_losses_clipped = F.smooth_l1_loss(value_pred_clipped, return_batch, reduction='none')
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                elif self.use_huber_loss:
                    value_loss = F.smooth_l1_loss(values, return_batch)
                elif self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # zero out the gradients
                self.optimiser.zero_grad()
                if rlloss_through_encoder:
                    self.optimiser_vae.zero_grad()

                # compute policy loss and backprop
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

                # compute vae loss and backprop
                if rlloss_through_encoder:
                    loss += self.args.vae_loss_coeff * compute_vae_loss()

                # compute gradients (will attach to all networks involved in this computation)
                try:
                    loss.backward()
                except Exception as exc:
                    # DEBUG: Log all relevant information for debugging purposes
                    print('Use CLIPPED value loss:', self.use_clipped_value_loss)
                    print('abs-max VALUE PREDS batch:', torch.max(torch.abs(value_preds_batch)))
                    print('abs-max RETURN batch:', torch.max(torch.abs(return_batch)))
                    print('abs-max VALUES values:', torch.max(torch.abs(values)))
                    print('abs-max ACTION LOG PROBS:', torch.max(torch.abs(action_log_probs)))
                    print('abs-max OLD ACTION LOG PROBS:', torch.max(torch.abs(old_action_log_probs_batch)))
                    print('abs-max OG LOG PROB DIFF:', torch.max(torch.abs(og_log_prob_diff)))
                    print('loss:', loss)
                    print('value_loss:', value_loss)
                    print('action_loss:', action_loss)
                    print('dist_entropy:', dist_entropy)
                    print('PPO epoch:', e)
                    # raise exc
                    print('exception:', exc)
                    print('traceback:', traceback.format_exc())


                # clip gradients
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.policy_max_grad_norm)
                if rlloss_through_encoder:
                    if self.args.encoder_max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(encoder.parameters(), self.args.encoder_max_grad_norm)

                # update
                self.optimiser.step()
                if rlloss_through_encoder:
                    self.optimiser_vae.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                loss_epoch += loss.item()

                if rlloss_through_encoder:
                    # recompute embeddings (to build computation graph)
                    utl.recompute_embeddings(
                        policy_storage, 
                        encoder, 
                        sample=False, 
                        update_idx=e + 1,                     
                        detach_every=self.args.tbptt_stepsize 
                            if hasattr(self.args, 'tbptt_stepsize') else None)

        if (not rlloss_through_encoder) and (self.optimiser_vae is not None):
            for _ in range(self.args.num_vae_updates):
                compute_vae_loss(update=True)

        if self.lr_scheduler_policy is not None:
            self.lr_scheduler_policy.step()
        if self.lr_scheduler_encoder is not None:
            self.lr_scheduler_encoder.step()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, loss_epoch

    def act(self, state, latent, belief, task, deterministic=False):
        return self.actor_critic.act(state=state, latent=latent, belief=belief, task=task, deterministic=deterministic)
