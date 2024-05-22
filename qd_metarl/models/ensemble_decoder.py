import torch
import torch.nn as nn
from torch.nn import functional as F

from qd_metarl.utils import env_utils as utl
from qd_metarl.utils.torch_utils import DeviceConfig


class EnsembleStateTransitionDecoder(nn.Module):
    def __init__(self, 
                 args, 
                 layers, 
                 latent_dim, 
                 action_dim, 
                 action_embed_dim, 
                 state_dim, 
                 state_embed_dim, 
                 pred_type='deterministic', 
                 state_feature_extractor=utl.FeatureExtractor, 
                 ensemble_size=1):
        super(EnsembleStateTransitionDecoder, self).__init__()

        self.args = args
        self.ensemble_size = ensemble_size
        self.latent_dim = latent_dim

        self.state_encoder = state_feature_extractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)

        # Adjust input dimension to account for the ensemble
        curr_input_dim = latent_dim * ensemble_size + state_embed_dim + action_embed_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        # Output layers for both decoders
        if pred_type == 'gaussian':
            self.fc_out_full = nn.Linear(curr_input_dim, 2 * state_dim)
            self.fc_out_partial = nn.Linear(curr_input_dim, 2 * state_dim)
        else:
            self.fc_out_full = nn.Linear(curr_input_dim, state_dim)
            self.fc_out_partial = nn.Linear(curr_input_dim, state_dim)

    def forward(self, latent_state, state, actions, omit_idx=None):
        if actions is not None:
            actions = utl.squash_action(actions, self.args)
        ha = self.action_encoder(actions)
        hs = self.state_encoder(state)

        # Reshape latent_state to (BS, latent_dim, ensemble_size) for processing
        latent_state = latent_state.reshape(*latent_state.shape[:-1], self.latent_dim * self.ensemble_size)  # Adjust 'latent_dim' as necessary

        # Full latent representation
        h_full = torch.cat((latent_state, hs, ha), dim=-1)
        for layer in self.fc_layers:
            h_full = F.relu(layer(h_full))
        out_full = self.fc_out_full(h_full)

        # Modify latent_state for partial decoder by zeroing out omitted latent
        if omit_idx is not None:
            latent_state_partial = latent_state.clone()
            latent_state_partial[:, :, omit_idx] = 0
        else:
            latent_state_partial = latent_state

        h_partial = torch.cat((latent_state_partial, hs, ha), dim=-1)
        for layer in self.fc_layers:
            h_partial = F.relu(layer(h_partial))
        out_partial = self.fc_out_partial(h_partial)

        return out_full, out_partial


class EnsembleRewardDecoder(nn.Module):
    def __init__(self,
                 args,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 num_states,
                 multi_head=False,
                 pred_type='deterministic',
                 input_prev_state=True,
                 input_action=True,
                 state_feature_extractor=utl.FeatureExtractor,
                 ensemble_size=1
                 ):
        super(EnsembleRewardDecoder, self).__init__()

        self.args = args
        self.multi_head = multi_head
        self.input_prev_state = input_prev_state
        self.input_action = input_action
        self.ensemble_size = ensemble_size
        self.latent_dim = latent_dim

        # Different initialization based on multi_head flag
        if self.multi_head:
            # One output head per state to predict rewards
            curr_input_dim = latent_dim * ensemble_size
            self.fc_layers = nn.ModuleList([])
            for i in range(len(layers)):
                self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
                curr_input_dim = layers[i]
            self.fc_out_full = nn.Linear(curr_input_dim, num_states)
            self.fc_out_partial = nn.Linear(curr_input_dim, num_states)
        else:
            # Get state as input and predict reward probability
            self.state_encoder = state_feature_extractor(state_dim, state_embed_dim, F.relu)
            if self.input_action:
                self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)
            curr_input_dim = latent_dim * ensemble_size + state_embed_dim
            if input_prev_state:
                curr_input_dim += state_embed_dim
            if input_action:
                curr_input_dim += action_embed_dim
            self.fc_layers = nn.ModuleList([])
            for i in range(len(layers)):
                self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
                curr_input_dim = layers[i]

            # Output layers for both decoders
            if pred_type == 'gaussian':
                self.fc_out_full = nn.Linear(curr_input_dim, 2)
                self.fc_out_partial = nn.Linear(curr_input_dim, 2)
            else:
                self.fc_out_full = nn.Linear(curr_input_dim, 1)
                self.fc_out_partial = nn.Linear(curr_input_dim, 1)

    def forward(self, latent_state, next_state, prev_state=None, actions=None, omit_idx=None):
        # we do the action-normalisation (the the env bounds) here
        if actions is not None:
            actions = utl.squash_action(actions, self.args)
        
        # Reshape latent_state to (BS, latent_dim, ensemble_size) for processing
        latent_state = latent_state.reshape(*latent_state.shape[:-1], self.latent_dim, self.ensemble_size)

        # Modify latent_state for partial decoder by zeroing out omitted latent
        latent_state_partial = latent_state.clone()
        if omit_idx is not None:
            latent_state_partial[:, :, omit_idx] = 0

        # Reshape back to original latent shape
        latent_state = latent_state.reshape(*latent_state.shape[:-2], -1)
        latent_state_partial = latent_state_partial.reshape(*latent_state_partial.shape[:-2], -1)

        # Construct input for full and partial decoders
        h_full = self.construct_input(latent_state, next_state, prev_state, actions)
        h_partial = self.construct_input(latent_state_partial, next_state, prev_state, actions)

        # Process through fully connected layers
        for i in range(len(self.fc_layers)):
            h_full = F.relu(self.fc_layers[i](h_full))
            h_partial = F.relu(self.fc_layers[i](h_partial))

        # Output from full and partial decoders
        out_full = self.fc_out_full(h_full)
        out_partial = self.fc_out_partial(h_partial)

        return out_full, out_partial

    def construct_input(self, latent_state, next_state, prev_state, actions):
        h = latent_state.clone()
        if not self.multi_head:
            hns = self.state_encoder(next_state)
            if len(hns.shape) == 4 and len(latent_state.shape) == 3:
                print('WARNING: shape mismatch in RewardDecoder.forward()')
                hns = hns.unsqueeze(2)
            h = torch.cat((h, hns), dim=-1)
            if self.input_action:
                ha = self.action_encoder(actions)
                h = torch.cat((h, ha), dim=-1)
            if self.input_prev_state:
                hps = self.state_encoder(prev_state)
                h = torch.cat((h, hps), dim=-1)
        return h


class EnsembleTaskDecoder(nn.Module):
    def __init__(self, 
                 layers, 
                 latent_dim, 
                 pred_type, 
                 task_dim, 
                 num_tasks, 
                 ensemble_size=1):
        super(EnsembleTaskDecoder, self).__init__()

        # "task_description" or "task id"
        self.pred_type = pred_type
        self.ensemble_size = ensemble_size
        self.latent_dim = latent_dim

        # Adjust input dimension to account for the ensemble
        curr_input_dim = latent_dim * ensemble_size
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        output_dim = task_dim if pred_type == 'task_description' else num_tasks
        self.fc_out_full = nn.Linear(curr_input_dim, output_dim)
        self.fc_out_partial = nn.Linear(curr_input_dim, output_dim)

    def forward(self, latent_state, omit_idx=None):
        # Reshape latent_state to (BS, latent_dim, ensemble_size) for processing
        latent_state = latent_state.view(-1, self.latent_dim, self.ensemble_size)

        # Modify latent_state for partial decoder by zeroing out omitted latent
        latent_state_partial = latent_state.clone()
        if omit_idx is not None:
            latent_state_partial[:, :, omit_idx] = 0

        # Full latent representation
        h_full = latent_state
        for layer in self.fc_layers:
            h_full = F.relu(layer(h_full))
        out_full = self.fc_out_full(h_full)

        # Partial latent representation
        h_partial = latent_state_partial
        for layer in self.fc_layers:
            h_partial = F.relu(layer(h_partial))
        out_partial = self.fc_out_partial(h_partial)

        return out_full, out_partial
