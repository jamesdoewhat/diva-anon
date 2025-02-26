import numpy as np
import torch
from qd_metarl.utils.torch_utils import DeviceConfig


class DictRolloutStorageVAE(object):
    def __init__(self, 
                 num_processes, 
                 max_trajectory_len, 
                 zero_pad, 
                 max_num_rollouts,
                 state_dim, 
                 action_dim, 
                 vae_buffer_add_thresh, 
                 task_dim, 
                 state_dtype=torch.float32):
        """
        Store everything that is needed for the VAE update
        :param num_processes:
        """
        self.obs_dim = state_dim
        self.action_dim = action_dim
        self.task_dim = task_dim

        self.vae_buffer_add_thresh = vae_buffer_add_thresh  # prob of adding new trajectories
        self.max_buffer_size = max_num_rollouts  # maximum buffer len (number of trajectories)
        self.insert_idx = 0  # at which index we're currently inserting new data
        self.buffer_len = 0  # how much of the buffer has been filled

        # how long a trajectory can be at max (horizon)
        self.max_traj_len = max_trajectory_len
        # whether to zero-pad to maximum length (zero's at the end!)
        self.zero_pad = zero_pad

        # buffers for completed rollouts (stored on CPU)
        if self.max_buffer_size > 0:
            print('Dims: ', [v for _, v in state_dim.items()])
            print('Max traj len: ', self.max_traj_len)
            print('Max buffer size: ', self.max_buffer_size)
            self.prev_state = {k: torch.zeros((self.max_traj_len, self.max_buffer_size, *v), dtype=state_dtype) for k, v in state_dim.items()}
            self.next_state = {k: torch.zeros((self.max_traj_len, self.max_buffer_size, *v), dtype=state_dtype) for k, v in state_dim.items()}
            self.actions = torch.zeros((self.max_traj_len, self.max_buffer_size, self.action_dim))
            self.rewards = torch.zeros((self.max_traj_len, self.max_buffer_size, 1))
            if task_dim is not None:
                self.tasks = torch.zeros((self.max_buffer_size, task_dim))
            else:
                self.tasks = None
            self.trajectory_lens = [0] * self.max_buffer_size

        # storage for each running process (stored on GPU)
        self.num_processes = num_processes
        self.curr_timestep = torch.zeros((num_processes)).long()  # count environment steps so we know where to insert
        self.running_prev_state = {k: torch.zeros((self.max_traj_len, num_processes, *v), dtype=state_dtype).to(DeviceConfig.DEVICE) for k, v in state_dim.items()}  # for each episode will have obs 0...N-1
        self.running_next_state = {k: torch.zeros((self.max_traj_len, num_processes, *v), dtype=state_dtype).to(DeviceConfig.DEVICE) for k, v in state_dim.items()}  # for each episode will have obs 1...N
        self.running_rewards = torch.zeros((self.max_traj_len, num_processes, 1)).to(DeviceConfig.DEVICE)
        self.running_actions = torch.zeros((self.max_traj_len, num_processes, action_dim)).to(DeviceConfig.DEVICE)
        if task_dim is not None:
            self.running_tasks = torch.zeros((num_processes, task_dim)).to(DeviceConfig.DEVICE)
        else:
            self.running_tasks = None

    def get_running_batch(self):
        """
        Returns the batch of data from the current running environments
        (zero-padded to maximal trajectory length since different processes can have different trajectory lengths)
        :return:
        """
        return self.running_prev_state, self.running_next_state, self.running_actions, self.running_rewards, self.curr_timestep

    def insert(self, prev_state, actions, next_state, rewards, done, task):

        # add to temporary buffer

        already_inserted = False
        if len(np.unique(self.curr_timestep)) == 1:
            for key in prev_state:
                self.running_prev_state[key][self.curr_timestep[0]] = prev_state[key]
            for key in next_state:
                self.running_next_state[key][self.curr_timestep[0]] = next_state[key]
            self.running_rewards[self.curr_timestep[0]] = rewards
            self.running_actions[self.curr_timestep[0]] = actions
            if task is not None:
                self.running_tasks = task
            self.curr_timestep += 1
            already_inserted = True

        already_reset = False
        if done.sum() == self.num_processes:  # check if we can process the entire batch at once

            # add to permanent (up to max_buffer_len) buffer
            if self.max_buffer_size > 0:
                if self.vae_buffer_add_thresh >= np.random.uniform(0, 1):
                    # check where to insert data
                    if self.insert_idx + self.num_processes > self.max_buffer_size:
                        # keep track of how much we filled the buffer (for sampling from it)
                        self.buffer_len = self.insert_idx
                        # this will keep some entries at the end of the buffer without overwriting them,
                        # but the buffer is large enough to make this negligible
                        self.insert_idx = 0
                    else:
                        self.buffer_len = max(self.buffer_len, self.insert_idx)
                    # add; note: num trajectories are along dim=1,
                    # trajectory length along dim=0, to match pytorch RNN interface
                    for key in prev_state:
                        self.prev_state[key][:, self.insert_idx:self.insert_idx + self.num_processes] = self.running_prev_state[key]
                    for key in next_state:
                        self.next_state[key][:, self.insert_idx:self.insert_idx + self.num_processes] = self.running_next_state[key]
                    self.actions[:, self.insert_idx:self.insert_idx+self.num_processes] = self.running_actions
                    self.rewards[:, self.insert_idx:self.insert_idx+self.num_processes] = self.running_rewards
                    if (self.tasks is not None) and (self.running_tasks is not None):
                        insert_shape = self.tasks[self.insert_idx:self.insert_idx+self.num_processes].shape
                        self.tasks[self.insert_idx:self.insert_idx+self.num_processes] = self.running_tasks.reshape(insert_shape)
                    self.trajectory_lens[self.insert_idx:self.insert_idx+self.num_processes] = self.curr_timestep.clone()
                    self.insert_idx += self.num_processes

            # empty running buffer
            for key in prev_state:
                self.running_prev_state[key] *= 0
            for key in next_state:
                self.running_next_state[key] *= 0
            self.running_rewards *= 0
            self.running_actions *= 0
            if self.running_tasks is not None:
                self.running_tasks *= 0
            self.curr_timestep *= 0

            already_reset = True


        if (not already_inserted) or (not already_reset):

            for i in range(self.num_processes):

                if not already_inserted:
                    for key in prev_state[i]:
                        self.running_prev_state[key][self.curr_timestep[i]] = prev_state[i][key]
                        self.running_next_state[key][self.curr_timestep[i]] = next_state[i][key]
                        self.running_rewards[key][self.curr_timestep[i]] = rewards[i][key]
                        self.running_actions[key][self.curr_timestep[i]] = actions[i][key]
                    if self.running_tasks is not None:
                        self.running_tasks[i] = task[i]
                    self.curr_timestep[i] += 1

                if not already_reset:
                    # if we are at the end of a task, dump the data into the larger buffer
                    if done[i]:

                        # add to permanent (up to max_buffer_len) buffer
                        if self.max_buffer_size > 0:
                            if self.vae_buffer_add_thresh >= np.random.uniform(0, 1):
                                # check where to insert data
                                if self.insert_idx + 1 > self.max_buffer_size:
                                    # keep track of how much we filled the buffer (for sampling from it)
                                    self.buffer_len = self.insert_idx
                                    # this will keep some entries at the end of the buffer without overwriting them,
                                    # but the buffer is large enough to make this negligible
                                    self.insert_idx = 0
                                else:
                                    self.buffer_len = max(self.buffer_len, self.insert_idx)
                                # add; note: num trajectories are along dim=1,
                                # trajectory length along dim=0, to match pytorch RNN interface
                                for key in self.prev_state:
                                    self.prev_state[key][:, self.insert_idx] = self.running_prev_state[key].to('cpu')
                                    self.next_state[key][:, self.insert_idx] = self.running_next_state[key].to('cpu')
                                    self.actions[key][:, self.insert_idx] = self.running_actions[key].to('cpu')
                                    self.rewards[key][:, self.insert_idx] = self.running_rewards[key].to('cpu')
                                if self.tasks is not None:
                                    self.tasks[self.insert_idx] = self.running_tasks[i].to('cpu')
                                self.trajectory_lens[self.insert_idx] = self.curr_timestep[i].clone()
                                self.insert_idx += 1

                        # empty running buffer
                        for key in self.running_prev_state:
                            self.running_prev_state[key][:, i] *= 0
                            self.running_next_state[key][:, i] *= 0
                            self.running_rewards[key][:, i] *= 0
                            self.running_actions[key][:, i] *= 0
                        if self.running_tasks is not None:
                            self.running_tasks[i] *= 0
                        self.curr_timestep[i] = 0


    def ready_for_update(self):
        return len(self) > 0

    def __len__(self):
        return self.buffer_len

    def get_batch(self, batchsize=5, replace=False):
        # TODO: check if we can get rid of num_enc_len and num_rollouts (call it batchsize instead)

        batchsize = min(self.buffer_len, batchsize)

        # select the indices for the processes from which we pick
        rollout_indices = np.random.choice(range(self.buffer_len), batchsize, replace=replace)
        # trajectory length of the individual rollouts we picked
        trajectory_lens = np.array(self.trajectory_lens)[rollout_indices]

        # select the rollouts we want
        prev_obs = dict()
        next_obs = dict()
        for key in self.prev_state:
            prev_obs[key] = self.prev_state[key][:, rollout_indices, :].to(DeviceConfig.DEVICE)
            next_obs[key] = self.next_state[key][:, rollout_indices, :].to(DeviceConfig.DEVICE)
        actions = self.actions[:, rollout_indices, :].to(DeviceConfig.DEVICE)
        rewards = self.rewards[:, rollout_indices, :].to(DeviceConfig.DEVICE)
        if self.tasks is not None:
            tasks = self.tasks[rollout_indices].to(DeviceConfig.DEVICE)
        else:
            tasks = None

        return prev_obs, next_obs, actions, rewards, tasks, trajectory_lens


class RolloutStorageVAE(object):
    def __init__(self, 
                 num_processes, 
                 max_trajectory_len, 
                 zero_pad, 
                 max_num_rollouts,
                 state_dim, 
                 action_dim, 
                 vae_buffer_add_thresh, 
                 task_dim,
                 state_dtype=torch.float32):
        """
        Store everything that is needed for the VAE update
        :param num_processes:
        """
        # Check if state_dim is an int or tuple
        if isinstance(state_dim, tuple):
            # Tuple
            self.obs_dim = state_dim[-1]  # Assuming (height, width, channels)
            self.obs_shape = state_dim
        else:
            # Int
            self.obs_dim = state_dim
            self.obs_shape = (state_dim,)

        self.action_dim = action_dim
        self.task_dim = task_dim

        self.vae_buffer_add_thresh = vae_buffer_add_thresh  # prob of adding new trajectories
        self.max_buffer_size = max_num_rollouts  # maximum buffer len (number of trajectories)
        self.insert_idx = 0  # at which index we're currently inserting new data
        self.buffer_len = 0  # how much of the buffer has been filled

        # how long a trajectory can be at max (horizon)
        self.max_traj_len = max_trajectory_len
        # whether to zero-pad to maximum length (zero's at the end!)
        self.zero_pad = zero_pad

        print('Dim: ', state_dim)
        print('Max traj len: ', self.max_traj_len)
        print('Max buffer size: ', self.max_buffer_size)

        # buffers for completed rollouts (stored on CPU)
        if self.max_buffer_size > 0:
            self.prev_state = torch.zeros((self.max_traj_len, self.max_buffer_size, *self.obs_shape), dtype=state_dtype)
            self.next_state = torch.zeros((self.max_traj_len, self.max_buffer_size, *self.obs_shape), dtype=state_dtype)
            self.actions = torch.zeros((self.max_traj_len, self.max_buffer_size, action_dim))
            self.rewards = torch.zeros((self.max_traj_len, self.max_buffer_size, 1))
            if task_dim is not None:
                self.tasks = torch.zeros((self.max_buffer_size, task_dim))
            else:
                self.tasks = None
            self.trajectory_lens = [0] * self.max_buffer_size

        # storage for each running process (stored on GPU)
        self.num_processes = num_processes
        self.curr_timestep = torch.zeros((num_processes)).long()  # count environment steps so we know where to insert
        self.running_prev_state = torch.zeros((self.max_traj_len, num_processes, *self.obs_shape), dtype=state_dtype).to(DeviceConfig.DEVICE)  # for each episode will have obs 0...N-1
        self.running_next_state = torch.zeros((self.max_traj_len, num_processes, *self.obs_shape), dtype=state_dtype).to(DeviceConfig.DEVICE)  # for each episode will have obs 1...N 
        self.running_rewards = torch.zeros((self.max_traj_len, num_processes, 1)).to(DeviceConfig.DEVICE)
        self.running_actions = torch.zeros((self.max_traj_len, num_processes, action_dim)).to(DeviceConfig.DEVICE)
        if task_dim is not None:
            self.running_tasks = torch.zeros((num_processes, task_dim)).to(DeviceConfig.DEVICE)
        else:
            self.running_tasks = None

    def get_running_batch(self):
        """
        Returns the batch of data from the current running environments
        (zero-padded to maximal trajectory length since different processes can have different trajectory lengths)
        :return:
        """
        return self.running_prev_state, self.running_next_state, self.running_actions, self.running_rewards, self.curr_timestep

    def insert(self, prev_state, actions, next_state, rewards, done, task):

        # add to temporary buffer

        already_inserted = False
        if len(np.unique(self.curr_timestep)) == 1:
            self.running_prev_state[self.curr_timestep[0]] = prev_state
            self.running_next_state[self.curr_timestep[0]] = next_state
            self.running_rewards[self.curr_timestep[0]] = rewards
            self.running_actions[self.curr_timestep[0]] = actions
            if task is not None:
                self.running_tasks = task
            self.curr_timestep += 1
            already_inserted = True

        already_reset = False
        if done.sum() == self.num_processes:  # check if we can process the entire batch at once

            # add to permanent (up to max_buffer_len) buffer
            if self.max_buffer_size > 0:
                if self.vae_buffer_add_thresh >= np.random.uniform(0, 1):
                    # check where to insert data
                    if self.insert_idx + self.num_processes > self.max_buffer_size:
                        # keep track of how much we filled the buffer (for sampling from it)
                        self.buffer_len = self.insert_idx
                        # this will keep some entries at the end of the buffer without overwriting them,
                        # but the buffer is large enough to make this negligible
                        self.insert_idx = 0
                    else:
                        self.buffer_len = max(self.buffer_len, self.insert_idx)
                    # add; note: num trajectories are along dim=1,
                    # trajectory length along dim=0, to match pytorch RNN interface
                    self.prev_state[:, self.insert_idx:self.insert_idx + self.num_processes] = self.running_prev_state
                    self.next_state[:, self.insert_idx:self.insert_idx + self.num_processes] = self.running_next_state
                    self.actions[:, self.insert_idx:self.insert_idx+self.num_processes] = self.running_actions
                    self.rewards[:, self.insert_idx:self.insert_idx+self.num_processes] = self.running_rewards
                    if (self.tasks is not None) and (self.running_tasks is not None):
                        insert_shape = self.tasks[self.insert_idx:self.insert_idx+self.num_processes].shape
                        self.tasks[self.insert_idx:self.insert_idx+self.num_processes] = self.running_tasks.reshape(insert_shape)
                    self.trajectory_lens[self.insert_idx:self.insert_idx+self.num_processes] = self.curr_timestep.clone()
                    self.insert_idx += self.num_processes

            # empty running buffer
            self.running_prev_state *= 0
            self.running_next_state *= 0
            self.running_rewards *= 0
            self.running_actions *= 0
            if self.running_tasks is not None:
                self.running_tasks *= 0
            self.curr_timestep *= 0

            already_reset = True

        if (not already_inserted) or (not already_reset):

            for i in range(self.num_processes):

                if not already_inserted:
                    self.running_prev_state[self.curr_timestep[i], i] = prev_state[i]
                    self.running_next_state[self.curr_timestep[i], i] = next_state[i]
                    self.running_rewards[self.curr_timestep[i], i] = rewards[i]
                    self.running_actions[self.curr_timestep[i], i] = actions[i]
                    if self.running_tasks is not None:
                        self.running_tasks[i] = task[i]
                    self.curr_timestep[i] += 1

                if not already_reset:
                    # if we are at the end of a task, dump the data into the larger buffer
                    if done[i]:

                        # add to permanent (up to max_buffer_len) buffer
                        if self.max_buffer_size > 0:
                            if self.vae_buffer_add_thresh >= np.random.uniform(0, 1):
                                # check where to insert data
                                if self.insert_idx + 1 > self.max_buffer_size:
                                    # keep track of how much we filled the buffer (for sampling from it)
                                    self.buffer_len = self.insert_idx
                                    # this will keep some entries at the end of the buffer without overwriting them,
                                    # but the buffer is large enough to make this negligible
                                    self.insert_idx = 0
                                else:
                                    self.buffer_len = max(self.buffer_len, self.insert_idx)
                                # add; note: num trajectories are along dim=1,
                                # trajectory length along dim=0, to match pytorch RNN interface
                                self.prev_state[:, self.insert_idx] = self.running_prev_state[:, i].to('cpu')
                                self.next_state[:, self.insert_idx] = self.running_next_state[:, i].to('cpu')
                                self.actions[:, self.insert_idx] = self.running_actions[:, i].to('cpu')
                                self.rewards[:, self.insert_idx] = self.running_rewards[:, i].to('cpu')
                                if self.tasks is not None:
                                    self.tasks[self.insert_idx] = self.running_tasks[i].to('cpu')
                                self.trajectory_lens[self.insert_idx] = self.curr_timestep[i].clone()
                                self.insert_idx += 1

                        # empty running buffer
                        self.running_prev_state[:, i] *= 0
                        self.running_next_state[:, i] *= 0
                        self.running_rewards[:, i] *= 0
                        self.running_actions[:, i] *= 0
                        if self.running_tasks is not None:
                            self.running_tasks[i] *= 0
                        self.curr_timestep[i] = 0

    def ready_for_update(self):
        return len(self) > 0

    def __len__(self):
        return self.buffer_len

    def get_batch(self, batchsize=5, replace=False):
        # TODO: check if we can get rid of num_enc_len and num_rollouts (call it batchsize instead)

        batchsize = min(self.buffer_len, batchsize)

        # select the indices for the processes from which we pick
        rollout_indices = np.random.choice(range(self.buffer_len), batchsize, replace=replace)
        # trajectory length of the individual rollouts we picked
        trajectory_lens = np.array(self.trajectory_lens)[rollout_indices]

        # select the rollouts we want
        prev_obs = self.prev_state[:, rollout_indices, :]
        next_obs = self.next_state[:, rollout_indices, :]
        actions = self.actions[:, rollout_indices, :]
        rewards = self.rewards[:, rollout_indices, :]
        if self.tasks is not None:
            tasks = self.tasks[rollout_indices].to(DeviceConfig.DEVICE)
        else:
            tasks = None

        return prev_obs.to(DeviceConfig.DEVICE), next_obs.to(DeviceConfig.DEVICE), actions.to(DeviceConfig.DEVICE), \
               rewards.to(DeviceConfig.DEVICE), tasks, trajectory_lens
