import argparse
import os
import pickle
# import pickle5 as pickle
import random
import warnings
from distutils.util import strtobool
import time
import cv2
from io import BytesIO
import uuid
from moviepy.editor import ImageSequenceClip
from PIL import Image
import psutil

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.autograd.set_detect_anomaly(True)

from qd_metarl.utils.torch_utils import tensor
from qd_metarl.utils.torch_utils import DeviceConfig


def reset_env(env, args, indices=None, state=None, task=None, eval=False, num_processes=None):
    # if indices is not None:
    #    print('HLP - resetting envs with indices: ', indices)

    if num_processes is None:
        num_processes = args.num_processes
    """ env can be many environments or just one """
    # reset all environments
    if (indices is None) or (len(indices) == num_processes):
        # state = env.reset().float().to(device)
        if task is None:
            # For backwards compatibility
            reset_ret = env.reset()
        else:
            reset_ret = env.reset(task=task)

        # TODO: why can't we just always expect level seeds? I suppose
        # this would require all envs to return level_seeds, but can't we just
        # do this in the varibad env wrapper?
        if args.use_plr and not eval:  # No level_seeds in eval
            state, level_seeds = reset_ret
        else:
            state, level_seeds = reset_ret, None

        state = tensor(state, device=DeviceConfig.DEVICE)
    # reset only the ones given by indices
    else:
        assert state is not None
        if task is not None:
            assert len(indices) == len(task)
        # Iterate through indices and reset corresponding envs.
        for t, i in enumerate(indices):
            if task is None:
                reset_ret = env.reset(index=i)
            else:
                reset_ret = env.reset(index=i, task=task[t])
            
            if args.use_plr and not eval:  # No level_seeds in eval
                state[i], level_seeds = reset_ret  
            else:
                state[i], level_seeds = reset_ret, None
    
    # TODO: Not sure if this is correct; modified from original call to get_belief (see history)
    belief = (torch.from_numpy(np.array(env.get_belief)).float().to(DeviceConfig.DEVICE) 
              if args.pass_belief_to_policy else None)
    
    # If task is passed in, we don't need to retrieve it from the env
    if task is None:
        task = torch.from_numpy(env.get_task()).float().to(DeviceConfig.DEVICE) 
                
    
    return state, belief, task, level_seeds



def squash_action(action, args):
    if args.norm_actions_post_sampling:
        return torch.tanh(action)
    else:
        return action


def env_step(env, action, args):
    st0 = time.time()
    act = squash_action(action.detach(), args)
    st1 = time.time()
    next_obs, reward, done, infos = env.step(act)
    et1 = time.time()
    infos[0]['time/ES-helpers.env_step;env.step(act)'] = et1 - st1
    next_obs = tensor(next_obs, device=DeviceConfig.DEVICE)
    reward = tensor(reward, device=DeviceConfig.DEVICE)

    # TODO: Not sure if this is correct; modified from original call to get_belief (see history)
    belief = (torch.from_numpy(np.array(env.get_belief)).float().to(DeviceConfig.DEVICE) 
              if args.pass_belief_to_policy else None)
    task = torch.from_numpy(env.get_task()).float().to(DeviceConfig.DEVICE) 
    et0 = time.time()
    infos[0]['time/ES-helpers.env_step;REST'] = et0 - st0
    return [next_obs, belief, task], reward, done, infos


def select_action(args,
                  policy,
                  deterministic,
                  state=None,
                  belief=None,
                  task=None,
                  latent_sample=None, 
                  latent_mean=None, 
                  latent_logvar=None):
    """ Select action using the policy. """
    latent = get_latent_for_policy(args=args, latent_sample=latent_sample, 
                                   latent_mean=latent_mean,
                                   latent_logvar=latent_logvar)
    action = policy.act(state=state, latent=latent, belief=belief, task=task, 
                        deterministic=deterministic)
    if isinstance(action, list) or isinstance(action, tuple):
        value, action = action
    else:
        value = None
    action = action.to(DeviceConfig.DEVICE)

    # TODO DEBUG:
    if len(action.shape) == 3:  # For some reason we are getting an extra dim
                                # here now; I don't know what caused this
                                # change...
        action = action.squeeze(2)

    return value, action


def get_latent_for_policy(args, latent_sample=None, latent_mean=None, 
                          latent_logvar=None):

    if ((latent_sample is None) and (latent_mean is None)
            and (latent_logvar is None)):
        return None

    if args.add_nonlinearity_to_latent:
        latent_sample = F.relu(latent_sample)
        latent_mean = F.relu(latent_mean)
        latent_logvar = F.relu(latent_logvar)

    if args.sample_embeddings:
        latent = latent_sample
    else:
        latent = torch.cat((latent_mean, latent_logvar), dim=-1)

    if latent.shape[0] == 1:
        latent = latent.squeeze(0)

    return latent


def update_encoding(encoder, next_obs, action, reward, done, hidden_state):
    # reset hidden state of the recurrent net when we reset the task
    if done is not None:
        hidden_state = encoder.reset_hidden(hidden_state, done)

    with torch.no_grad():
        latent_sample, latent_mean, latent_logvar, hidden_state = encoder(
            actions=action.float(), 
            states=next_obs, 
            rewards=reward, 
            hidden_state=hidden_state, 
            return_prior=False)

    # TODO: move the sampling out of the encoder!
    return latent_sample, latent_mean, latent_logvar, hidden_state


def seed(seed, deterministic_execution=False):
    print('Seeding random, torch, numpy.')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    if deterministic_execution:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('Note that due to parallel processing results will be similar '
              'but not identical. '
              'Use only one process and set --deterministic-execution to True '
              'if you want identical results '
              '(only recommended for debugging).')


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def recompute_embeddings(
        policy_storage,
        encoder,
        sample,
        update_idx,
        detach_every
):
    # get the prior
    latent_sample = [policy_storage.latent_samples[0].detach().clone()]
    latent_mean = [policy_storage.latent_mean[0].detach().clone()]
    latent_logvar = [policy_storage.latent_logvar[0].detach().clone()]

    latent_sample[0].requires_grad = True
    latent_mean[0].requires_grad = True
    latent_logvar[0].requires_grad = True

    # loop through experience and update hidden state
    # (we need to loop because we sometimes need to reset the hidden state)
    h = policy_storage.hidden_states[0].detach()
    for i in range(policy_storage.actions.shape[0]):
        # reset hidden state of the GRU when we reset the task
        h = encoder.reset_hidden(h, policy_storage.done[i + 1])

        # Check if state is dictionary, to make sure we are slicing correctly
        if isinstance(policy_storage.next_state, dict):
            next_obs = {k: v[i:i + 1] for k, v in policy_storage.next_state.items()}
        else:
            next_obs = policy_storage.next_state[i:i + 1]

        ts, tm, tl, h = encoder(policy_storage.actions.float()[i:i + 1],
                                next_obs,
                                policy_storage.rewards_raw[i:i + 1],
                                h,
                                sample=sample,
                                return_prior=False,
                                detach_every=detach_every
                                )

        # print(i, reset_task.sum())
        # print(i, (policy_storage.latent_mean[i + 1] - tm).sum())
        # print(i, (policy_storage.latent_logvar[i + 1] - tl).sum())
        # print(i, (policy_storage.hidden_states[i + 1] - h).sum())

        latent_sample.append(ts)
        latent_mean.append(tm)
        latent_logvar.append(tl)

    if update_idx == 0:
        try:
            assert (torch.cat(policy_storage.latent_mean) 
                    - torch.cat(latent_mean)).sum() == 0
            assert (torch.cat(policy_storage.latent_logvar) 
                    - torch.cat(latent_logvar)).sum() == 0
        except AssertionError:
            warnings.warn('You are not recomputing the embeddings correctly!')
            import pdb
            pdb.set_trace()

    policy_storage.latent_samples = latent_sample
    policy_storage.latent_mean = latent_mean
    policy_storage.latent_logvar = latent_logvar


class FeatureExtractor(nn.Module):
    """ Used for extrating features for states/actions/rewards """

    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            ret = self.activation_function(self.fc(inputs))
            return ret
        else:
            return torch.zeros(0, ).to(DeviceConfig.DEVICE)
        

class FeatureExtractorConv(nn.Module):
    """ Used for extracting features from images """
    
    def __init__(self, image_shape, output_size, activation_function):
        super(FeatureExtractorConv, self).__init__()
        # Extract the number of channels from the shape
        height, width, input_channels = image_shape
        self.output_size = output_size
        self.activation_function = activation_function
        # Define a simple convolutional network
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1), # Output: 32x48x48
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # Output: 64x24x24
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # Output: 128x12x12
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # Output: 256x6x6
            nn.ReLU(),
        )
        self.fc_size = 256 * 6 * 6
        if self.output_size != 0:
            self.fc = nn.Linear(self.fc_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        # Recover time dimension if it exists
        dims_to_recover = 0
        # (bs, chan, h, w) <- (bs, h, w, chan)
        if len(inputs.shape) == 6:  # Assuming shape is [rollout, time, batch, height, width, channels]
            input_p = inputs.permute(0, 1, 2, 5, 3, 4)
            print('type(input_p): ', type(input_p))
            print('input_p.shape: ', input_p.shape)
            print('input_p.dtype: ', input_p.dtype)
            input_p = input_p.reshape(-1, *input_p.shape[3:])
            dims_to_recover = 3
        elif len(inputs.shape) == 5:  # Assuming shape is [time, batch, height, width, channels]
            input_p = inputs.permute(0, 1, 4, 2, 3)
            input_p = input_p.reshape(-1, *input_p.shape[2:]) # Flatten time and batch
            dims_to_recover = 2
        elif len(inputs.shape) == 4:  # Assuming shape is [batch, height, width, channels]
            input_p = inputs.permute(0, 3, 1, 2)
        elif len(inputs.shape) == 3:  # Assuming shape is [height, width, channels]
            input_p = inputs.permute(2, 0, 1)
        else:
            raise ValueError("Unexpected number of dimensions in the input tensor.")
        x = self.conv_layers(input_p)
        x = x.reshape(x.size(0), -1)  # Flatten (note, x.view doesn't work 
                                      # bc contig. mem. issues related to conv)
        if self.output_size != 0:
            ret = self.activation_function(self.fc(x))
            if dims_to_recover != 0:
                ret = ret.reshape(*[inputs.shape[i] for i in range(dims_to_recover)], -1)
            return ret
        else:
            return torch.zeros(0, ).to(DeviceConfig.DEVICE)


def sample_gaussian(mu, logvar, num=None):
    std = torch.exp(0.5 * logvar)
    if num is not None:
        std = std.repeat(num, 1)
        mu = mu.repeat(num, 1)
    eps = torch.randn_like(std)
    return mu + std * eps


def save_obj(obj, folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'rb') as f:
        return pickle.load(f)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # PyTorch version.
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).float().to(DeviceConfig.DEVICE)
        self.var = torch.ones(shape).float().to(DeviceConfig.DEVICE)
        self.count = epsilon

    def update(self, x):
        x = x.view((-1, x.shape[-1]))
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, 
                                       batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def bool_arg(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))


def int_arg(value):
    """ Accepts scientific notation. """
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid int value: '{value}'")


def get_task_dim(args):
    global make_vec_envs
    from qd_metarl.environments.parallel_envs import make_vec_envs
    env, plr_components = make_vec_envs(env_name=args.env_name, seed=args.seed, 
                        num_processes=args.num_processes,
                        gamma=args.policy_gamma, device=DeviceConfig.DEVICE,
                        trials_per_episode=args.trials_per_episode,
                        normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                        plr=args.use_plr,
                        tasks=None,
                        **self.args.vec_env_kwargs
                        )
    return env.task_dim


def get_num_tasks(args):
    global make_vec_envs
    from qd_metarl.environments.parallel_envs import make_vec_envs
    env, plr_components = make_vec_envs(env_name=args.env_name, seed=args.seed, 
                        num_processes=args.num_processes,
                        gamma=args.policy_gamma, device=DeviceConfig.DEVICE,
                        trials_per_episode=args.trials_per_episode,
                        normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                        plr=args.use_plr,
                        tasks=None,
                        **self.args.vec_env_kwargs
                        )
    try:
        num_tasks = env.num_tasks
    except AttributeError:
        num_tasks = None
    return num_tasks


def clip(value, low, high):
    """Imitates `{np,tf}.clip`.

    `torch.clamp` doesn't support tensor valued low/high so this provides the
    clip functionality.

    TODO(hartikainen): The broadcasting hasn't been extensively tested yet,
        but works for the regular cases where
        `value.shape == low.shape == high.shape` or 
        when `{low,high}.shape == ()`.
    """
    low, high = torch.tensor(low), torch.tensor(high)

    assert torch.all(low <= high), (low, high)

    clipped_value = torch.max(torch.min(value, high), low)
    return clipped_value


def shape(value):
    if isinstance(value, torch.Tensor):
        return value.shape
    if isinstance(value, np.ndarray):
        return value.shape
    if isinstance(value, list):
        return np.array(value).shape
    if isinstance(value, dict): 
        # Return shape of all values in dict
        return {k: shape(v) for k, v in value.items()}


MAX_ITERATIONS = 100000  # or whatever you anticipate as the upper limit

def generate_video_name(iter_idx):
    # Subtract the iteration index from the max value to get a descending prefix
    prefix = MAX_ITERATIONS - iter_idx
    # Format it with leading zeros to ensure consistent filename length
    # and append the original iteration index after a separator (e.g., "_")
    filename = f"{prefix:05d}_{iter_idx}_video.webm" 
    return filename


import numpy as np
from moviepy.editor import ImageSequenceClip
import os
import uuid
from io import BytesIO

def video_from_images(fps, images, n_processes=5):
    # Calculate the number of black and white frames needed
    black_frames_needed = int(0.5 * fps)
    white_frames_needed = int(1/3 * fps)
    
    # Diagnostic: Check the structure of the input images
    print(f"Total number of environments: {len(images)}")
    
    # Assuming all images are of the same size, we can get the shape from the first image
    height, width, _ = images[0][0][0].shape

    # Create black and white frames
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    white_frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    selected_images = []

    for i, env_images in enumerate(images[:n_processes]):
        # Diagnostic: Check the number of trials and frames in each environment
        print(f"Environment {i+1}: Total trials: {len(env_images)}, First trial frames: {len(env_images[0])}, Last trial frames: {len(env_images[-1])}")

        # Add white frames at the beginning of each environment/process
        if i > 0:  # Don't add for the very first process
            selected_images.extend([white_frame] * white_frames_needed)

        # First trial
        first_trial_images = env_images[0]
        selected_images.extend(first_trial_images)
        # Add black frames at the end of trial
        selected_images.extend([black_frame] * black_frames_needed)

        # Last trial
        last_trial_images = env_images[-1]
        selected_images.extend(last_trial_images)
        # Add black frames at the end of trial
        selected_images.extend([black_frame] * black_frames_needed)

    # Diagnostic: Check the number of selected frames for the video
    print(f"Total frames selected for video: {len(selected_images)}")

    # Convert images from BGR to RGB (if they are BGR)
    # Assuming the images are numpy arrays
    if selected_images[0].shape[2] == 3:  # Check for RGB or BGR images
        selected_images = [img[:, :, ::-1] for img in selected_images]

    # Use moviepy to create the clip
    clip = ImageSequenceClip(selected_images, fps=fps)

    # Use a temporary file to store the video
    temp_file = f"temp_video_{uuid.uuid4()}.webm"

    # Write the clip to a WebM file
    clip.write_videofile(temp_file, codec='libvpx-vp9')

    # Read the temporary file into a bytes buffer
    with open(temp_file, 'rb') as f:
        buffer = BytesIO(f.read())

    # Clean up by removing the temporary file
    os.remove(temp_file)

    return buffer



def stitch_images(img_list, n=5):
    """
    Stitch a list of images into a nxn grid and return as a numpy array.

    Parameters:
    - img_list (list): A list of np.array images.

    Returns:
    - numpy.ndarray: Numpy array representation of the stitched image.
    """
    
    # Check if img_list has n*n images
    if len(img_list) != n * n:
        # If not, recursively stitch the first (n-1)*(n-1) images
        return stitch_images(img_list[:(n-1)*(n-1)], n=n-1)
    
    img_height, img_width, _ = img_list[0].shape  # Added channel unpacking

    # Create an empty image with the size for nxn images
    stitched_img = Image.new('RGB', (img_width * n, img_height * n))

    # Iterate over each image and paste it into the correct position in the stitched image
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            # Ensure index is within bounds - for debugging
            if idx >= len(img_list):
                raise IndexError(f"Trying to access index {idx} in img_list of length {len(img_list)}")
            # Convert numpy array image to PIL image before pasting
            image_pil = Image.fromarray(img_list[idx])
            stitched_img.paste(image_pil, (j * img_width, i * img_height))

    # Convert the stitched image to a numpy array
    return np.array(stitched_img) 
