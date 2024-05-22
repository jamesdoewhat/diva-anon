import multiprocessing as mp
import os
import warnings
import traceback
import time

import numpy as np
from qd_metarl.environments.env_utils.vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars


def worker(remote, parent_remote, env_fn_wrappers):
    os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    def step_env(env, action):
        ob, reward, done, info = env.step(action)
        if done:
            ob = env.reset()
        return ob, reward, done, info
    
    parent_remote.close()  # Close unused end of pipe
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'reset':
                if data is not None:
                    # Calls VariBAD wrapper in wrappers.py, which resets task
                    # first and then environment
                    remote.send([env.reset(task=d) for env, d in zip(envs, data)])
                else:
                    remote.send([env.reset() for env in envs])
            elif cmd == 'reset_mdp':
                remote.send([env.reset_mdp() for env in envs])
            elif cmd == 'render':
                remote.send([env.render(mode='rgb_array') for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send([(env.observation_space, env.action_space) for env in envs])
            elif cmd == 'get_task':
                remote.send([env.get_task() for env in envs])
            elif cmd == 'get_belief':
                remote.send([env.get_belief() for env in envs])
            elif cmd == 'reset_task':
                [env.unwrapped.reset_task(d) for env, d in zip(envs, data)]
            elif cmd == 'get_spaces_spec':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space, envs[0].spec)))
            elif cmd in ['task_dim', 'belief_dim']:
                remote.send(getattr(envs[0], cmd))
            elif cmd in ['num_states', '_max_trial_steps', 'bit_map_size', 
                         'genotype_size', 'qd_bounds', 'genotype_bounds',
                         'size', 'bit_map_shape', 'gt_type', 'compute_measures',
                         'get_measures_info', 'process_genotype', 
                         'is_valid_genotype', 'compute_measures_static']:
                # For these, we assume it is the same for all tasks in distribution
                remote.send(getattr(envs[0].unwrapped, cmd))
            else:
                # try to get the attribute directly
                remote.send([getattr(env.unwrapped, cmd) for env in envs])
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None, context='spawn', in_series=1):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a single process
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """

        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        self.env_fns = env_fns
        self.context = context
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()

        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv().x
        self.viewer = None
        VecEnv.__init__(self, nenvs, observation_space, action_space)
        print('SubprocVecEnv: created with {} envs in {} processes'.format(nenvs, self.nremotes))

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.nremotes)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = _flatten_list(results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos
    
    def restart_remotes(self):
        
        print(f'\nRestarting {self.nremotes} remotes...')
        print('Closing current remotes...')
        for i, remote in enumerate(self.remotes):
            try:
                remote.send(('close', None))
                print(f'Successfully closed remote {i}.')
            except Exception as e:
                print(f'Exception in closing remote {i}:', e)
        
        print('Sleeping for 60 seconds before restarting...')
        global time; import time
        time.sleep(60)

        print('Restarting...')
        ctx = mp.get_context(self.context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, self.env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        
        print('Closing work remotes...')
        for remote in self.work_remotes:
            remote.close()

        # Print PIDs and check if processes are alive
        print('Checking if processes are up and running...')
        for i, p in enumerate(self.ps):
            print(f'Process {i} PID: {p.pid}, is_alive: {p.is_alive()}')
            if not p.is_alive():
                print(f'Process {i} with PID {p.pid} is not running.')
                # Perform additional analysis here, e.g., check logs, restart process, etc.

    def reset(self, task=None, index=None):

        if index is not None: 
            raise NotImplementedError("SubprocVecEnv does not support resetting a single environment")

        self._assert_not_closed()
        # Did VariBAD have a bug here? Why are we calling each remote 
        # separately, but passing in all tasks? Shouldn't we pass one task to
        # each remote? See notes on page 33 of notebook.
        if task is not None: 
            assert len(task) == len(self.remotes)

        # while True:
        for i, remote in enumerate(self.remotes):
            remote.send(('reset', [task[i]]))
        # try:
        obs = [remote.recv() for remote in self.remotes]
            #     break
            # except EOFError:
            #     print('EOFError in reset!')
            #     print(f'Tasks: {task}')
            #     print('Traceback:', traceback.format_exc())
            #     self.restart_remotes()
            #     print('Sleeping for 5 seconds before retrying...')
            #     time.sleep(5)
            # print('Retrying!')

        obs = _flatten_list(obs)
        obs = _flatten_obs(obs)
        return obs

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        imgs = _flatten_list(imgs)
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def get_env_attr(self, attr):
        self.remotes[0].send((attr, None))
        return self.remotes[0].recv()
    
    def get_env_attrs(self, attr):
        for remote in self.remotes:
            remote.send((attr, None))
        vals = [remote.recv() for remote in self.remotes]
        return vals

    def get_task(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_task', None))
        tasks = [remote.recv() for remote in self.remotes]
        return np.stack(tasks)

    def __del__(self):
        if not self.closed:
            self.close()

def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)

def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]