import argparse
from qd_metarl.utils.env_utils import bool_arg, int_arg

# Common args for maze basic env

def get_common_rl2_arg_parser():
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    # --- GENERAL ---
    parser.add_argument('--num-frames', type=int_arg, default=30_000_000, help='number of frames to train')
    parser.add_argument('--trials-per-episode', type=int, default=2, help='number of MDP trials for adaptation')
    parser.add_argument('--test-env-name', default='ToyGrid-v0', help='environment to test on')
    
    # -- RL (things to change in VariBAD to get to RL2 setup) --
    parser.add_argument('--disable-decoder', type=bool_arg, default=True,help='train without decoder')
    parser.add_argument('--disable-kl-term', type=bool_arg, default=True,help='dont use the KL regularising loss term')
    parser.add_argument('--add-nonlinearity-to-latent', type=bool_arg, default=True,help='Use relu before feeding latent to policy')
    parser.add_argument('--rlloss-through-encoder', type=bool_arg, default=True,help='backprop rl loss through encoder')
    parser.add_argument('--condition-policy-on-state', type=bool_arg, default=False, help='after the encoder, concatenate env state and latent variable')
    # note: the latent_dim is just a layer in the policy (name comes from varibad code)
    parser.add_argument('--latent-dim', type=int, default=128, help='dimensionality of latent space')

    # --- ToyGrid ---
    parser.add_argument('--size', type=int, default=11,help='Size of the maze')

    # --- POLICY ---

    # what to pass to the policy (note this is after the encoder)
    parser.add_argument('--pass-state-to-policy', type=bool_arg, default=False, help='condition policy on state')
    parser.add_argument('--pass-latent-to-policy', type=bool_arg, default=True, help='condition policy on VAE latent')
    parser.add_argument('--pass-belief-to-policy', type=bool_arg, default=False, help='condition policy on ground-truth belief')
    parser.add_argument('--pass-task-to-policy', type=bool_arg, default=False, help='condition policy on ground-truth task description')

    # using separate encoders for the different inputs ("None" uses no encoder)
    parser.add_argument('--policy-state-embedding-dim', type=int, default=None)
    parser.add_argument('--policy-latent-embedding-dim', type=int, default=None)
    parser.add_argument('--policy-belief-embedding-dim', type=int, default=None)
    parser.add_argument('--policy-task-embedding-dim', type=int, default=None)

    # normalising (inputs/rewards/outputs)
    parser.add_argument('--norm-state-for-policy', type=bool_arg, default=False, help='normalise state input')
    parser.add_argument('--norm-latent-for-policy', type=bool_arg, default=False, help='normalise latent input')
    parser.add_argument('--norm-belief-for-policy', type=bool_arg, default=False, help='normalise belief input')
    parser.add_argument('--norm-task-for-policy', type=bool_arg, default=False, help='normalise task input')
    parser.add_argument('--norm-rew-for-policy', type=bool_arg, default=True, help='normalise rew for RL train')
    parser.add_argument('--norm-actions-pre-sampling', type=bool_arg, default=False, help='normalise policy output')
    parser.add_argument('--norm-actions-post-sampling', type=bool_arg, default=False, help='normalise policy output')

    # network
    parser.add_argument('--policy-layers', nargs='+', default=[128])
    parser.add_argument('--policy-activation-function', type=str, default='tanh', help='tanh/relu/leaky-relu')
    parser.add_argument('--policy-initialisation', type=str, default='normc', help='normc/orthogonal')
    parser.add_argument('--policy-anneal-lr', type=bool_arg, default=False, help='anneal LR over time')

    # RL algorithm
    parser.add_argument('--policy', type=str, default='ppo', help='choose: a2c, ppo')
    parser.add_argument('--policy-optimiser', type=str, default='adam', help='choose: rmsprop, adam')

    # PPO specific
    parser.add_argument('--ppo-num-epochs', type=int, default=2, help='number of epochs per PPO update')
    parser.add_argument('--ppo-num-minibatch', type=int, default=4, help='number of minibatches to split the data')
    parser.add_argument('--ppo-use-huberloss', type=bool_arg, default=True, help='use huberloss instead of MSE')
    parser.add_argument('--ppo-use-clipped-value-loss', type=bool_arg, default=True, help='clip value loss')
    parser.add_argument('--ppo-clip-param', type=float, default=0.05, help='clamp param')

    # other hyperparameters
    parser.add_argument('--lr-policy', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument('--lr-vae', type=float, default=7e-4)
    # since we use RL2, we have to match this LR (for the encoder) with the one for the policy
    parser.add_argument('--num-processes', type=int, default=12,help='how many training CPU processes / parallel environments to use (default: 16)')
    parser.add_argument('--policy-num-steps', type=int, default=120,help='number of env steps to do (per process) before updating')
    parser.add_argument('--policy-eps', type=float, default=1e-8, help='optimizer epsilon (1e-8 for ppo, 1e-5 for a2c)')
    parser.add_argument('--policy-init-std', type=float, default=1.0, help='only used for continuous actions')
    parser.add_argument('--policy-value-loss-coef', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument('--policy-entropy-coef', type=float, default=0.01, help='entropy term coefficient')
    parser.add_argument('--policy-gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--policy-use-gae', type=bool_arg, default=True,help='use generalized advantage estimation')
    parser.add_argument('--policy-tau', type=float, default=0.9, help='gae parameter')
    parser.add_argument('--use-proper-time-limits', type=bool_arg, default=False,help='treat timeout and death differently (important in mujoco)')
    parser.add_argument('--policy-max-grad-norm', type=float, default=0.5, help='max norm of gradients')
    parser.add_argument('--encoder-max-grad-norm', type=float, default=None, help='max norm of gradients')
    parser.add_argument('--decoder-max-grad-norm', type=float, default=None, help='max norm of gradients')

    # --- VAE TRAINING ---

    # general
    parser.add_argument('--size-vae-buffer', type=int, default=0, help='how many trajectories (!) to keep in VAE buffer')
    parser.add_argument('--precollect-len', type=int, default=0, help='how many frames to pre-collect before training begins (useful to fill VAE buffer)')
    parser.add_argument('--vae-buffer-add-thresh', type=float, default=1, help='probability of adding a new trajectory to buffer')
    parser.add_argument('--vae-batch-num-trajs', type=int, default=25, help='how many trajectories to use for VAE update')
    parser.add_argument('--tbptt-stepsize', type=int, default=None, help='stepsize for truncated backpropagation through time; None uses max (horizon of BAMDP)')
    parser.add_argument('--vae-subsample-elbos', type=int, default=None, help='for how many timesteps to compute the ELBO; None uses all')
    parser.add_argument('--vae-subsample-decodes', type=int, default=None, help='number of reconstruction terms to subsample; None uses all')
    parser.add_argument('--vae-avg-elbo-terms', type=bool_arg, default=False, help='Average ELBO terms (instead of sum)')
    parser.add_argument('--vae-avg-reconstruction-terms', type=bool_arg, default=False, help='Average reconstruction terms (instead of sum)')
    parser.add_argument('--num-vae-updates', type=int, default=3, help='how many VAE update steps to take per meta-iteration')
    parser.add_argument('--pretrain-len', type=int, default=0, help='for how many updates to pre-train the VAE')
    parser.add_argument('--kl-weight', type=float, default=0.01, help='weight for the KL term')

    parser.add_argument('--split-batches-by-task', type=bool_arg, default=False, help='split batches up by task (to save memory or if tasks are of different length)')
    parser.add_argument('--split-batches-by-elbo', type=bool_arg, default=False, help='split batches up by elbo term (to save memory of if ELBOs are of different length)')
    
    # - encoder
    parser.add_argument('--action-embedding-size', type=int, default=8)
    parser.add_argument('--state-embedding-size', type=int, default=16)
    parser.add_argument('--reward-embedding-size', type=int, default=16)
    parser.add_argument('--encoder-layers-before-gru', nargs='+', type=int, default=[])
    parser.add_argument('--encoder-gru-hidden-size', type=int, default=128,  help='dimensionality of RNN hidden state')
    parser.add_argument('--encoder-layers-after-gru', nargs='+', type=int, default=[])

    # - decoder: rewards
    parser.add_argument('--decode-reward', type=bool_arg, default=False,  help='use reward decoder')
    parser.add_argument('--rew-loss-coeff', type=float, default=1.0,  help='weight for state loss (vs reward loss)')
    parser.add_argument('--input-prev-state', type=bool_arg, default=True,  help='use prev state for rew pred')
    parser.add_argument('--input-action', type=bool_arg, default=True,  help='use prev action for rew pred')
    parser.add_argument('--reward-decoder-layers', nargs='+', type=int, default=[32, 32])
    parser.add_argument('--multihead-for-reward', type=bool_arg, default=False, help='one head per reward pred (i.e. per state)')
    parser.add_argument('--rew-pred-type', type=str, default='bernoulli', help='choose: '
                             'bernoulli (predict p(r=1|s))'
                             'categorical (predict p(r=1|s) but use softmax instead of sigmoid)'
                             'deterministic (treat as regression problem)')

    # - decoder: state transitions
    parser.add_argument('--decode-state', type=bool_arg, default=False,  help='use state decoder')
    parser.add_argument('--state-loss-coeff', type=float, default=1.0,  help='weight for state loss')
    parser.add_argument('--state-decoder-layers', nargs='+', type=int, default=[32, 32])
    parser.add_argument('--state-pred-type', type=str, default='deterministic',  help='choose: deterministic, gaussian')

    # - decoder: ground-truth task ("varibad oracle", after Humplik et al. 2019)
    parser.add_argument('--decode-task', type=bool_arg, default=False,  help='use task decoder')
    parser.add_argument('--task-loss-coeff', type=float, default=1.0,  help='weight for task loss')
    parser.add_argument('--task-decoder-layers', nargs='+', type=int, default=[32, 32])
    parser.add_argument('--task-pred-type', type=str, default='task_id',  help='choose: task_id, task_description')

    # --- ABLATIONS ---

    # combining vae and RL loss
    parser.add_argument('--vae-loss-coeff', type=float, default=1.0, help='weight for VAE loss (vs RL loss)')

    # for the policy training
    parser.add_argument('--sample-embeddings', type=bool_arg, default=False, help='sample embedding for policy, instead of full belief')

    # for other things
    parser.add_argument('--disable-metalearner', type=bool_arg, default=False, help='Train feedforward policy')
    parser.add_argument('--single-task-mode', type=bool_arg, default=False, help='train policy on one (randomly chosen) environment only')

    # --- OTHERS ---

    # logging, saving, evaluation
    parser.add_argument('--log-interval', type=int, default=100,  help='log interval, one log per n updates')
    parser.add_argument('--save-interval', type=int, default=1000,  help='save interval, one save per n updates')
    parser.add_argument('--save-intermediate-models', type=bool_arg, default=False,  help='save all models')
    parser.add_argument('--eval-interval', type=int, default=100,  help='eval interval, one eval per n updates')
    parser.add_argument('--eval-save-video', type=bool_arg, default=True, help='save video of rollouts during each evaluation')
    parser.add_argument('--video-fps', type=int, default=10, help='fps for video saving')
    parser.add_argument('--vis-interval', type=int, default=500,  help='visualisation interval, one eval per n updates')
    parser.add_argument('--results-log-dir', default=None,  help='directory to save results (None uses ./logs)')
    
    return parser