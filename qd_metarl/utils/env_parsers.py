from qd_metarl.config.racing import args_racing_oplr_varibad, args_racing_varibad, \
    args_racing_qd_varibad, args_racing_gen_varibad, \
    args_racing_plr_gen_varibad, \
    args_racing_oaqd_varibad, args_racing_accel_varibad, args_racing_rplr_varibad
from qd_metarl.config.alchemy import args_alchemy_oplr_varibad, args_alchemy_varibad, \
    args_alchemy_qd_varibad, args_alchemy_gen_varibad, \
    args_alchemy_plr_gen_varibad, \
    args_alchemy_accel_varibad, args_alchemy_rplr_varibad, \
    args_alchemy_oaqd_varibad
from qd_metarl.config.toygrid import args_toygrid_oplr_varibad, args_toygrid_varibad, \
    args_toygrid_qd_varibad, args_toygrid_hyperx, args_toygrid_rplr_varibad, \
    args_toygrid_accel_varibad, args_toygrid_gen_varibad


ENV_PARSERS = {
    'alchemy': args_alchemy_varibad.get_parser,
    'alchemy_qd': args_alchemy_qd_varibad.get_parser,
    'alchemy_gen': args_alchemy_gen_varibad.get_parser,
    'alchemy_plr': args_alchemy_oplr_varibad.get_parser,
    'alchemy_plr_gen': args_alchemy_plr_gen_varibad.get_parser,
    'alchemy_oaqd': args_alchemy_oaqd_varibad.get_parser,
    'alchemy_accel': args_alchemy_accel_varibad.get_parser,
    'alchemy_rplr': args_alchemy_rplr_varibad.get_parser,
    
    'racing': args_racing_varibad.get_parser,
    'racing_qd': args_racing_qd_varibad.get_parser,
    'racing_gen': args_racing_gen_varibad.get_parser,
    'racing_plr': args_racing_oplr_varibad.get_parser,
    'racing_plr_gen': args_racing_plr_gen_varibad.get_parser,
    'racing_oaqd': args_racing_oaqd_varibad.get_parser,
    'racing_accel': args_racing_accel_varibad.get_parser,
    'racing_rplr': args_racing_rplr_varibad.get_parser,
    
    'toygrid': args_toygrid_varibad.get_parser,
    'toygrid_plr': args_toygrid_oplr_varibad.get_parser,
    'toygrid_qd': args_toygrid_qd_varibad.get_parser,
    'toygrid_hyperx': args_toygrid_hyperx.get_parser,
    'toygrid_rplr': args_toygrid_rplr_varibad.get_parser,
    'toygrid_accel': args_toygrid_accel_varibad.get_parser,
    'toygrid_gen': args_toygrid_gen_varibad.get_parser,
}


def get_vec_env_kwargs(args):
    """ Set environment-specific arguments """
    # Set flag to determine observation space / other peculiarities 
    vec_env_kwargs = dict()
    if 'MazeEnv' in args.env_name:
        vec_env_kwargs['gt_type'] = args.gt_type
        vec_env_kwargs['size'] = args.size
    elif 'ToyGrid' in args.env_name:
        vec_env_kwargs['gt_type'] = args.gt_type
        vec_env_kwargs['size'] = args.size
    elif 'Alchemy' in args.env_name:
        vec_env_kwargs['gt_type'] = args.gt_type
        vec_env_kwargs['alchemy_mods'] = args.alchemy_mods
        vec_env_kwargs['max_steps'] = args.max_steps
        vec_env_kwargs['use_dynamic_items'] = args.alchemy_use_dynamic_items
        vec_env_kwargs['num_trials'] = args.trials_per_episode
        if args.alchemy_use_dynamic_items == False:
            # NOTE: We'd have to pass in use_dynamic_items everwhere to make this work, 
            # but instead, we should just include this option as part of the gt_type, since
            # this is already being passed in. Then, we can set dynamic_stones based on the
            # gt_type, and remove it as an API argument.
            raise NotImplementedError('Static stones are no longer fully supported')
    elif 'Racing' in args.env_name:
        vec_env_kwargs['fps'] = args.racing_fps
        vec_env_kwargs['obs_type'] = args.racing_obs_type
        vec_env_kwargs['max_steps'] = args.max_steps
        vec_env_kwargs['gt_type'] = args.gt_type
    elif 'XLand' in args.env_name:
        vec_env_kwargs['height'] = args.xland_grid_size
        vec_env_kwargs['width'] = args.xland_grid_size
        vec_env_kwargs['grid_type'] = args.xland_grid_type
        # NOTE: We want to pass this argument into gym.make so that this
        # gymnasium environment is compatible with the outdated gym API
        # we're using.
        # See https://gymnasium.farama.org/content/gym_compatibility/
        vec_env_kwargs['apply_api_compatibility'] = True
    return vec_env_kwargs