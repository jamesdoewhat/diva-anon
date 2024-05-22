
def get_plr_args_dict(args, plr_num_actors):
    """ Constructs a dictionary of arguments for convenient shorthand access. """
    return dict(   
        num_actors=plr_num_actors,
        strategy=args.plr_level_replay_strategy,
        replay_schedule=args.plr_level_replay_schedule,
        score_transform=args.plr_level_replay_score_transform,
        temperature=args.plr_level_replay_temperature_start,
        eps=args.plr_level_replay_eps,
        rho=args.plr_level_replay_rho,
        replay_prob=args.plr_replay_prob, 
        alpha=args.plr_level_replay_alpha,
        staleness_coef=args.plr_staleness_coef,
        staleness_transform=args.plr_staleness_transform,
        staleness_temperature=args.plr_staleness_temperature,
        seed_buffer_size=args.plr_seed_buffer_size)