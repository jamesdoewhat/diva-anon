import warnings
from gym.envs.registration import register
from .box2d.car_racing_f1 import RACETRACKS
from .box2d.car_racing_bezier import CarRacingBezier

warnings.filterwarnings('ignore', category=UserWarning, module='gym.envs.registration')



# DeepMind Alchemy
# -----------------------------------------------------------------------------

register(
    id='AlchemyRandom-v0',
    entry_point='environments.alchemy.alchemy_qd:ExtendedSymbolicAlchemy',
    kwargs={'env_type': 'random',
	        'distribution_type': 'SB'}
)

# QD

register(
    id='AlchemyRandomQD-v0',
    entry_point='environments.alchemy.alchemy_qd:ExtendedSymbolicAlchemy',
    kwargs={'env_type': 'random',
            'distribution_type': 'QD'}
)

# CarRacing (from DCD)
# -----------------------------------------------------------------------------

register(
    'CarRacing-Bezier-v0',
    entry_point='environments.box2d.car_racing_bezier:CarRacingBezier',
    kwargs={
	    'distribution_type': 'SB',
    },
)

register(
    'CarRacing-F1-v0',
    entry_point='environments.box2d.car_racing_bezier:CarRacingBezier',
    kwargs={
	    'distribution_type': 'SB',
        'use_f1_tracks': True,
    },
)

# for name, track in RACETRACKS.items():
# 	class_name = f"CarRacingF1-{track.name}"
# 	register(
# 		f'CarRacingF1-{track.name}-v0', 
# 		entry_point='environments.box2d.car_racing_f1' + f':{class_name}',
# 	    max_episode_steps=track.max_episode_steps,
# 	    reward_threshold=900
# 	)
	
# QD

register(
    'CarRacing-BezierQD-v0',
    entry_point='environments.box2d.car_racing_bezier:CarRacingBezier',
    kwargs={
	    'distribution_type': 'QD',
    },
)

# MazeEnv (from DSAGE)
# -----------------------------------------------------------------------------

register(
    'MazeEnvBasic-v0',
    entry_point='environments.maze.envs.maze:MazeEnv',
    kwargs={
            'bit_map': None,
            'max_steps': 30,
            'start_pos': None,
            'goal_pos': None,
            'distribution_type': 'SB',
	        'gt_type': 'BM+SGL',
            'start_sampler': 'uniform',
            'goal_sampler': 'uniform',
			'start_sampler_region': 'top-left',
			'goal_sampler_region': 'bottom-right',
            },
    # max_episode_steps=30,
)

# QD

register(
    'MazeEnvBasicQD-v0',
    entry_point='environments.maze.envs.maze:MazeEnv',
    kwargs={
		    'bit_map': None,
            'max_steps': 30,
            'start_pos': None,
            'goal_pos': None,
            'distribution_type': 'QD',
            'gt_type': 'BM+SGL',
            },
    # max_episode_steps=30,
)

# -----------------------------------------------------------------------------
# ToyGrid envs
# -----------------------------------------------------------------------------

register(
    'ToyGrid-v0',
    entry_point='environments.toygrid.toygrid:ToyGrid',
	kwargs={
		'max_steps': 30,
		'distribution_type': 'SB',
		'goal_sampler': 'uniform',
		'goal_sampler_region': 'left',
    },
)

# QD

register(
    'ToyGridQD-v0',
    entry_point='environments.toygrid.toygrid:ToyGrid',
	kwargs={
		'max_steps': 30,
		'distribution_type': 'QD',
    },
)

# -----------------------------------------------------------------------------
# Keys environment
# -----------------------------------------------------------------------------

register(
    'Keys-v0',
    entry_point='environments.navigation.keys:Keys',
    kwargs={'max_steps': 100,
			'grid_size': 5,},
    # max_episode_steps=100,
)


# -----------------------------------------------------------------------------
# VariBAD envs:
# -----------------------------------------------------------------------------


# Mujoco
# -----------------------------------------------------------------------------

# - randomised reward functions

# register(
#     'AntDir-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDirEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )

# register(
#     'AntDir2D-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDir2DEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200,
# )

# register(
#     'AntGoal-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.ant_goal:AntGoalEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )

# register(
#     'HalfCheetahDir-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.half_cheetah_dir:HalfCheetahDirEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )

# register(
#     'HalfCheetahVel-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel:HalfCheetahVelEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )

# register(
#     'HumanoidDir-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.humanoid_dir:HumanoidDirEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )

# # - randomised dynamics

# register(
#     id='Walker2DRandParams-v0',
#     entry_point='environments.mujoco.rand_param_envs.walker2d_rand_params:Walker2DRandParamsEnv',
#     max_episode_steps=200
# )

# register(
#     id='HopperRandParams-v0',
#     entry_point='environments.mujoco.rand_param_envs.hopper_rand_params:HopperRandParamsEnv',
#     max_episode_steps=200
# )


# 2D Navigation
# -----------------------------------------------------------------------------

# register(
#     'PointEnv-v0',
#     entry_point='environments.navigation.point_robot:PointEnv',
#     kwargs={'goal_radius': 0.2,
#             'max_episode_steps': 100,
#             'goal_sampler': 'semi-circle'
#             },
#     max_episode_steps=100,
# )

# register(
#     'SparsePointEnv-v0',
#     entry_point='environments.navigation.point_robot:SparsePointEnv',
#     kwargs={'goal_radius': 0.2,
#             'max_episode_steps': 100,
#             'goal_sampler': 'semi-circle'
#             },
#     max_episode_steps=100,
# )


# GridWorld
# -----------------------------------------------------------------------------

# register(
#     'GridNavi-v0',
#     entry_point='environments.navigation.gridworld:GridNavi',
#     kwargs={'num_cells': 5, 'num_steps': 15},
# )



# -----------------------------------------------------------------------------
# HyperX envs:
# -----------------------------------------------------------------------------

# register(
#     'SparsePointEnv-v0',
#     entry_point='environments.navigation.point_robot:SparsePointEnv',
#     kwargs={'goal_radius': 0.2,
#             'max_episode_steps': 100},
#     max_episode_steps=100,
# )

# # Multi-Stage GridWorld Rooms
# register(
#     'RoomNavi-v0',
#     entry_point='environments.navigation.rooms:RoomNavi',
#     kwargs={'num_cells': 3, 'corridor_len': 3, 'num_steps': 50},
# )

# # Mountain Treasure
# register(
#     'TreasureHunt-v0',
#     entry_point='environments.navigation.treasure_hunt:TreasureHunt',
#     kwargs={'max_episode_steps': 100,
#             'mountain_height': 1,
#             'treasure_reward': 10,
#             'timestep_penalty': -5,
#             },
# )