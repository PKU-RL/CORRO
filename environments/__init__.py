from gym.envs.registration import register

# # ----------- Mujoco ----------- # #
register(
    'HalfCheetahDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_dir:HalfCheetahDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahVel-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel:HalfCheetahVelEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntSemiCircle-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_semicircle:AntSemiCircleEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntSemiCircleSparse-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_semicircle:SparseAntSemiCircleEnv',
            'max_episode_steps': 200, 'goal_radius': 0.2},
    max_episode_steps=200
)

# # ----------- GridWorld ----------- # #

register(
    'GridNavi-v2',
    entry_point='environments.toy_navigation.gridworld:GridNavi',
    kwargs={'num_cells': 5, 'num_steps': 15, 'n_tasks': 2,
            'is_sparse': False, 'return_belief_rewards': True,
            'seed': None},
    # kwargs={'num_cells': 5, 'num_steps': 30, 'n_tasks': 2},
)

register(
    'GridBlock-v2',
    entry_point='environments.toy_navigation.gridworld_block:GridBlockNavi',
    kwargs={'num_cells': 5, 'num_steps': 25, 'n_tasks': 2,
            'is_sparse': False, 'return_belief_rewards': False,
            'seed': None},
)

# # ----------- Point Robot ----------- # #
register(
    'PointRobot-v0',
    entry_point='environments.toy_navigation.point_robot:PointEnv',
    kwargs={'max_episode_steps': 60, 'n_tasks': 2},
)


register(
    'PointRobotSparse-v0',
    entry_point='environments.toy_navigation.point_robot:SparsePointEnv',
    kwargs={'max_episode_steps': 60, 'n_tasks': 2, 'goal_radius': 0.2},
)

# point robot with fixed initial state
register(
    'PointRobot-v1',
    entry_point='environments.toy_navigation.point_robot:PointEnv',
    kwargs={'max_episode_steps': 10, 'n_tasks': 2, 'modify_init_state_dist': False},
)


# rand param envs
register(
    'HopperRandParams-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.rand_param_envs.hopper_rand_params:HopperRandParamsEnv',
            }
)

register(
    'WalkerRandParams-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.rand_param_envs.walker2d_rand_params:Walker2DRandParamsEnv',
            }
)