import numpy as np

from environments.mujoco.ant_multitask_base import MultitaskAntEnv


class AntDirEnv(MultitaskAntEnv):

    def __init__(self, task={}, n_tasks=2, max_episode_steps=200,
                 forward_backward=False, modify_init_state_dist=False, on_circle_init_state=False,  **kwargs):
        self.forward_backward = forward_backward
        self._max_episode_steps = max_episode_steps
        self.num_tasks = n_tasks

        super(AntDirEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)

        #ctrl_cost = .5 * np.square(action).sum()
        #contact_cost = 0.5 * 1e-3 * np.sum(
        #    np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        #survive_reward = 1.0
        reward = forward_reward #- ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        #notdone = np.isfinite(state).all() \
        #          and state[2] >= 0.2 and state[2] <= 1.0
        done = False #not notdone
        ob = self._get_obs()
        #print(ob, torso_xyz_after, torso_velocity/self.dt)
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            #reward_ctrl=-ctrl_cost,
            #reward_contact=-contact_cost,
            #reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

    def sample_tasks(self, num_tasks):
        if self.forward_backward:
            assert num_tasks == 2
            velocities = np.array([0., np.pi])
        else:
            velocities = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
        tasks = [{'goal': velocity} for velocity in velocities]
        return tasks

    def set_goal(self, goal):
        self._goal = goal

    def set_all_goals(self, goals):
        assert self.num_tasks == len(goals)
        self.tasks = [{'goal': velocity[0]} for velocity in goals]
        self.reset_task(0)
        #print(self.tasks)

    def get_task(self):
        return self._goal

