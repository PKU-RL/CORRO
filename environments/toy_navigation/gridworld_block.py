# Haoqi Yuan, 21/04/06

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from gym import spaces
from gym.utils import seeding
from torchkit import pytorch_utils as ptu
from matplotlib.patches import Rectangle


''' gridworld with a 3*1 block at random position
agent should adapt to both different goals and different blocks 
task representation: (xg,yg,xb,yb,dir), xg,yg is goal, xb,yb is block center, dir=0,1 is block direction'''
class GridBlockNavi(gym.Env):
    def __init__(self,
                 num_cells=5,
                 num_steps=25,
                 n_tasks=2,
                 modify_init_state_dist=False,
                 is_sparse=False,
                 return_belief_rewards=False,  # output R+ instead of R
                 seed=None,
                 **kwargs,
                 ):
        super(GridBlockNavi, self).__init__()

        if seed is not None:
            self.seed(seed)

        self.num_cells = num_cells
        self.num_states = num_cells ** 2
        self.grid_size = (num_cells, num_cells)
        self.block_size = 3 # a 3*1 block

        self.is_sparse = is_sparse
        self.return_belief_rewards = return_belief_rewards
        self.modify_init_state_dist = modify_init_state_dist

        self._max_episode_steps = num_steps
        self.step_count = 0

        # observation: agent(x,y)
        self.observation_space = spaces.Box(low=0, high=self.num_cells - 1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)  # noop, up, right, down, left

        # possible starting states - when not modify_init_state_dist
        self.starting_states = [(0.0, 0.0)]  # , (self.num_cells-1, 0)]#,
        # (0, self.num_cells-1), (self.num_cells-1, self.num_cells-1)]

        self.states = [(x, y) for y in np.arange(0, num_cells) for x in np.arange(0, num_cells)]

        # wall's center should be in inner 3*3 world, then assign wall's direction, 
        # then assign goal position at free positions
        self.possible_goals = []
        for xw in range (1, num_cells-1):
            for yw in range(1, num_cells-1):
                for dr in range(0, 2):
                    for g in self.states:
                        # goal should not in the block
                        # goal should not be too close to initial state
                        if not (self._goal_in_block(g, (xw,yw), dr) or (g[0]<=1 and g[1]<=1)):
                            self.possible_goals.append((g[0], g[1], xw, yw, dr))

        '''
        # remove starting states and some possible goals
        self.possible_goals = self.states.copy()
        for s in self.starting_states:
            self.possible_goals.remove(s)
        self.possible_goals.remove((0, 1))
        self.possible_goals.remove((1, 1))
        self.possible_goals.remove((1, 0))
        '''
        #print(self.possible_goals)
        #print(len(self.possible_goals))

        self.num_tasks = min(n_tasks, len(self.possible_goals))

        self.goals = random.sample(self.possible_goals, self.num_tasks)
        self.reset_task(0)
        #print(self.goals)

        if self.return_belief_rewards:
            #print("ret belief")
            raise NotImplementedError
            self._belief_state = self._reset_belief()

    # is the goal in the block?
    def _goal_in_block(self, g, w, dr):
        if dr == 0: # xw-1,xw,xw+1
            return g[1] == w[1] and g[0] >= w[0]-1 and g[0] <= w[0]+1
        else: # yw-1,yw,yw+1
            return g[0] == w[0] and g[1] >= w[1]-1 and g[1] <= w[1]+1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def get_all_task_idx(self):
        return range(len(self.goals))

    def get_task(self):
        return self._goal #(xg,yg,xb,yb,dir)

    def set_goal(self, goal):
        self._goal = np.asarray(goal)

    def set_state(self, state):
        self._state = np.asarray(state)

    # set self.goals to the given goals set
    # used in fully offline meta-rl eval stage
    def set_all_goals(self, goals):
        assert self.num_tasks == len(goals)
        self.goals = [np.asarray(i, dtype=np.int) for i in goals]
        self.reset_task(0)


    def reset_task(self, idx=None):
        ' reset goal and state '
        if idx is not None:
            self._goal = np.array(self.goals[idx])
        self.reset()

    def _reset_belief(self):
        return None
        # Not implemented!

        self._belief_state = np.zeros((self.num_cells ** 2))
        for pg in self.possible_goals:
            idx = self.task_to_id(np.array(pg))
            self._belief_state[idx] = 1.0 / len(self.possible_goals)
        return self._belief_state

    def reset_model(self):
        # reset the environment state
        if self.modify_init_state_dist:
            raise NotImplementedError # should not modify init state distrib
            self._state = np.array(random.choice(self.states))    # For data collection
            while (self._state == self._goal).all():    # do not start in goal
                self._state = np.array(random.choice(self.states))
        else:
            self._state = np.array(random.choice(self.starting_states))
        self._belief_state = self._reset_belief()
        return self.get_obs()

    # obs is agent's state (x,y)
    def get_obs(self):
        return np.copy(self._state)

    def update_belief(self, state):
        pass
        # Not implemented!

        if self.is_goal_state():
            self._belief_state *= 0
            self._belief_state[self.task_to_id(self._goal)] = 1
        else:
            self._belief_state[self.task_to_id(state)] = 0
            self._belief_state = np.ceil(self._belief_state)
            self._belief_state /= sum(self._belief_state)

    def reset(self):
        self.step_count = 0
        return self.reset_model()

    def reward(self, state, action=None):
        if state[0] == self._goal[0] and state[1] == self._goal[1]:
            return 1.
        else:
            return 0. if self.is_sparse else -0.1

    def state_transition(self, action):
        """
        Moving the agent between states
        """
        old_state = np.array(self._state)

        if action == 1:  # up
            self._state[1] = min([self._state[1] + 1, self.num_cells - 1])
        elif action == 2:  # right
            self._state[0] = min([self._state[0] + 1, self.num_cells - 1])
        elif action == 3:  # down
            self._state[1] = max([self._state[1] - 1, 0])
        elif action == 4:  # left
            self._state[0] = max([self._state[0] - 1, 0])

        # block
        if self._goal_in_block(self._state, (self._goal[2], self._goal[3]), self._goal[4]):
            self._state = old_state

    def step(self, action):
        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action[0]
        assert self.action_space.contains(action)

        info = {'task': self.get_task()}

        done = False

        # perform state transition
        self.state_transition(action)

        # check if maximum step limit is reached
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            done = True

        # compute belief rewards reward
        if self.return_belief_rewards:
            raise NotImplementedError

            self.update_belief(self._state)
            belief_reward = self._compute_belief_reward()
            info.update({'belief_reward': belief_reward})

        # compute reward
        reward = self.reward(self._state)

        return self.get_obs(), reward, done, info

    def _compute_belief_reward(self):
        raise NotImplementedError

        num_possible_goal_belief = np.sum(self._belief_state != 0)  # num. goals for which belief isn't 0
        non_goal_rew = 0. if self.is_sparse else -0.1
        belief_reward = (1. + non_goal_rew * (num_possible_goal_belief - 1)) / num_possible_goal_belief
        return belief_reward

    def is_goal_state(self):
        if self._state[0] == self._goal[0] and self._state[1] == self._goal[1]:
            return True
        else:
            return False

    '''
    def task_to_id(self, goals):
        mat = torch.arange(0, self.num_cells ** 2).long().reshape((self.num_cells, self.num_cells)).transpose(1, 0)
        if isinstance(goals, list) or isinstance(goals, tuple):
            goals = np.array(goals)
        if isinstance(goals, np.ndarray):
            goals = torch.from_numpy(goals)
        goals = goals.long()

        if goals.dim() == 1:
            goals = goals.unsqueeze(0)

        goal_shape = goals.shape
        if len(goal_shape) > 2:
            goals = goals.reshape(-1, goals.shape[-1])

        classes = mat[goals[:, 0], goals[:, 1]]
        classes = classes.reshape(goal_shape[:-1])

        return classes

    
    def id_to_task(self, classes):
        mat = torch.arange(0, self.num_cells ** 2).long().reshape((self.num_cells, self.num_cells)).numpy().T
        goals = np.zeros((len(classes), 2))
        classes = classes.numpy()
        for i in range(len(classes)):
            pos = np.where(classes[i] == mat)
            goals[i, 0] = float(pos[0][0])
            goals[i, 1] = float(pos[1][0])
        goals = torch.from_numpy(goals).to(ptu.device).float()
        return goals


    def goal_to_onehot_id(self, pos):
        cl = self.task_to_id(pos)
        if cl.dim() == 1:
            cl = cl.view(-1, 1)
        nb_digits = self.num_cells ** 2
        # One hot encoding buffer that you create out of the loop and just keep reusing
        y_onehot = torch.FloatTensor(pos.shape[0], nb_digits).to(ptu.device)
        # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, cl, 1)
        return y_onehot

    def onehot_id_to_goal(self, pos):
        if isinstance(pos, list):
            pos = [self.id_to_task(p.argmax(dim=1)) for p in pos]
        else:
            pos = self.id_to_task(pos.argmax(dim=1))
        return pos
    '''

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def plot_env(self):
        # draw grid
        for i in range(self.num_cells):
            for j in range(self.num_cells):
                pos_i = i
                pos_j = j
                rec = Rectangle((pos_i, pos_j), 1, 1, facecolor='none', alpha=0.5,
                                edgecolor='k')
                plt.gca().add_patch(rec)
        # draw goal
        goal = np.array(self._goal[0:2]) + 0.5
        plt.plot(goal[0], goal[1], 'kx')
        # draw agent
        state = np.array(self._state) + 0.5
        plt.plot(state[0], state[1], 'o')
        # draw block
        block = np.array(self._goal[2:4]) + 0.5
        plt.plot(block[0], block[1], '*')
        if self._goal[4] == 0:
            plt.plot(block[0]+1, block[1], '*')
            plt.plot(block[0]-1, block[1], '*')
        else:
            plt.plot(block[0], block[1]+1, '*')
            plt.plot(block[0], block[1]-1, '*')

    
    def plot_behavior(self, observations, plot_env=True, **kwargs):
        if plot_env:
            self.plot_env()
        # shift obs and goal by half a stepsize
        if isinstance(observations, tuple) or isinstance(observations, list):
            observations = torch.cat(observations)
        observations = observations + 0.5

        # visualise behaviour, current position, goal
        plt.plot(observations[:, 0], observations[:, 1], **kwargs)
        plt.plot(observations[-1, 0], observations[-1, 1], **kwargs)
        # make it look nice
        plt.xticks([])
        plt.yticks([])
        plt.xlim([0, self.num_cells])
        plt.ylim([0, self.num_cells])
        plt.axis('equal')
    

    # user interaction to debug the env
    def play_debug(self):
        self.reset()
        print('goal {} state {}'.format(self._goal, self._state))
        self.plot_env()
        plt.show()

        while True:
            print('input action: 0~4')
            act = int(input())
            obs, rew, done, _ = self.step(act)
            print(obs, rew, done)
            self.plot_env()
            plt.show()
            if done:
                break

    # check if (s,a,r,s') contain the transition information
    def is_sample_contain_transition(self, s, a, r, s_):
        if isinstance(a, np.ndarray) and a.ndim == 1:
            a = a[0]
        assert self.action_space.contains(a)

        s_tmp = np.array(s)
        if a == 1:  # up
            s_tmp[1] = min([s_tmp[1] + 1, self.num_cells - 1])
        elif a == 2:  # right
            s_tmp[0] = min([s_tmp[0] + 1, self.num_cells - 1])
        elif a == 3:  # down
            s_tmp[1] = max([s_tmp[1] - 1, 0])
        elif a == 4:  # left
            s_tmp[0] = max([s_tmp[0] - 1, 0])

        if self._goal_in_block(s_tmp, (self._goal[2], self._goal[3]), self._goal[4]):
            assert s[0] == s_[0] and s[1] == s_[1]
            return True
        else:
            assert s_tmp[0] == s_[0] and s_tmp[1] == s_[1]
            return False

    # check if (s,a,r,s') contain the reward func information
    def is_sample_contain_reward(self, s, a, r, s_):
        return r > 0

    def is_sample_contain_task(self, s, a, r, s_):
        return self.is_sample_contain_reward(s, a, r, s_) or self.is_sample_contain_transition(s, a, r, s_)



