import os

import gym
import imageio
import numpy as np
from PIL import Image

from pddlgym.parser import parse_plan_step
from pddlgym.planning import run_planner
from pddlgym.utils import run_random_agent_demo, run_planning_demo

def demo_random(env_name, render=True, problem_index=0, verbose=True):
    env = gym.make("PDDLEnv{}-v0".format(env_name.capitalize()))
    if not render: env._render = None
    env.fix_problem_index(problem_index)

    run_random_agent_demo(env=env, planner_name=env_name, outdir='results', verbose=verbose, seed=0)

def run_random_agent_demo(env, planner_name,outdir='/tmp', max_num_steps=10, fps=3,
                          verbose=False, seed=None):
    outdir = "{}/{}".format(outdir, planner_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if env._render:
        video_path = os.path.join(outdir, 'random_{}_demo.gif'.format(env.spec.id))
        env = VideoWrapper(env, video_path, fps=fps)

    if seed is not None:
        env.seed(seed)

    obs, _ = env.reset()

    if seed is not None:
        env.action_space.seed(seed)

    for t in range(max_num_steps):
        if verbose:
            print("Obs:", obs)

        action = env.action_space.sample()
        if verbose:
            print("Act:", action)

        obs, reward, done, _ = env.step(action)
        env.render()
        if verbose:
            print("Rew:", reward)

        if done:
            break

    if verbose:
        print("Final obs:", obs)
        print()
    env.close()

class VideoWrapper(gym.Wrapper):
    def __init__(self, env, out_path, fps=30, size=None):
        super().__init__(env)
        self.out_path_prefix = '.'.join(out_path.split('.')[:-1])
        self.out_path_suffix = out_path.split('.')[-1]
        self.fps = fps
        self.size = size
        self.reset_count = 0

    def reset(self):
        obs = super().reset()

        # Handle problem-dependent action spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.out_path = self.out_path_prefix + str(self.reset_count) + \
                        '.' + self.out_path_suffix
        self.reset_count += 1

        self.images = []
        img = super().render()
        img = self.process_image(img)
        self.images.append(img)

        return obs

    def step(self, action):
        obs, reward, done, debug_info = super().step(action)

        img = super().render()
        img = self.process_image(img)
        self.images.append(img)

        return obs, reward, done, debug_info

    def close(self):
        imageio.mimsave(self.out_path, self.images, fps=self.fps)
        print("Wrote out video to {}.".format(self.out_path))
        return super().close()

    def process_image(self, img):
        if self.size is None:
            return img
        return np.array(Image.fromarray(img).resize(self.size), dtype=img.dtype)


if __name__ == '__main__':
    def run_all(render=True, verbose=True):
        # demo_random("sokoban", render=render, verbose=verbose)
        demo_random("rearrangement", render=render, problem_index=6, verbose=verbose)
        # demo_random("minecraft", render=render, verbose=verbose)
        # demo_random("blocks_operator_actions", render=render, verbose=verbose)

    run_all()
