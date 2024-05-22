# Adapted from https://github.com/BKHMSI/symbolic_alchemy_visualizer/blob/main/visualize.py
import os
import yaml
import argparse
import numpy as np

from typing import Dict

import torch
from torch.nn import functional as F
from vpython import rate
from vpython.vpython import wtext
from pyvirtualdisplay import Display
from PIL import ImageGrab

from dm_alchemy import symbolic_alchemy
from dm_alchemy.encode import chemistries_proto_conversion
from dm_alchemy.types import utils

from qd_metarl.environments.alchemy.vis_utils import *


class AlchemyVisualizer:
    def __init__(self) -> None:
        self.dt = 1/50  # We asume 50 frames a second

        # Start a virtual display using the current process PID as the display number
        self.display = Display(visible=0, size=(800, 600))
        self.display.start()
        print("VIRTUAL DISPLAY STARTED")
        print('display num:', self.display.display, 'os.environ["DISPLAY"]:', os.environ["DISPLAY"])
        print('display running:', self.display.is_alive())
        
    def setup(self, 
              game_state,
              state: int
    ) -> None:
        """
        Args:
            game_state: The game state of the environment (env.game_state)
            state: The initial state of the environment ("symbolic_obs").
        """
        potions = game_state.existing_potions()
        stones = game_state.existing_stones()

        print("PRE DRAW CUBE")
        self.edges = draw_cube(COORDS, game_state._graph)
        print("POST DRAW CUBE")
        draw_text_coords()
        self.arrow_objects = draw_arrows(potions, game_state, state)

        self.arrow_beliefs = [0]*3

        potion_objects = [0]*12
        self.potion_objects = draw_potions(state, potion_objects)

        stone_objects = [0]*3
        self.stone_objects = draw_stones(stones, stone_objects) 

        self.episode_reward = 0
        self.reward_text = wtext(text=f"<b>Episode Reward: {self.episode_reward}</b>", 
                                 pos=scene.title_anchor)
        self.trial_step_text = wtext(text=f" | <b>Trial: 1</b> | <b>Step: 0</b>", 
                                 pos=scene.title_anchor)
        self.action_text = wtext(text=f" | <b>Action: -</b> | <b>Stone: -</b>", 
                                 pos=scene.title_anchor)
        
        self._last_rendered_frame = None
        print('END OF SETUP')

    def step(self, 
             game_state, 
             state, 
             action, 
             next_state, 
             timestep,
             trial_number,
             is_new_trial) -> None:
        """
        From a step in the environment, update the visualizer.

        Args:
            game_state: The game state of the environment (env.game_state)
            state: The previous state of the environment ("symbolic_obs").
            action: The action taken by the agent.
            next_state: The next state of the environment ("symbolic_obs").
            timestep: The timestep returned by the environment.
            is_new_trial: Whether the current step is the start of a new trial
                (env.is_new_trial()).
            trial_number: The current trial number (env.trial_number).

        Returns:
            Current frame.
        """
        if not self.running: return

        done = timestep.last()
        next_state = timestep.observation["symbolic_obs"]

        self.episode_reward += timestep.reward
        self.reward_text.text = f"<b>Episode Reward: {int(self.episode_reward)}</b>"
        self.trial_step_text.text = f" | <b>Trial: {trial_number+1}</b> | <b>Step: {self.env._steps_this_trial}</b>"
        
        for i, idx in enumerate(range(15, 39, 2)):
            if next_state[idx+1] == 1:
                self.potion_objects[i].visible = False

        self.stone_objects = draw_stones(self.env.game_state.existing_stones(), self.stone_objects)

        if is_new_trial: 
            
            self.trial_step_text.text = f" | <b>Trial: {trial_number+1}</b> | <b>Step: {self.env._steps_this_trial}</b>"
            self.action_text.text = f" | <b>Action: -</b> | <b>Stone: -</b>"

            self.state = next_state
            for arrow in self.arrow_objects: arrow.visible = False  
            self.arrow_objects = draw_arrows(game_state.existing_potions(), 
                                             game_state, next_state)

            self.potion_objects = draw_potions(next_state, self.potion_objects)
            for potion in self.potion_objects: potion.visible = True 
            self.stone_objects = draw_stones(self.env.game_state.existing_stones(), self.stone_objects)
            for stone in self.stone_objects: stone.visible = True 
        
            self.p_action = np.zeros((1, self.n_actions))
            self.p_reward = np.zeros((1, 1))
        elif not done:

            stone_idx = (int(action)-1) // 7
            potion_color_idx = (int(action)-1) % 7

            if action == 0:
                stone_idx = 0
                potion_color_idx = 7
            
            self.action_text.text = f" | <b>Action: {POTION_COLOURS[potion_color_idx]}</b> | <b>Stone: {STONE_MAP[stone_idx]}</b>"
            if potion_color_idx == 6:
                self.stone_objects[stone_idx].visible = False

            stone_feats = state[stone_idx*5:(stone_idx+1)*5]
            stone_feats_p1 = next_state[stone_idx*5:(stone_idx+1)*5]

            if potion_color_idx < 6 and any(stone_feats!=stone_feats_p1):
                self.arrow_beliefs[potion_color_idx//2] = 1

            penalty = 0
            if len(game_state.existing_stones()) > 0:
                # if action doesn't have any effect on stone and action is not NoOp
                if all(state[:15]==next_state[:15]) and int(action) != 0:
                    penalty = -0.2
                # choosing an empty or non-existent potion or using a cached stone
                elif all(state==next_state) and int(action) != 0:
                    penalty = -1
                
                # choosing the same potion color consecutively 
                if int(action) == np.array(self.p_action).argmax() and int(action) % 7 != 0:
                    penalty += -1

            self.p_action = np.array([np.eye(self.n_actions)[int(action)]])
            self.p_reward = np.array([[timestep.reward + penalty]])
        else:
            self.running = False

        for arrow in self.arrow_objects: arrow.visible = False 
        self.arrow_objects = draw_arrows(game_state.existing_potions(), game_state, state)
        for arrow in self.arrow_objects:
            index = ARROWS_COLOR_MAP.index(arrow.color)
            if self.arrow_beliefs[index//2] == 1:
                arrow.opacity = 1

        # Capture the screenshot
        screenshot = ImageGrab.grab(bbox=(0, 0, 800, 600))
        img_np = np.array(screenshot)  # Convert PIL Image to numpy array
        self.last_rendered_frame = img_np

        return img_np


#     def animate(self) -> None:
#         self.running = True
#         while True:
#             rate(1/self.dt)
#             if not self.running:
#                 break
            
# if __name__ == "__main__":

#     # NOTE: OLD CODE from original repo

#     os.environ['KMP_DUPLICATE_LIB_OK']='True'

#     parser = argparse.ArgumentParser(description='Paramaters')
#     parser.add_argument('-c', '--config',  type=str, default="config.yaml", help='path of config file')
#     args = parser.parse_args()

#     with open(args.config, 'r', encoding="utf-8") as fin:
#         config = yaml.load(fin, Loader=yaml.FullLoader)

#     visualizer = AlchemyVisualizer(config)
#     visualizer.setup(episode=2)
#     visualizer.animate() 