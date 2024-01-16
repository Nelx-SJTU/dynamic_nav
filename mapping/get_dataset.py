import os
from random import choice
import yaml
import numpy as np
import cv2 as cv
import igibson
from igibson.envs.igibson_env import iGibsonEnv
import pandas as pd
import copy

config_data = yaml.load(open("configs/turtlebot_nav.yaml", "r"), Loader=yaml.FullLoader)
scene = ["Beechwood_0_int"]
env = iGibsonEnv(config_file=config_data, scene_id=choice(scene), mode="gui_non_interactive")

obs = env.reset()
q = None    # recording keyboard input (in keyboard-control mode)
orn = pos = None    # orn: global angle    pos: global position
rgb = depth = optical_flow = None   # recording sensor information
control_type = "keyboard"   # control mode: 1.random  2.keyboard

posx_list = []
posy_list = []
orn_list = []
max_steps = 1600

get_dataset = False
DS_num = 100
if get_dataset:
    path = "./YOUR_DATASET_PATH/dataset/"+scene[0]+"/DS_"+str(DS_num)
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        os.mkdir(path+'/rgb')
        os.mkdir(path+'/depth')
        os.mkdir(path+'/position')
        os.mkdir(path + '/seg')


def select_action(env, type="random"):
    global q
    if type == "random":
        return env.action_space.sample(), " "
    elif type == "keyboard":
        if q == ord("w"):
            return np.array([0.7, 0.0]), " "
        elif q == ord("a"):
            return np.array([0.0, -0.2]), " "
        elif q == ord("d"):
            return np.array([0.0, 0.2]), " "
        elif q == ord("s"):
            return np.array([-0.7, 0.0]), " "
        elif q == ord(" "):
            return np.array([0.0, 0.0]), " "

        # open and close the door
        elif q == ord("o"):
            return np.array([0.0, 0.0]), "open"
        elif q == ord("p"):
            return np.array([0.0, 0.0]), "close"



for i in range(max_steps):
    if control_type == "keyboard":  # get the input of keyboard   w:up  s:down  a:left  d:right
        while True:
            q = cv.waitKey(1)
            if not q == -1:
                break
    action, action_info = select_action(env, control_type)

    if action_info == "open":
        env.scene.should_open_all_doors = True
        env.scene.reset_scene_objects()
    elif action_info == "close":
        env.scene.should_open_all_doors = False
        env.scene.reset_scene_objects()

    obs, reward, done, info = env.step(action)

    # get the information (of the environment) from the sensors (there are many other sensors except for these three)
    rgb, depth = obs['rgb'], obs['depth']
    r = copy.copy(rgb[:, :, 0])
    rgb[:, :, 0] = rgb[:, :, 2]
    rgb[:, :, 2] = r

    # get the GLOBAL position and angle of the robot
    pos_tmp = env.robots[0].get_position()[:2]
    orn_tmp = env.robots[0].get_rpy()[-1:-2:-1]

    print('pos :', pos_tmp)
    print('orn :', orn_tmp)

    if get_dataset:
        posx_list.append(pos_tmp[0])
        posy_list.append(pos_tmp[1])
        orn_list.append(orn_tmp[0])

        cv.imwrite("./YOUR_DATASET_PATH/dataset/"+scene[0]+"/DS_"+str(DS_num)+"/rgb/DS"+str(DS_num)+"_rgb_"+str(i)+".jpg", rgb*225)
        cv.imwrite("./YOUR_DATASET_PATH/dataset/"+scene[0]+"/DS_"+str(DS_num)+"/depth/DS"+str(DS_num)+"_d_" + str(i) + ".jpg", depth*225)


if get_dataset:
    dataframe = pd.DataFrame({"posx": posx_list, "posy": posy_list, "orn": orn_list})
    dataframe.to_csv("./YOUR_DATASET_PATH/dataset/"+scene[0]+"/DS_"+str(DS_num)+"/position/DS"+str(DS_num)+"_pos.csv", index=False, sep=',')

env.close()
