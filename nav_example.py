import cv2
import copy
import igibson
import math
import os
import torch
import yaml
import numpy as np
from PIL import Image
from random import choice
from igibson.envs.igibson_env import iGibsonEnv
from torchvision import utils, transforms
from mapping.mapping_model import Mapping_Model
from planning.planning_model import Planning_Model
from mapping.topo_judge_model import Topo_Judge_Model
from src.planning_utils import *
from src.mapping_utils import *
from src.robot import select_action


# scene = ['Beechwood_0_int', 'Ihlen_0_int', 'Rs_int']
scene = ['Rs_int']
scene_size = "normal"

env_config_data = yaml.load(open("configs/env_config.yaml", "r"), Loader=yaml.FullLoader)
nav_config_data = yaml.load(open("configs/nav_config.yaml", "r"), Loader=yaml.FullLoader)
env = iGibsonEnv(config_file=env_config_data, scene_id=choice(scene), mode="gui_non_interactive")
env_nav = mapPlanningEnv(robot_size=2, env_size=128)

obs = env.reset()
q = None    # recording keyboard input (in keyboard-control mode)
orn = pos = None    # orn: global angle    pos: global position
rgb = depth = optical_flow = None   # recording sensor information

control_type = "auto"   # control mode: 1.random  2.keyboard
preload_map = False
topo_judge_mode = "none"  # model, tradition, none
map_mode = "all"  # simple, all

# ------------------------------- main ------------------------------- #
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

imsize = (224, 224)  # image size for resnet 50
loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

# ------------------------------- LOAD MODEL ------------------------------- #
load_model = True
load_model_name = "model_Beechwood_0_int_0119"

mapping_model = Mapping_Model(in_channels=3).to(device)
mapping_model.load_state_dict(torch.load("./model/mapping_model_default.pt", map_location=device))
topo_judge_model = Topo_Judge_Model(in_channels=2).to(device)
topo_judge_model.load_state_dict(torch.load("./model/topo_judge_model_default.pt", map_location=device))
nav_model = Planning_Model(in_channels=1, out_channels=1).to(device)
nav_model.load_state_dict(torch.load("./model/planning_model_default.pt", map_location=device))

max_steps = 1000
view_angle = 52
block_thickness = 30

# nav parameters
small_size = 128
edge_padding = 150
depth_num = 1080


# ------------ Setting of scene size ------------ #
if scene_size == "normal":
    map_global_size = 2000
    map_global = np.full(shape=(map_global_size, map_global_size), fill_value=127, dtype=np.int16)
    map_explore_area = np.full(shape=(map_global_size, map_global_size), fill_value=0, dtype=np.int16)
    pos_ini = [1000,1000]
    orn_ini = 0
    door_state = [0,  # restaurant
                  0,  # reading room
                  0,  # laundry
                  0,  # bathroom
                  0]  # restaurant 2 out

elif scene_size == "large":
    map_global_size = 4000
    map_global = np.full(shape=(map_global_size, map_global_size), fill_value=127, dtype=np.int16)
    map_explore_area = np.full(shape=(map_global_size, map_global_size), fill_value=0, dtype=np.int16)
    pos_ini = [2000,2000]
    orn_ini = 0
    door_state = [0,  # restaurant
                  0,  # reading room
                  0,  # laundry
                  0,  # bathroom
                  0]  # restaurant 2 out


memory_bank_list = []
memory_bank_name = []
auto_save = False  # in region, auto save
auto_extract = False  # in region, auto extract

action = np.array([0.0, 0.2])
action_info = " "
map_explore_edge_show = np.full((300,300), 0).astype(np.uint8)
map_grid_3color = np.full((300, 300), 127).astype(np.uint8)  # black, white, grey
nav_model_output = -20

print("prepared")


# -------------------------------------------------------------------------------------------------- #

for step in range(max_steps):

    if control_type == "keyboard":  # get the input of keyboard   w:up  s:down  a:left  d:right
        while True:
            q = cv2.waitKey(1)
            if not q == -1:
                break
        action, action_info = select_action(env, q, control_type)

    if action_info == "open":
        env.scene.should_open_all_doors = True
        env.scene.reset_scene_objects()
        open_all_doors = True
    elif action_info == "close":
        env.scene.should_open_all_doors = False
        env.scene.reset_scene_objects()
        open_all_doors = False
    elif action_info == "auto_save":
        if auto_save == True:
            auto_save = False
        elif auto_save == False:
            auto_save = True
    elif action_info == "auto_extract":
        if auto_extract == True:
            auto_extract = False
        elif auto_extract == False:
            auto_extract = True

    obs, reward, done, info = env.step(action)

    rgb, depth, pos_tmp, orn_tmp = get_information(env, obs, reward, done, info, action, step)

    if step == 0:
        pos_ini = [x_map(pos_tmp[0]), y_map(pos_tmp[1])]
        orn_ini = orn_tmp
        print('position initialized')
        print('pos_ini = ', pos_ini)

    local_polar_map, lin_polar_img_padding, recovered_lin_polar_img, robot_pos_relative, \
    robot_orn_relative, map_global = mapping_task(loader, device, block_thickness, view_angle,
                                                  pos_tmp, pos_ini,orn_tmp, map_global,
                                                  map_global_size, mapping_model)

    map_global_show = global_map_coordinate_align(scene, robot_pos_relative, pos_ini)

    map_grid, map_grid_show = get_grid_map(scene, map_global_show, robot_pos_relative, pos_ini)

    action_info = topological_change_judge(map_grid, robot_pos_relative, robot_orn_relative,
                                           lin_polar_img_padding, view_angle, topo_judge_model,
                                           topo_judge_mode)

    memory_bank_list = memory_bank_proceeding(action_info, scene, memory_bank_list, pos_tmp, orn_tmp,
                                              map_grid_show, memory_bank_name, door_state)

    map_global_show = cv2.resize(map_global_show, (300, 300))
    map_grid_show = cv2.resize(map_grid_show, (300, 300))  # only black and white

    map_explore_area_show = cv2.resize(map_explore_area, (300, 300))
    _, map_explore_area_show = cv2.threshold(map_explore_area_show, 187, 255, cv2.THRESH_BINARY)

    # get explore edge
    if step % 20 == 0 and step >= 30:

        target_explore_pos = get_next_waypoint(scene, map_global_show, map_grid_show, pos_tmp)

        navigation_task(scene, device, pos_tmp, edge_padding, small_size, map_grid_show, depth_num,
                        target_explore_pos, nav_model)

        cv2.circle(map_explore_edge_show,
                   [target_explore_pos[1], target_explore_pos[0]],
                   5, 190, 2)  # target_explore_point
        cv2.circle(map_global_show,
                   [target_explore_pos[1], target_explore_pos[0]],
                   5, 190, 2)  # target_explore_point

    apply_action(step, orn_tmp, nav_model_output)

    visualize_map(scene, map_global_show, pos_tmp, orn_tmp, map_mode, map_grid_show,
                  map_grid_3color, map_explore_edge_show)


cv2.imwrite("./running/map.jpg", map_global)
env.close()
