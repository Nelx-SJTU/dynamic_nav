import numpy as np
import cv2
import copy
import torch
from PIL import Image
from torchvision import utils, transforms


def x_map(x):
    return -106.9711815*x + 502.5497438 + 500


def y_map(y):
    return 103.6528545*y + 503.570678 + 500


def map_combine(map_g, map_g_size, map_l, map_l_size, robot_pos_t, robot_ang):
    robot_pos = robot_pos_t
    robot_pos[0] += map_g_size//2
    robot_pos[1] += map_g_size//2

    robot_ang_degree = robot_ang * 180/np.pi
    robot_ang_rad = robot_ang

    map_l_padding = np.full((2*map_l_size, 2*map_l_size), 127, np.int16)

    map_l_padding[int(0.5*map_l_size):int(1.5*map_l_size), int(0.5*map_l_size):int(1.5*map_l_size)] = map_l

    _M = cv2.getRotationMatrix2D((int(map_l_size),int(map_l_size)) , float(robot_ang_degree), 1)

    map_l_padding = cv2.warpAffine(src=map_l_padding,
                                   M=_M,
                                   dsize=map_l_padding.shape,
                                   borderValue=(255, 255, 255))

    map_l_rot = map_l_padding[80:-80, 80:-80]

    robot_pos_new = (int(map_l_size/2-80+map_l_size/2+map_l_size/2*np.sin(robot_ang_rad)),
                     int(map_l_size/2-80+map_l_size/2+map_l_size/2*np.cos(robot_ang_rad)))

    map_g[int(robot_pos[1] - robot_pos_new[1]):
          int(robot_pos[1] - robot_pos_new[1] + map_l_rot.shape[1]),
          int(robot_pos[0] - robot_pos_new[0]):
          int(robot_pos[0] - robot_pos_new[0] + map_l_rot.shape[0])] \
        += np.array((map_l_rot-127) * 0.08, np.int16)

    _, map_g = cv2.threshold(map_g, 230, 255, cv2.THRESH_TRUNC)
    _, map_g = cv2.threshold(map_g, 10, 255, cv2.THRESH_TOZERO)

    return map_g


def map_polar_2_xy(local_polar_map, block_thickness, view_angle):
    src = np.ascontiguousarray(local_polar_map)

    local_polar_map = cv2.rotate(local_polar_map, cv2.ROTATE_90_CLOCKWISE)
    local_polar_map = cv2.rotate(local_polar_map, cv2.ROTATE_90_CLOCKWISE)
    local_polar_map = cv2.rotate(local_polar_map, cv2.ROTATE_90_CLOCKWISE)
    for i in range(224):
        for j in range(224 - block_thickness):
            if np.max(local_polar_map[i][224 - 1 - j - block_thickness:224 - 1 - j]) < 100:
                local_polar_map[i][:224 - 1 - j - block_thickness] = 127
    local_polar_map = cv2.rotate(local_polar_map, cv2.ROTATE_90_CLOCKWISE)

    center = (int(local_polar_map.shape[1]), int(local_polar_map.shape[0] / 2))
    maxRadius = local_polar_map.shape[0]

    lin_polar_img_padding = np.full((704, 224), 127, dtype=np.uint8)
    lin_polar_img_padding[int(704 / 2 - view_angle):int(704 / 2 + view_angle)] = \
        cv2.resize(cv2.rotate(local_polar_map, cv2.ROTATE_90_CLOCKWISE), (224, view_angle * 2))

    flags = cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS

    recovered_lin_polar_img = cv2.warpPolar(lin_polar_img_padding, (src.shape[0], src.shape[1] * 2), center, maxRadius,
                                            flags | cv2.WARP_INVERSE_MAP)

    recovered_lin_polar_img = recovered_lin_polar_img[:224]
    recovered_lin_polar_img = cv2.rotate(recovered_lin_polar_img, cv2.ROTATE_90_CLOCKWISE)

    for h in range(224):
        length = int(h * (224 / 2) // 224)
        for w in range(length + 1):
            recovered_lin_polar_img[h][w - 1] = 127
            recovered_lin_polar_img[h][224 - w - 1] = 127

    for w in range(224):
        for h in range(60):
            if np.linalg.norm([w - 112, h - 224]) > 223:
                recovered_lin_polar_img[h][w] = 127

    recovered_lin_polar_img = cv2.resize(recovered_lin_polar_img, (320, 320))
    cv2.imwrite("./running/recovered_lin_polar_img.jpg", recovered_lin_polar_img)
    return lin_polar_img_padding


def map_cut(_map, robot_position, robot_angle, width, depth):
    x = int(robot_position[0] - width // 2)
    y = int(robot_position[1] - depth)
    w = int(width)
    d = int(depth)

    rows, cols = _map.shape

    _M = cv2.getRotationMatrix2D((int(robot_position[0]), int(robot_position[1])),
								 robot_angle, 1)
    _map = cv2.warpAffine(src=_map, M=_M, dsize=(rows, cols), borderValue=(255, 255, 255))  # M为上面的旋转矩阵

    roi = (x, y, w, d)
    if roi != (0, 0, 0, 0):
        crop = _map[y:y + d, x:x + w]
        return crop


def get_information(env, obs, reward, done, info, action, step):
    # get the information (of the environment) from the sensors (there are many other sensors except for these three)
    rgb, depth = obs['rgb'], obs['depth']
    r = copy.copy(rgb[:, :, 0])
    rgb[:, :, 0] = rgb[:, :, 2]
    rgb[:, :, 2] = r
    rgb = rgb * 225
    depth = depth * 225
    cv2.imwrite("./running/rgb.jpg", rgb)
    cv2.imwrite("./running/depth.jpg", depth)

    # get the GLOBAL position and angle of the robot
    pos_tmp = env.robots[0].get_position()[:2]
    orn_tmp = env.robots[0].get_rpy()[-1:-2:-1]

    return rgb, depth, pos_tmp, orn_tmp


def mapping_task(loader, device, block_thickness, view_angle, pos_tmp, pos_ini,
                 orn_tmp, map_global, map_global_size, mapping_model):
    rgb = Image.open("./running/rgb.jpg")
    rgb = loader(rgb).unsqueeze(0)
    rgb = rgb.to(device, torch.float)

    local_polar_map = mapping_model(rgb)
    utils.save_image(local_polar_map[0][0],
                     "./running/local_polar_map.jpg", normalize=True)

    local_polar_map = cv2.imread("./running/local_polar_map.jpg", cv2.IMREAD_GRAYSCALE)

    # --------------------------- map_polar_2_xy --------------------------- #
    lin_polar_img_padding = map_polar_2_xy(local_polar_map=local_polar_map, block_thickness=block_thickness,
                                           view_angle=view_angle)

    recovered_lin_polar_img = cv2.imread("./running/recovered_lin_polar_img.jpg", cv2.IMREAD_GRAYSCALE)

    robot_pos_relative = [int(x_map(pos_tmp[0]) - pos_ini[0]),
                          int(y_map(pos_tmp[1]) - pos_ini[1])]
    robot_orn_relative = orn_tmp + np.pi / 2

    map_global = map_combine(map_global, map_global_size, recovered_lin_polar_img, 320,
                             robot_pos_relative, robot_orn_relative)

    cv2.imwrite("./running/map.jpg", map_global)

    return local_polar_map, lin_polar_img_padding, recovered_lin_polar_img, robot_pos_relative, \
           robot_orn_relative, map_global


def global_map_coordinate_align(scene, robot_pos_relative, pos_ini):
    map_global_show = cv2.imread("./running/map.jpg", cv2.IMREAD_GRAYSCALE)

    # ------------ Setting for Rs_int ------------ #
    if scene[0] == 'Rs_int':
        cv2.circle(map_global_show, robot_pos_relative, 5, 200, 2)
        map_global_show = map_global_show[int(1500 - pos_ini[1]):int(2500 - pos_ini[1]),
                          int(1500 - pos_ini[0]):int(2500 - pos_ini[0])]

    # ------------ Setting for Beechwood_0_int ------------ #
    if scene[0] == 'Beechwood_0_int':
        cv2.circle(map_global_show, robot_pos_relative, 5, 200, 2)
        map_global_show = map_global_show[int(2000 - pos_ini[1]):int(4000 - pos_ini[1]),
                          int(2200 - pos_ini[0]):int(4200 - pos_ini[0])]

    cv2.namedWindow('Global Map', cv2.WINDOW_NORMAL)
    cv2.imwrite("./running/map_1000.jpg", map_global_show)

    return map_global_show


def get_grid_map(scene, map_global_show, robot_pos_relative, pos_ini):
    # grid map (only black-obstacle, white-reachable area, grey-unexplored)
    map_grid = cv2.imread("./running/map.jpg", cv2.IMREAD_GRAYSCALE)
    _, map_grid = cv2.threshold(map_grid, 187, 255, cv2.THRESH_BINARY)
    cv2.circle(map_global_show, robot_pos_relative, 5, 170, 2)

    # ------------ Setting for Rs_int ------------ #
    if scene[0] == 'Rs_int':
        map_grid_show = map_grid[int(1500 - pos_ini[1]):int(2500 - pos_ini[1]),
                        int(1500 - pos_ini[0]):int(2500 - pos_ini[0])]

    # ------------ Setting for Beechwood_0_int ------------ #
    if scene[0] == 'Beechwood_0_int':
        map_grid_show = map_grid[int(2000 - pos_ini[1]):int(4000 - pos_ini[1]),
                        int(2200 - pos_ini[0]):int(4200 - pos_ini[0])]

    cv2.imwrite("./running/map_grid_1000.jpg", map_grid_show)

    return map_grid, map_grid_show


def topological_change_judge(map_grid, robot_pos_relative, robot_orn_relative, lin_polar_img_padding,
                             view_angle, topo_judge_model, topo_judge_mode):
    # Get local map from memoried global map
    map_local_memory = map_cut(_map=map_grid, robot_position=robot_pos_relative,
                               robot_angle=-robot_orn_relative, # angle is in degree
                               width=320, depth=320)
    map_local_memory = cv2.resize(map_local_memory, (224, 224))

    src = np.rot90(map_local_memory)
    src = np.ascontiguousarray(src)
    if src is None:
        print("Could not initialize capturing...\n")

    center = (int(src.shape[1]), int(src.shape[0] / 2))
    maxRadius = map_local_memory.shape[0]

    lin_polar_img = cv2.warpPolar(src, None, center, maxRadius, cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS)

    for i in range(lin_polar_img.shape[0]):
        for j in range(5, lin_polar_img.shape[1]):
            if lin_polar_img[i, j] <= 20:
                lin_polar_img[i, j:] = 0
                break

    lin_polar_img = lin_polar_img[int((lin_polar_img.shape[0] / 2 - view_angle)):
                                  -int((lin_polar_img.shape[0] / 2 - view_angle))]
    lin_polar_img = cv2.resize(lin_polar_img, (224, 224))

    lin_polar_img = cv2.rotate(lin_polar_img, cv2.ROTATE_90_CLOCKWISE)  # rotation 90 degree
    lin_polar_img = cv2.rotate(lin_polar_img, cv2.ROTATE_90_CLOCKWISE)
    lin_polar_img = cv2.rotate(lin_polar_img, cv2.ROTATE_90_CLOCKWISE)

    _, lin_polar_img = cv2.threshold(lin_polar_img, 20, 255, cv2.THRESH_BINARY)

    lin_polar_img_compare = cv2.resize(lin_polar_img, (50, 50))  # memory
    lin_polar_img_padding_compare = cv2.resize(
        lin_polar_img_padding[
        int(lin_polar_img_padding.shape[0] / 2) - 26:int(lin_polar_img_padding.shape[0] / 2) + 26],
        (50, 50))
    lin_polar_img_padding_compare = cv2.rotate(lin_polar_img_padding_compare, cv2.ROTATE_90_CLOCKWISE)
    lin_polar_img_padding_compare = cv2.rotate(lin_polar_img_padding_compare, cv2.ROTATE_90_CLOCKWISE)
    lin_polar_img_padding_compare = cv2.rotate(lin_polar_img_padding_compare, cv2.ROTATE_90_CLOCKWISE)

    action_info = " "

    if topo_judge_mode == "model":
        topo_change_mode, action_info = topo_judge_model(lin_polar_img_compare, lin_polar_img_padding_compare)
    elif topo_judge_mode == "tradition":
        memory_open_area = 0
        perception_open_area = 0
        for xcnt in range(50):
            for ycnt in range(50):
                if lin_polar_img_compare[xcnt, ycnt] >= 200:
                    memory_open_area += 1
                if lin_polar_img_padding_compare[xcnt, ycnt] >= 200:
                    perception_open_area += 1

        # print(memory_open_area, perception_open_area)
        if perception_open_area - memory_open_area > 900:
            print('door open')
            topo_change_mode = 1
            action_info = "extract_compulsively"
        elif perception_open_area - memory_open_area < -900:
            print('door close')
            topo_change_mode = 2
            action_info = "save_compulsively"

    return action_info


def memory_bank_proceeding(action_info, scene, memory_bank_list, pos_tmp, orn_tmp, map_grid_show,
                           memory_bank_name, door_state):
    # compulsively memory save
    # -------------------------- CONECTIVITY DISCRIMINATOR -------------------------- #
    if action_info == "save_compulsively":
        # ------------ Setting for Rs_int ------------ #
        if scene[0] == 'Rs_int':
            # Rs_int bathroom: (708, 650) -> (908, 850) in (1000*1000) cv2 coordinate
            # Rs_int bathroom: (708, 1000-650) -> (908, 1000-850) in (1000*1000) numpy coordinate
            memory_tmp = copy.copy(map_grid_show[650:850, 708:908])
            memory_bank_list.append(memory_tmp)

        # ------------ Setting for Beechwood_0_int ------------ #
        if scene[0] == 'Beechwood_0_int':
            if pos_tmp[0] > -10.24 and pos_tmp[0] < -8.24 and pos_tmp[1] > 1.10 and pos_tmp[1] < 3.10 and door_state[
                0] == 0:
                # Beechwood_0_int restaurant: (1590, 680) -> (1963, 1095) in cv2 coordinate
                memory_tmp = copy.copy(map_grid_show[680:1095, 1590:1963])
                memory_bank_list.append(memory_tmp)
                memory_bank_name.append('restaurant')
                door_state[0] = 1

            if pos_tmp[0] > -3.08 and pos_tmp[0] < -1.08 and pos_tmp[1] > 0.58 and pos_tmp[1] < 2.58:
                if orn_tmp[0] > -np.pi / 2 + 0.15 and orn_tmp[0] < 0 - 0.15 and door_state[1] == 0:
                    # Beechwood_0_int reading room: (670, 850) -> (1010, 1170) in cv2 coordinate
                    memory_tmp = copy.copy(map_grid_show[850:1170, 670:1010])
                    memory_bank_list.append(memory_tmp)
                    memory_bank_name.append('reading room')
                    door_state[1] = 1
                if orn_tmp[0] > 0 + 0.15 and orn_tmp[0] < np.pi / 2 - 0.15 and door_state[2] == 0:
                    # Beechwood_0_int laundry: (670, 1170) -> (1010, 1434) in cv2 coordinate
                    memory_tmp = copy.copy(map_grid_show[1170:1434, 670:1010])
                    memory_bank_list.append(memory_tmp)
                    memory_bank_name.append('laundry')
                    door_state[2] = 1
                if orn_tmp[0] > np.pi / 2 + 0.15 and orn_tmp[0] < np.pi - 0.15 and door_state[3] == 0:
                    # Beechwood_0_int bathroom: (1010, 1170) -> (1220, 1434) in cv2 coordinate
                    memory_tmp = copy.copy(map_grid_show[1170:1434, 1010:1220])
                    memory_bank_list.append(memory_tmp)
                    memory_bank_name.append('bathroom')
                    door_state[3] = 1

    elif action_info == "extract_compulsively":
        # ------------ Setting for Rs_int ------------ #
        if scene[0] == 'Rs_int':
            del memory_bank_list[0]

        # ------------ Setting for Beechwood_0_int ------------ #
        if scene[0] == 'Beechwood_0_int':
            if pos_tmp[0] > -10.24 and pos_tmp[0] < -8.24 and pos_tmp[1] > 1.10 and pos_tmp[1] < 3.10 and door_state[
                0] == 1:
                # Beechwood_0_int restaurant
                door_state[0] = 0
                del memory_bank_list[memory_bank_name.index('restaurant')]
                del memory_bank_name[memory_bank_name.index('restaurant')]

            if pos_tmp[0] > -3.08 and pos_tmp[0] < -1.08 and pos_tmp[1] > 0.58 and pos_tmp[1] < 2.58:
                if orn_tmp[0] > -np.pi / 2 + 0.15 and orn_tmp[0] < 0 - 0.15 and door_state[1] == 1:
                    # Beechwood_0_int reading room
                    door_state[1] = 0
                    del memory_bank_list[memory_bank_name.index('reading room')]
                    del memory_bank_name[memory_bank_name.index('reading room')]
                if orn_tmp[0] > 0 + 0.15 and orn_tmp[0] < np.pi / 2 - 0.15 and door_state[2] == 1:
                    # Beechwood_0_int laundry
                    door_state[2] = 0
                    del memory_bank_list[memory_bank_name.index('laundry')]
                    del memory_bank_name[memory_bank_name.index('laundry')]
                if orn_tmp[0] > np.pi / 2 + 0.15 and orn_tmp[0] < np.pi - 0.15 and door_state[3] == 1:
                    # Beechwood_0_int bathroom
                    door_state[3] = 0
                    del memory_bank_list[memory_bank_name.index('bathroom')]
                    del memory_bank_name[memory_bank_name.index('bathroom')]

    # Memory Bank Visualize
    if len(memory_bank_list) >= 1:
        # ------------ Setting for Rs_int ------------ #
        if scene[0] == 'Rs_int':
            # only delete on map_grid_show
            map_grid_show[650:850, 708:908] = 0
            memory_bank_visualize = memory_bank_list[0]
            for mbcnt in range(len(memory_bank_list) - 1):
                memory_bank_visualize = np.hstack((memory_bank_visualize, memory_bank_list[mbcnt + 1]))

        # ------------ Setting for Beechwood_0_int ------------ #
        if scene[0] == 'Beechwood_0_int':
            if door_state[0] == 1:
                map_grid_show[680:1095, 1590:1963] = 0
            if door_state[1] == 1:
                map_grid_show[850:1170, 670:1010] = 0
            if door_state[2] == 1:
                map_grid_show[1170:1434, 670:1010] = 0
            if door_state[3] == 1:
                map_grid_show[1170:1434, 1010:1220] = 0
            memory_bank_visualize = memory_bank_list[0]
            memory_bank_visualize = cv2.resize(memory_bank_visualize,
                                               (int(memory_bank_visualize.shape[0] * 200 / memory_bank_visualize.shape[
                                                   1]), 200))
            for mbcnt in range(len(memory_bank_list) - 1):
                tmp = cv2.resize(memory_bank_list[mbcnt + 1], (
                int(memory_bank_list[mbcnt + 1].shape[0] * 200 / memory_bank_list[mbcnt + 1].shape[1]), 200))
                memory_bank_visualize = np.hstack((memory_bank_visualize, tmp))

        cv2.imshow('Memory Bank', memory_bank_visualize)
    else:
        cv2.imshow('Memory Bank', np.zeros((200, 200)))

    return memory_bank_list


def visualize_map(scene, map_global_show, pos_tmp, orn_tmp, map_mode, map_grid_show,
                  map_grid_3color, map_explore_edge_show):
    if scene[0] == 'Beechwood_0_int':
        cv2.circle(map_global_show,
                   [int((pos_tmp[0] * -107 + 502 + 500 - 190) * 300 / 2000),
                    int((pos_tmp[1] * 104 + 503 + 500) * 300 / 2000)],
                   5, 50, 2)  # Robot pos
        cv2.line(map_global_show,
                 [int((pos_tmp[0] * -107 + 502 + 500 - 190) * 300 / 2000),
                  int((pos_tmp[1] * 104 + 503 + 500) * 300 / 2000)],
                 [int((pos_tmp[0] * -107 + 502 + 500 - 190) * 300 / 2000 - 40 * np.cos(orn_tmp)),
                  int((pos_tmp[1] * 104 + 503 + 500) * 300 / 2000 + 40 * np.sin(orn_tmp))],
                 50, 2)
    else:
        cv2.circle(map_global_show,
                   [int((x_map(pos_tmp[0]) - 500) * 300 / 1000), int((y_map(pos_tmp[1]) - 500) * 300 / 1000)],
                   5, 50, 2)  # Robot pos
        cv2.line(map_global_show,
                 [int((x_map(pos_tmp[0]) - 500) * 300 / 1000), int((y_map(pos_tmp[1]) - 500) * 300 / 1000)],
                 [int((x_map(pos_tmp[0]) - 500) * 300 / 1000 - 40 * np.cos(orn_tmp)),
                  int((y_map(pos_tmp[1]) - 500) * 300 / 1000 + 40 * np.sin(orn_tmp))],
                 50, 2)

    if map_mode == "simple":
        map_global_show = cv2.resize(map_global_show, (100, 100))
        map_grid_show = cv2.resize(map_grid_show, (100, 100))

        map_visualize_1 = np.vstack((map_global_show, map_grid_show))

        cv2.imshow('Global Map', map_visualize_1)
        cv2.waitKey(60)
    elif map_mode == "all":
        map_visualize_1 = np.vstack((map_global_show, map_grid_show))
        map_visualize_2 = np.vstack((map_grid_3color, map_explore_edge_show))
        map_visualize = np.hstack((map_visualize_1, map_visualize_2))
        cv2.imwrite("./running/test_2.jpg", map_visualize)
        cv2.imshow('Global Map', map_visualize)
        cv2.waitKey(60)


