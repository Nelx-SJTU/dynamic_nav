import numpy as np
import torch
import cv2
import math
from src.mapping_utils import x_map, y_map


def compute_orn(map_middle_point, r, local_map_size):
    middle_point = np.zeros(2)
    middle_flag = np.zeros(2)
    for h in range(r):
        if map_middle_point[int(map_middle_point.shape[0] / 2 - r / 2)][
            int(map_middle_point.shape[1] / 2 - r / 2) + h] > 0:
            middle_point[0] = int(map_middle_point.shape[0] / 2 - r / 2)
            middle_point[1] = int(map_middle_point.shape[1] / 2 - r / 2) + h
        elif map_middle_point[int(map_middle_point.shape[0] / 2 + r / 2)][
            int(map_middle_point.shape[1] / 2 - r / 2) + h] > 0:
            middle_point[0] = int(map_middle_point.shape[0] / 2 + r / 2)
            middle_point[1] = int(map_middle_point.shape[1] / 2 - r / 2) + h
    for w in range(r):
        if map_middle_point[int(map_middle_point.shape[1] / 2 - r / 2) + w][
            int(map_middle_point.shape[1] / 2 - r / 2)] > 0:
            middle_point[0] = int(map_middle_point.shape[1] / 2 - r / 2) + w
            middle_point[1] = int(map_middle_point.shape[1] / 2 - r / 2)
        elif map_middle_point[int(map_middle_point.shape[1] / 2 - r / 2) + w][
            int(map_middle_point.shape[1] / 2 + r / 2)] > 0:
            middle_point[0] = int(map_middle_point.shape[1] / 2 - r / 2) + w
            middle_point[1] = int(map_middle_point.shape[1] / 2 + r / 2)
    middle_point = np.zeros(2)

    if (middle_point == middle_flag).all():
        for w in range(map_middle_point.shape[0]):
            for h in range(map_middle_point.shape[1]):
                if map_middle_point[w][h] > 0:
                    middle_point[0] = w
                    middle_point[1] = h
                elif map_middle_point[local_map_size - 1 - w][h] > 0:
                    middle_point[0] = local_map_size - 1 - w
                    middle_point[1] = h
                elif map_middle_point[h][w] > 0:
                    middle_point[0] = h
                    middle_point[1] = w
                elif map_middle_point[h][local_map_size - 1 - w] > 0:
                    middle_point[0] = h
                    middle_point[1] = local_map_size - 1 - w

    print('midpoint', middle_point)

    rot = np.arctan2(int(middle_point[0] - map_middle_point.shape[0] / 2),
                     int(middle_point[1] - map_middle_point.shape[1] / 2)) + np.pi
    rot = torch.tensor([rot])
    return rot


def generate_depth(_map, robot_position, num):
    rows, cols = _map.shape
    depth_round = np.zeros((1, num))
    rotation = 360 / num
    x = robot_position[0]
    y = robot_position[1]
    for i in range(int(num/4)):
        flag1, flag2, flag3, flag4 = 1, 1, 1, 1
        _M = cv2.getRotationMatrix2D((int(robot_position[1]), int(robot_position[0])), rotation*i, 1)
        map_t = cv2.warpAffine(src=_map, M=_M, dsize=(rows, cols), borderValue=(255, 255, 255))
        for j in range(rows):
            if flag1 and map_t[x][y-j] < 200:
                depth_round[0][i] = j
                flag1 = 0
            if flag2 and map_t[x-j][y] < 200:
                depth_round[0][i+int(num/4)] = j
                flag2 = 0
            if flag3 and map_t[x][y+j] < 200:
                depth_round[0][i+int(num/2)] = j
                flag3 = 0
            if flag4 and map_t[x+j][y] < 200:
                depth_round[0][i+int(num/4*3)] = j
                flag4 = 0
    return depth_round


def adjust_output(model_output, _map):
    buf = np.array([100, 100, 100, 100]).astype(int)
    x = model_output[0]
    y = model_output[1]
    output = np.zeros(2).astype(int)
    flag1, flag2, flag3, flag4 = 0, 0, 0, 0
    for j in range(_map.shape[0]):
        if flag1 < 3 and _map[x][max(y - j, 0)] > 220:
            buf[0] = j
            flag1 += 1
        if flag2 < 3 and _map[max(x - j,0)][y] > 200:
            buf[1] = j
            flag2 += 1
        if flag3 < 3 and _map[x][min(y + j, _map.shape[0]-1)] > 200:
            buf[2] = j
            flag3 += 1
        if flag4 < 3 and _map[min(x + j, _map.shape[0]-1)][y] > 200:
            buf[3] = j
            flag4 += 1
    idx = buf.argmin()
    if idx == 0:
        output[0] = model_output[0]
        output[1] = model_output[1] - buf[idx]
    if idx == 1:
        output[0] = model_output[0] - buf[idx]
        output[1] = model_output[1]
    if idx == 2:
        output[0] = model_output[0]
        output[1] = model_output[1] + buf[idx]
    if idx == 3:
        output[0] = model_output[0] + buf[idx]
        output[1] = model_output[1]
    return output


class mapPlanningEnv():
    def __init__(self, robot_size=20, env_size=2000):
        self.env_size = env_size
        self.action_space = 3
        self.robot_size = robot_size
        self.collision = False
        self.reach_goal = False

        self.current_step = 0
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = []  # A list to save all tepisode scores, used to check if task is solved

        self.robot_pos_ini = np.array([0, 0])
        self.robot_pos = np.array([0, 0])
        self.robot_orn_ini = 0.0
        self.robot_orn = 0.0
        self.target_pos = np.array([0, 0])

        self.forward_distance = 3
        self.turn_angle = 5 * np.pi / 180  # in rad

    def generate_random_position(self):
        for _ in range(200):
            random_pos = np.random.randint(low=self.robot_size, high=np.min(self.env_size) - self.robot_size - 1,
                                           size=2, dtype='int')

            if np.min(self.map[random_pos[0] - self.robot_size:random_pos[0] + self.robot_size,
                      random_pos[1] - self.robot_size:random_pos[1] + self.robot_size]) >= 100:
                return random_pos
        raise ValueError('cannot find a empty position in 200 random steps')


    def reset(self, map, map_torch):
        self.map = map
        self.map_torch = map_torch

        self.episodeScore = 0
        self.collision = False
        self.reach_goal = False

        # Reset robot position
        self.robot_pos_ini = self.generate_random_position()
        self.target_pos = self.generate_random_position()
        test1 = np.linalg.norm(self.robot_pos_ini - self.target_pos)
        while np.linalg.norm(self.robot_pos_ini - self.target_pos) < self.env_size * 0.1 \
                or np.linalg.norm(self.robot_pos_ini - self.target_pos) > self.env_size * 0.2:
            self.target_pos = self.generate_random_position()

        self.robot_orn_ini = np.random.randint(low=0, high=71, size=1, dtype='int')
        self.robot_orn_ini = self.robot_orn_ini * 5 * np.pi / 180
        self.robot_pos = self.robot_pos_ini

        return self.robot_pos_ini, self.robot_orn_ini[0], self.target_pos


    def apply_action(self, action):
        if action == 0:  # go ahead
            self.robot_pos[0] += int(self.forward_distance * np.sin(self.robot_orn))
            self.robot_pos[1] += int(self.forward_distance * np.cos(self.robot_orn))
        elif action == 1:  # turn left
            self.robot_orn += self.turn_angle
        elif action == 2:  # turn right
            self.robot_orn -= self.turn_angle
        else:
            raise ValueError('action is out of action_space, action =', action)

    def get_observations(self):
        return self.map, (self.robot_pos, self.robot_orn, self.target_pos)

    def get_reward(self, action):
        reward = 0
        if action == 0:
            reward += 1
        # Collision with edge of map
        if np.max(self.robot_pos) >= self.env_size - self.forward_distance - self.robot_size \
                or np.min(self.robot_pos) <= self.forward_distance + self.robot_size:
            reward = -600
            self.collision = True
        # Collision with black area in map
        elif np.min(self.map[self.robot_pos[1] - self.robot_size:self.robot_pos[1] + self.robot_size,
                    self.robot_pos[0] - self.robot_size:self.robot_pos[0] + self.robot_size]) <= 50:
            reward = -600
            self.collision = True
        # Reach goal
        else:
            if np.linalg.norm(self.robot_pos - self.target_pos) < 5:
                reward = 5000
                self.reach_goal = True
        return reward

    def is_done(self):
        if self.reach_goal or self.collision:
            return True
        else:
            return False

    def get_info(self):
        return False

    def step(self, action):
        self.apply_action(action)
        return (
            self.get_observations(),
            self.get_reward(action),
            self.is_done(),
            self.get_info(),
        )

    def solved(self):
        return False


def get_next_waypoint(scene, map_global_show, map_grid_show, pos_tmp):
    map_grid_3color = np.full((300, 300), 127)  # black, white, grey

    for xcnt in range(300):
        for ycnt in range(300):
            if map_global_show[xcnt, ycnt] > 147:
                map_grid_3color[xcnt, ycnt] = 255
            elif map_global_show[xcnt, ycnt] < 107:
                map_grid_3color[xcnt, ycnt] = 0
    map_grid_3color = np.array(map_grid_3color, dtype=np.uint8)

    map_explore_edge = np.full((300, 300), 0)
    map_explore_edge = np.array(map_explore_edge, dtype=np.uint8)
    explore_contours_all, _ = cv2.findContours(map_grid_show, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    explore_contours = []
    for contour_t in explore_contours_all:
        if cv2.contourArea(contour_t) > 45:
            explore_contours.append(contour_t)

    map_explore_edge_show = cv2.drawContours(map_explore_edge, explore_contours, -1, 255, 1)

    # cost of new point
    map_explore_cost = np.full((300, 300), -np.inf)

    for xcnt in range(300):
        for ycnt in range(300):
            if map_explore_edge_show[xcnt, ycnt] == 255:
                unexplored_pixel_cnt = 0
                if np.min(map_grid_3color[xcnt - 3:xcnt + 3,
                          ycnt - 3:ycnt + 3]) != 0 and 5 <= xcnt <= 294 and 5 <= ycnt <= 294:
                    for pixel_cnt in range(-5, 5):
                        if map_global_show[xcnt + pixel_cnt, ycnt - 5] == 127:
                            unexplored_pixel_cnt += 1
                        elif map_global_show[xcnt + pixel_cnt, ycnt + 5] == 127:
                            unexplored_pixel_cnt += 1
                        elif map_global_show[xcnt - 5, ycnt + pixel_cnt] == 127:
                            unexplored_pixel_cnt += 1
                        elif map_global_show[xcnt + 5, ycnt + pixel_cnt] == 127:
                            unexplored_pixel_cnt += 1
                if scene[0] == 'Beechwood_0_int':
                    map_explore_cost[xcnt, ycnt] = unexplored_pixel_cnt - \
                                                   np.linalg.norm(np.array(
                                                       [int((pos_tmp[0] * -107 + 502 + 500 - 190) * 300 / 2000),
                                                        int((pos_tmp[1] * 104 + 503 + 500) * 300 / 2000)]) -
                                                                  np.array([xcnt, ycnt])) / 10
                else:
                    map_explore_cost[xcnt, ycnt] = unexplored_pixel_cnt - \
                                                   np.linalg.norm(np.array([(x_map(pos_tmp[0]) - 500) * 300 / 1000,
                                                                            (y_map(pos_tmp[1]) - 500) * 300 / 1000]) -
                                                                  np.array([xcnt, ycnt])) / 10

    target_explore_pos = [int(np.argmax(map_explore_cost) / 300) - 1,
                          np.mod(np.argmax(map_explore_cost), 300) - 1]

    return target_explore_pos


def navigation_task(scene, device, pos_tmp, edge_padding, small_size, map_grid_show, depth_num,
                    target_explore_pos, nav_model):
    if scene[0] == 'Beechwood_0_int' or scene[0] == 'Rs_int':
        robot_pos_for_nav = np.array([0, 0])
        robot_pos_for_nav[1] = int((pos_tmp[
                                        0] * -107 + 502 + 500 - 190) * small_size / 2000)
        robot_pos_for_nav[0] = int((pos_tmp[1] * 104 + 503 + 500) * small_size / 2000)

        map_t_padding = np.zeros((128 + 2 * edge_padding, 128 + 2 * edge_padding))
        map_t_padding[edge_padding:-edge_padding, edge_padding:-edge_padding] = cv2.resize(map_grid_show, (128, 128))

        depth_input = generate_depth(_map=map_t_padding,
                                     robot_position=(
                                     robot_pos_for_nav[0] + edge_padding, robot_pos_for_nav[1] + edge_padding),
                                     num=depth_num)
        depth_input = torch.tensor(depth_input).to(device).float().unsqueeze(0)

        target_input = np.zeros((1, 2))
        target_input[0][0] = math.sqrt((robot_pos_for_nav[0] - target_explore_pos[0] * small_size / 300) * (
                    robot_pos_for_nav[0] - target_explore_pos[0] * small_size / 300)
                                       + (robot_pos_for_nav[1] - target_explore_pos[1] * small_size / 300) * (
                                                   robot_pos_for_nav[1] - target_explore_pos[1] * small_size / 300))
        target_input[0][1] = np.arctan2(target_explore_pos[0] * small_size / 300 - robot_pos_for_nav[0],
                                        target_explore_pos[1] * small_size / 300 - robot_pos_for_nav[1]) + np.pi
        if target_input[0][1] < 0:
            target_input[0][1] += 2 * np.pi
        target_input = torch.tensor(target_input).to(device).float().unsqueeze(0)

        nav_model_output = nav_model(depth_input, target_input)
        nav_model_output = nav_model_output.cpu().detach().numpy()
        nav_model_output = np.squeeze(nav_model_output, axis=0)
        nav_model_output = nav_model_output[0]

    else:
        # Use local Planner directly
        if scene[0] == 'Beechwood_0_int':
            nav_model_output = np.arctan2(target_explore_pos[0] - int((pos_tmp[1] * 104 + 503 + 500) * 300 / 2000),
                                          target_explore_pos[1] - int(
                                              (pos_tmp[0] * -107 + 502 + 500 - 190) * 300 / 2000)) + np.pi
        else:
            nav_model_output = np.arctan2(
                target_explore_pos[0] - (x_map(pos_tmp[0]) - 500) * 300 / 1000,
                target_explore_pos[1] - (y_map(pos_tmp[1]) - 500) * 300 / 1000 + np.pi)

        if nav_model_output < 0:
            nav_model_output += 2 * np.pi

    if nav_model_output <= np.pi:
        nav_model_output = - nav_model_output
    elif nav_model_output > np.pi:
        nav_model_output = 2 * np.pi - nav_model_output
    print("nav_model_output", nav_model_output * 180 / np.pi)


def apply_action(step, orn_tmp, nav_model_output):
    if step >= 30:
        if nav_model_output-0.17 < orn_tmp < nav_model_output+0.17:
            print("forward")
            action = np.array([0.7, 0.0])
        elif orn_tmp < nav_model_output:
            print('turn left')
            action = np.array([0.0, -0.2])
        elif orn_tmp >= nav_model_output:
            print('turn right')
            action = np.array([0.0, 0.2])
