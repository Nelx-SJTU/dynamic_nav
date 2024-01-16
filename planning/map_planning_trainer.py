import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import cv2
from src.planning_utils import mapPlanningEnv, compute_orn, generate_depth
from planning_model import Planning_Model
from src.mapping_utils import map_cut
import HybridAstarPlanner.astar as astar
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

edge_padding = 150
local_map_size = 100
depth_num = 1080
batch_size = 1
epochNum = 1000
max_train_steps = 30
lr = 1e-5
env = mapPlanningEnv(robot_size=2, env_size=128)
load_model_name = "model_0214/283"
save_model_name = "model_0223"

mode = 'firsttrain'  # firsttrain/continuetrain/test
if mode == 'firsttrain':
    is_training = True
    load_model = False
elif mode == 'continuetrain':
    is_training = True
    load_model = True
else:  # test
    is_training = False
    load_model = True


if os.path.exists('./model/' + save_model_name):
    pass
else:
    os.mkdir('./model/' + save_model_name)

summary_path = "./summary/" + save_model_name
if os.path.exists(summary_path):
    pass
else:
    os.makedirs(summary_path)
writer = SummaryWriter(summary_path)


# [load model]
if load_model:
    model = Planning_Model(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load("./model/" + load_model_name + ".pt", map_location=device))
else:
    model = Planning_Model(in_channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss = nn.SmoothL1Loss()

print("prepared")


def train_step(map_t, map_t_padding, edge_padding):
    global optimizer
    # Set Random RobotPos and TargetPos
    robot_pos, robot_orno, target_pos = env.reset(map_t, map_t)
    flag = np.random.randint(0, 4)
    if flag == 0:
        robot_pos = np.random.randint(0, 4)

    robot_orn = robot_orno * 180 / np.pi
    print('robot_pos =', robot_pos)
    print('robot_orn =', robot_orn)
    print('target_pos =', target_pos)

    # Astar and compute the Astar rot
    obstacle_x, obstacle_y = astar.design_obstacles_cv2(obstacle_map=map_t, env_size=map_t.shape[0])  # 输出障碍物的坐标
    pathx, pathy = astar.astar_planning(sx=robot_pos[0], sy=robot_pos[1],
                                        gx=target_pos[0], gy=target_pos[1],
                                        ox=obstacle_x, oy=obstacle_y,
                                        reso=1.0,  # xy grid resolution
                                        rr=2.0)  # robot_radius

    map_Astar_path_padding = np.zeros((map_t.shape[0] + 2 * edge_padding, map_t.shape[1] + 2 * edge_padding))

    for i in range(len(pathx)):
        map_Astar_path_padding[int(pathx[i]) + edge_padding][int(pathy[i]) + edge_padding] = 255
    cv2.imwrite("output/map_Astar_path_padding.jpg", map_Astar_path_padding)

    map_middle_point = map_cut(_map=map_Astar_path_padding,
                               robot_position=(robot_pos[0] + edge_padding, robot_pos[1] + edge_padding),
                               robot_angle=robot_orn, width=local_map_size, depth=local_map_size)
    cv2.imwrite("output/map_middle_point.jpg", map_middle_point)

    r = 20
    rot = compute_orn(map_middle_point, r, local_map_size)

    cv2.imwrite("output/map_t_padding.jpg", map_t_padding)

    depth_input = generate_depth(_map=map_t_padding, robot_position=(robot_pos[0] + edge_padding, robot_pos[1] + edge_padding), num=depth_num)
    depth_input = torch.tensor(depth_input).to(device).float().unsqueeze(0)

    target_input = np.zeros((1, 2))
    target_input[0][0] = math.sqrt((robot_pos[0]-target_pos[0])*(robot_pos[0]-target_pos[0])
                                + (robot_pos[1]-target_pos[1])*(robot_pos[1]-target_pos[1]))
    target_input[0][1] = np.arctan2(target_pos[0]-robot_pos[0], target_pos[1]-robot_pos[1]) + np.pi - robot_orno
    if target_input[0][1] < 0:
        target_input[0][1] += 2 * np.pi
    target_input = torch.tensor(target_input).to(device).float().unsqueeze(0)

    optimizer.zero_grad()
    output = model(depth_input, target_input)
    output = output.squeeze(0)
    batch_l = loss(output, rot.to(device).type(torch.float32))

    batch_l.backward(retain_graph=True)
    optimizer.step()

    return batch_l


if is_training:
    map_t = "./images/img_128/map_17.png"
    map_t = cv2.imread(map_t, cv2.IMREAD_GRAYSCALE)
    map_t_padding = np.zeros((128 + 2 * edge_padding, 128 + 2 * edge_padding))
    map_t_padding[edge_padding:-edge_padding, edge_padding:-edge_padding] = map_t
    min_loss = 10.0
    for epoch in range(epochNum):
        print("epoch:", epoch)
        seg_loss = 0.0
        depth_loss = 0.0
        pos_loss = 0.0
        total_loss = 0.0

        for moment in range(max_train_steps):
            print("moment:", moment)
            batch_loss = train_step(map_t=map_t, map_t_padding=map_t_padding, edge_padding=edge_padding)
            # print(batch_loss)
            total_loss += batch_loss.item()

        if epoch % 20 == 0:
            torch.save(model.state_dict(), "./model/" + save_model_name + '/' + str(epoch) + '.pt')

        total_loss = total_loss / max_train_steps

        if total_loss < min_loss:
            torch.save(model.state_dict(), "./model/" + save_model_name + '/' + str(epoch) + '.pt')
            min_loss = total_loss
        writer.add_scalar(tag="total_loss", scalar_value=total_loss, global_step=epoch)
        writer.flush()
        print("Epoch %d complete" % (epoch))
        print("total loss is: ", total_loss)
