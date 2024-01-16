import cv2
import pandas as pd
import numpy as np
import os


def world_to_map(xy, trav_map_size, trav_map_resolution=0.1):
    return np.flip((np.array(xy) / trav_map_resolution + trav_map_size / 2.0)).astype(np.int)

def x_map(x):
    return -106.9711815*x+502.5497438 + 500 # +500 for map_size_global = 2000

def y_map(y):
    return 103.6528545*y+503.570678 + 500 # +500 for map_size_global = 2000

def map_cut(_map, robot_position, robot_angle, width, depth):
    x = int(robot_position[0] - width // 2)
    y = int(robot_position[1] - depth)
    w = int(width)
    d = int(depth)

    rows, cols, c = _map.shape
    _M = cv2.getRotationMatrix2D(robot_position, robot_angle, 1)
    _map = cv2.warpAffine(src=_map, M=_M, dsize=(rows, cols), borderValue=(255, 255, 255))  # M为上面的旋转矩阵

    roi = (x, y, w, d)
    if roi != (0, 0, 0, 0):
        crop = _map[y:y + d, x:x + w]
        return crop


scene = ["Beechwood_0_int"]

DS_num = 16
DS_len = 400  # number of images in folder DS
pos_data = pd.read_csv(f"./YOUR_DATASET_PATH/dataset/"+scene[0]+"/DS_"+str(DS_num)+"/position/DS"+str(DS_num)+"_pos.csv")
trav_map_1000 = cv2.imread('./YOUR_DATASET_PATH/trav_map/Beechwood_0_int/Beechwood_0_int_trav_mir.png')

print('trav_map_shape =', trav_map_1000.shape)

trav_map = np.full((3400,3400,3), 0, dtype=np.uint8)
trav_map[500:-500, 500:-500] = trav_map_1000
cv2.imwrite("./YOUR_ACCESSIBILITY_MAP_PATH/trav_map/Beechwood_0_int/Beechwood_0_int_trav_mir_2000.png", trav_map)

seg_trapezoid_path = "./YOUR_DATASET_PATH/dataset/"+scene[0]+"/DS_"+str(DS_num)+"/seg_trapezoid"
seg_scallop_path = "./YOUR_DATASET_PATH/dataset/"+scene[0]+"/DS_"+str(DS_num)+"/seg_scallop"
if os.path.exists(seg_trapezoid_path):
    pass
else:
    os.makedirs(seg_trapezoid_path)
if os.path.exists(seg_scallop_path):
    pass
else:
    os.makedirs(seg_scallop_path)
    os.mkdir(seg_scallop_path+"/seg_scallop_origin")


local_map_width = 320
local_map_depth = 320

output_mode = 'trapezoid'  # trapezoid, scallop

# parm in mode trapezoid
trapezoid_bottom_length = 0

# parm in mode scallop
scallop_width = 224


test = trav_map

for cnt in range(0, DS_len):
    pos_t = (x_map(pos_data['posx'][cnt]), y_map(pos_data['posy'][cnt]))

    orn_t = -pos_data['orn'][cnt]*180/np.pi-90
    map_t = map_cut(_map=trav_map, robot_position=pos_t, robot_angle=orn_t,
                    width=local_map_width, depth=local_map_depth)

    if output_mode == 'trapezoid':
        for h in range(local_map_depth):
            length = int(h*(local_map_width/2-trapezoid_bottom_length/2)//local_map_depth)  # 黑色部分长度
            for w in range(length):
                map_t[h][w] = [0, 0, 0]
                map_t[h][local_map_width-w-1] = [0, 0, 0]
        map_t = cv2.resize(map_t, (224, 224))

        img_path = "./YOUR_DATASET_PATH/dataset/"+scene[0]+"/DS_" + str(DS_num) + "/seg_trapezoid/DS" + str(DS_num) + "_seg_" + str(cnt) + ".jpg"
        cv2.imwrite(img_path, map_t)






