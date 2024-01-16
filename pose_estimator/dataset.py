import numpy as np
import torch as t
from torchvision import transforms
from random import shuffle
import os
from random import sample
'''
transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ])
'''
def load_dataset_(root, type, scene_id_list, batch_size, max_num=8000, device="cpu"):
    root = os.path.join(root, type)
    file = []
    max_n = int(max_num/batch_size)*batch_size
    num=0
    num_list=[]
    for id in scene_id_list:
        for episode in range(5):
            for step in range(0, 1000):
                file.append([os.path.join(root, id, "rgb", "%03i_%04i.npy" % (episode, step)),
                             os.path.join(root, id, "depth", "%03i_%04i.npy" % (episode, step)),
                             os.path.join(root, id, "pose", "%03i_%04i.npy" % (episode, step)),
                             ])
                if not num % 1000 == 0:
                    num_list.append(num)
                num += 1

    num_list = sample(num_list, max_n)
    num_list_minus_1 = (np.array(num_list)-1).tolist()
    num_list_sum = list(set(num_list) | set(num_list_minus_1))

    rgbd = np.array([None]*1000*100)
    pose_l = np.array([None]*1000*100)

    for i in num_list_sum:
        rgb, depth, pose = file[i]
        rgb = t.tensor(np.fromfile(rgb, dtype=np.float32).reshape(3, 144, 192), dtype=t.float32)
        depth = t.tensor(np.fromfile(depth, dtype=np.float32).reshape(1, 144, 192),
                         dtype=t.float32)
        if device == "cpu":
            rgbd[i] = t.cat([rgb, depth], dim=0)
            pose_l[i] = t.tensor(np.fromfile(pose, dtype=np.float32), dtype=t.float32)

        else:
            rgbd[i] = t.cat([rgb, depth], dim=0).cuda()
            pose_l[i] = t.tensor(np.fromfile(pose, dtype=np.float32), dtype=t.float32).cuda()


    return rgbd, pose_l, np.array(num_list).reshape(-1, batch_size)


def load_dataset_2(root, type, scene_id_list, batch_size, max_num=8000, device="cpu"):
    root = os.path.join(root, type)
    file = []
    max_n = int(max_num/batch_size)*batch_size
    num=0
    num_list=[]
    for id in scene_id_list:
        for episode in range(5):
            for step in range(0, 1000):
                file.append([os.path.join(root, id, "rgb", "%03i_%04i.npy" % (episode, step)),
                             os.path.join(root, id, "depth", "%03i_%04i.npy" % (episode, step)),
                             os.path.join(root, id, "pose", "%03i_%04i.npy" % (episode, step)),
                             ])
                if not num % 1000 == 0:
                    num_list.append(num)
                num += 1

    num_list = sample(num_list, max_n)
    num_list_minus_1 = (np.array(num_list)-1).tolist()
    num_list_sum = list(set(num_list) | set(num_list_minus_1))

    rgbd = np.array([None]*1000*100)
    pose_l = np.array([None]*1000*100)

    for i in num_list_sum:
        rgb, depth, pose = file[i]
        rgb = t.tensor(np.fromfile(rgb, dtype=np.float32).reshape(3, 144, 192), dtype=t.float32)
        depth = t.tensor(np.fromfile(depth, dtype=np.float32).reshape(1, 144, 192),
                         dtype=t.float32)
        if device == "cpu":
            rgbd[i] = t.cat([rgb, depth], dim=0)
            pose_l[i] = t.tensor(np.fromfile(pose, dtype=np.float32), dtype=t.float32)

        else:
            rgbd[i] = t.cat([rgb, depth], dim=0).cuda()
            pose_l[i] = t.tensor(np.fromfile(pose, dtype=np.float32), dtype=t.float32).cuda()


    return rgbd, pose_l, np.array(num_list).reshape(-1, batch_size)



def open_batch(root,batch_size,max_num=10000,device="cpu"):
    data=[]
    path = os.path.join(root,"rgb")
    for i,j,k in os.walk(path):
        rgb_l=[]
        depth_l=[]
        pose_l=[]
        count=0
        shuffle(k)
        for id in k:
            try:
                episode = int(id[:3])
                step=int(id[4:8])
                next = "%03i_%04i.npy"%(episode,step+1)
                rgb1 = t.tensor(np.fromfile(os.path.join(path,id),dtype=np.float32).reshape(3,144,192),dtype=t.float32)
                rgb2 = t.tensor(np.fromfile(os.path.join(path,next),dtype=np.float32).reshape(3,144,192),dtype=t.float32)
                depth= t.tensor(np.fromfile(os.path.join(root,"depth",next),dtype=np.float32).reshape(1,144,192),dtype=t.float32)
                pose = t.tensor(np.fromfile(os.path.join(root, "pose", next),dtype=float), dtype=t.float32)
                pose[-1] = -pose[-1]
                rgb = t.cat([rgb1,rgb2],dim=0)
                rgb_l.append(rgb)
                depth_l.append(depth)
                pose_l.append(pose)
                count+=1
            except:
                pass
            if count%batch_size==0 and len(rgb_l) > 0:
                if device=="cpu":
                    data.append(
                        [t.stack(rgb_l,dim=0),
                         t.stack(depth_l,dim=0),
                         t.stack(pose_l,dim=0)
                        ]
                    )
                else:
                    print(len(rgb_l))
                    data.append(
                        [t.stack(rgb_l, dim=0).cuda(),
                         t.stack(depth_l, dim=0).cuda(),
                         t.stack(pose_l, dim=0).cuda()
                         ]
                    )
                rgb_l=[]
                depth_l=[]
                pose_l=[]
            if count >= max_num:
                return data
    return data
def open_batch_name(root,batch_size,max_num=10000):
    data = []
    path = os.path.join(root, "rgb")
    for i, j, k in os.walk(path):
        rgb_l = []
        depth_l = []
        pose_l = []
        count = 0
        shuffle(k)
        for id in k:
            episode = int(id[:3])
            step = int(id[4:8])
            next = "%03i_%04i.npy" % (episode, step + 1)
            if not os.path.exists(os.path.join(path, next)):
                continue
            else:
                rgb1 = os.path.join(path, id)
                rgb2 = os.path.join(path, next)
                depth = os.path.join(root, "depth", next)
                pose = os.path.join(root, "pose", next)
                rgb_l.append([rgb1,rgb2])
                depth_l.append(depth)
                pose_l.append(pose)
                count += 1
            if count % batch_size == 0 and len(rgb_l)>0:
                data.append(
                    [rgb_l,
                     depth_l,
                     pose_l
                     ]
                )
                rgb_l = []
                depth_l = []
                pose_l = []
            if count >= max_num:
                return data
    return data
def load_dataset(root,type,scene_id,batch_size,max_num=10000,device="cpu"):
    path = os.path.join(root,type,scene_id)
    return open_batch(path,batch_size,max_num=max_num,device=device)

def load_dataset_name(root,type,scene_id,batch_size,max_num=10000):
    path = os.path.join(root, type, scene_id)
    return open_batch_name(path, batch_size, max_num=max_num)

def load_data(data,device):
    rgb,depth,pose = data
    depth_b=[]
    rgb_b=[]
    pose_b=[]
    for file in depth:
        depth_b.append(t.tensor(np.fromfile(file,dtype=np.float32).reshape(1,144,192),dtype=t.float32))
    depth_out= t.stack(depth_b,dim=0).to(device)
    for file in pose:
        pose_b.append(t.tensor(np.fromfile(file,dtype=float), dtype=t.float32))
    pose_out= t.stack(pose_b,dim=0).to(device)

    for file in rgb:
        rgb1 = t.tensor(np.fromfile(file[0],dtype=np.float32).reshape(3,144,192),dtype=t.float32)
        rgb2 = t.tensor(np.fromfile(file[1], dtype=np.float32).reshape(3, 144, 192), dtype=t.float32)
        rgb_b.append(t.cat([rgb1,rgb2],dim=0))
    rgb_out = t.stack(rgb_b,dim=0).to(device)

    return rgb_out, depth_out,pose_out

def load_type_dataset(root,type,batch_size,max_num=10000,device="cpu"):
    path = os.path.join(root, type)
    data = []
    for id in os.listdir(path):
        path_id = os.path.join(path, id)
        data.extend(open_batch(path_id, batch_size,max_num=int(max_num/len(os.listdir(path))), device=device))
    shuffle(data)
    return data

def load_all_scene_dataset(root,batch_size,max_num=10000,device="cpu"):
    data=[]
    for type in ["random","control"]:
        data.extend(load_type_dataset(root, type, batch_size,max_num= int(max_num/2), device=device))
    shuffle(data)
    return data