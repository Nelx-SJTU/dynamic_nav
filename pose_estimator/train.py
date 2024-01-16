from model import *
from dataset import *
from writer import *
from util import *
from torch import optim
from random import choice, sample
# from tensorboardX import SummaryWriter
from datetime import datetime
from util import adjust_learning_rate,smooth_loss, ssim_loss


batch_size = 256
initial_learning_rate = 5e-3
num_epoches = 100000
save_model_step= 500
lr_decay_step=50000
device = t.device("cuda" if t.cuda.is_available() else "cpu")


summary_path="summary_90"
model_save_path="model_90"
date=str(datetime.today())[:10]
model_name = "model_A_"
if not os.path.exists(summary_path):
    os.makedirs(summary_path)
if not os.path.exists(os.path.join(summary_path,date)):
    os.makedirs(os.path.join(summary_path,date))
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
if not os.path.exists(os.path.join(model_save_path, date)):
    os.makedirs(os.path.join(model_save_path, date))


net = PoseNet(True)
# net.load_state_dict(t.load("model_90/2022-06-26/model_A_9500.pkl"))
net.to(device)
root =[]
for i in os.listdir("../ig_dataset/"):
    root.append(i)
print(root, len(root))
rgbd_l, pose_l, dataset = load_dataset_("/tmp/igibson_nav_dataset", "control_90_", root, batch_size, 20000, device)
l1_loss = nn.L1Loss()

optimizer = optim.Adam(net.parameters(), lr=initial_learning_rate)
writer = SummaryWriter(summary_path+"/"+date)


net.train()
print("train start")
for episode in range(num_epoches):
    depth_l1_loss_avg=pos_l1_loss_avg=orn_l1_loss_avg=depth_smooth_loss_avg=depth_ssim_loss_avg=loss_avg=0
    learning_rate = adjust_learning_rate(initial_learning_rate, lr_decay_step, episode, optimizer)
    length = len(dataset)
    for data in dataset:
        rgbd1, rgbd2, pose = t.stack(rgbd_l[data-1].tolist(), dim=0), t.stack(rgbd_l[data].tolist(), dim=0), t.stack(pose_l[data].tolist(), dim=0)
        out_depth, out_pos, out_theta = net(rgbd1, rgbd2)
        ''' Loss '''
        out_pose = t.cat([out_pos, out_theta], dim=1)
        depth_l1_loss = l1_loss(out_depth, rgbd2[:, -1:])
        depth_smooth_loss = smooth_loss(out_depth)
        depth_ssim_loss = ssim_loss(out_depth, rgbd2[:, -1:])
        pos_l1_loss = l1_loss(out_pos,pose[:,0:2])
        orn_l1_loss=l1_loss(out_theta,pose[:,2:])
        loss = depth_l1_loss*0.85+pos_l1_loss*0.85+orn_l1_loss*0.85 + 0.15*depth_smooth_loss+0.25*depth_ssim_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.6)
        optimizer.step()

        depth_l1_loss_avg+=depth_l1_loss.cpu().item()
        pos_l1_loss_avg += pos_l1_loss.cpu().item()
        orn_l1_loss_avg += orn_l1_loss.cpu().item()
        depth_smooth_loss_avg += depth_smooth_loss.cpu().item()
        depth_ssim_loss_avg += depth_ssim_loss.cpu().item()
        loss_avg += loss.cpu().item()
    stats = state(depth_l1_loss_avg/length, pos_l1_loss_avg/length,orn_l1_loss_avg/length,
                  depth_smooth_loss_avg/length, depth_ssim_loss_avg/length,
                  loss_avg/length, learning_rate)
    writeSummary(writer, stats, episode)
    if episode>0 and episode%save_model_step==0:
        t.save(net.state_dict(), os.path.join(model_save_path, date)+"/"+model_name+str(episode)+".pkl")

t.save(net.state_dict(), os.path.join(model_save_path, date)+"/"+model_name+str(episode)+".pkl")
