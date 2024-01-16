import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
from torchvision import transforms
from PIL import Image
from mapping_model import Mapping_Model


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


imsize = (224, 224)
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


batch_size = 32
is_training = True
load_model = False
epochNum = 10
lr = 1e-5

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


if os.path.exists('./model'):
    pass
else:
    os.mkdir('./model')


load_model_name = "load_model_name"
save_model_name = "save_model_name"

# [load model] 
if load_model == True:
    model = Mapping_Model(in_channels=3).to(device)
    model.load_state_dict(torch.load("./model/"+load_model_name+".pt"))
else:
    model = Mapping_Model(in_channels=3).to(device)


summary_path = "./summary/"+load_model_name
if os.path.exists(summary_path):
    pass
else:
    os.makedirs(summary_path)
writer = SummaryWriter(summary_path)


optimizer = optim.Adam(model.parameters(), lr=lr)
loss = nn.L1Loss()
print("prepared")

batch_ptr = 0
a = np.zeros((320, 20, 15))


def train_step(t_DS_num, t_moment):
    global optimizer
    t_rgbd_data = image_loader("./YOUR_DATASET_PATH/dataset/DS_"+str(t_DS_num)+"/rgb/DS"+str(t_DS_num)+"_rgb_"+str(t_moment)+".jpg")
    t_seg_data = image_loader("./YOUR_DATASET_PATH/dataset/DS_"+str(t_DS_num)+"/seg_scallop/DS"+str(t_DS_num)+"_seg_"+str(t_moment)+".jpg")

    optimizer.zero_grad()
    output = model(t_rgbd_data)
    seg_pred = output[0][0]

    batch_l = loss(seg_pred, t_seg_data[0][0].to(device))

    # Write the smooth loss here
    smooth_loss = 0.0

    # Write the boundary loss here
    boundary_loss = 0.0

    batch_l = batch_l + smooth_loss + boundary_loss

    batch_l.backward(retain_graph=True)
    optimizer.step()

    return batch_l


DS = [1]
moment_per_DS = 400


if is_training == True:
    for epoch in range(epochNum):

        seg_loss=0.0
        depth_loss = 0.0
        pos_loss = 0.0
        total_loss = 0.0
        prevCode = torch.from_numpy(a).float().unsqueeze(0)

        for ds in DS:
            for moment in range(moment_per_DS):
                batch_loss = train_step(t_DS_num=ds, t_moment=moment)
                total_loss += batch_loss.item()

        torch.save(model.state_dict(), "YOUR_MODEL_PATH/model/" + save_model_name + '.pt')
        total_loss /= len(DS) * moment_per_DS

        writer.add_scalar(tag="total_loss", scalar_value=total_loss, global_step=epoch)
        writer.flush()

        print("Epoch %d complete" %(epoch))
        print("total loss is: ", total_loss)


if mode == 'test':
    test_pic_path = "YOUR_TEST_OUTPUT_PATH"
    if os.path.exists(test_pic_path):
        pass
    else:
        os.makedirs(test_pic_path)
    test_ds = 1
    moment_per_DS = 400
    for test_moment in range(moment_per_DS):
        test_img_path = "./YOUR_DATASET_PATH/dataset/DS_"+str(test_ds)+"/depth/DS"+str(test_ds)+"_d_"+str(test_moment)+".jpg"
        cognitive_map_output = model(image_loader(test_img_path))

        utils.save_image(cognitive_map_output[0][0], test_pic_path+"/DS"+str(test_ds)+"_output_"+str(test_moment)+".jpg", normalize=True)
