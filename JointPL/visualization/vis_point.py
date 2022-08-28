from ..utils.experiments import load_experiment
from omegaconf import OmegaConf
from ..datasets.wireframe_line import Wireframe_line
import numpy as np
import cv2
from ..utils.tensor import batch_to_device
import torchvision.transforms as T
import torch
import os
from ..settings import EXPER_PATH


def draw_keypoints(img_l, top_uvz, color=(255, 0, 0), idx=0):
    """Draw keypoints on an image"""
    vis_xyd = top_uvz.permute(0, 2, 1)[idx].detach().cpu().clone().numpy()
    vis = img_l.copy()
    cnt = 0
    for pt in vis_xyd[:,:2].astype(np.int32):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis, (x,y), 2, color, -1)
    return vis


test_conf = {'name': 'joint_pl_model'}
model = load_experiment('joint_pl_pretrained_model', test_conf)
net = model.keypoint_net
net = net.cuda()

default_train_conf = {
        'val_batch_size': 1,
}
conf = OmegaConf.create(default_train_conf)
Dataset = Wireframe_line(conf)
data_loader = Dataset.get_data_loader(split='val', shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

outdir = os.path.join(EXPER_PATH, 'joint_pl_pretrained_model/vis_point')
os.makedirs(f"{outdir}", exist_ok=True)

for it, data in enumerate(data_loader):

     data = batch_to_device(data, device, non_blocking=True)

     image = data['image']
     B, _, H, W = image.shape
     input_img =image
     source_score, source_uv_pred, source_feat, _, _ = net(input_img)

     vis_ori = (input_img[0].permute(1, 2, 0).detach().cpu().clone().squeeze())
     vis_ori -= vis_ori.min()
     vis_ori /= vis_ori.max()
     vis_ori = (vis_ori * 255).numpy().astype(np.uint8)

     _, top_k = source_score.view(B, -1).topk(300, dim=1)
     vis_ori = draw_keypoints(vis_ori, source_uv_pred.view(B, 2, -1)[:, :, top_k[0].squeeze()], (0, 0, 255))
     transform = T.ToPILImage()
     img = transform(vis_ori)
     img.save(f'{outdir}/00%s.jpg' %(it))

