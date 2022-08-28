import os.path as osp
import numpy as np
import torch
from ..utils.experiments import load_experiment, load_last_experiment
import os
from ..datasets.wireframe_line import Wireframe_line
from ..datasets.york import york
from ..utils.tensor import batch_to_device
from collections import OrderedDict
from ..models.line_loss import lcmap_head, lcoff_head, lleng_head, angle_head
from ..settings import EXPER_PATH
import argparse

parser = argparse.ArgumentParser(description='Script for generating npz file of line detection',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model", required=True, type=str, help="which model used to be evaluated")
parser.add_argument("--dataset", required=True, type=str, help="evaluate on which dataset")
parser.add_argument("--file", required=True, type=str, help="best or last file to be evaluated")

args = parser.parse_args()

eval_batch_size = 4

test_conf = {'name': 'joint_pl_model'}
if args.file == "best":
    model = load_experiment(args.model, test_conf)
else:
    model = load_last_experiment(args.model, test_conf)
net = model.keypoint_net
net = net.cuda()

net.eval()
net.training = False

if args.file == "best":
    if args.dataset =="Wireframe":
        npz = osp.join(EXPER_PATH, args.model, "npz_best")
    else:
        npz = osp.join(EXPER_PATH, args.model, "npz_york_best")
else:
    if args.dataset =="Wireframe":
        npz = osp.join(EXPER_PATH, args.model, "npz_last")
    else:
        npz = osp.join(EXPER_PATH, args.model, "npz_york_last")
osp.exists(npz) or os.makedirs(npz)

conf = {
    'val_batch_size': 4,
}
if args.dataset =="Wireframe":
    dataset = Wireframe_line(conf)
else:
    dataset = york(conf)
val_loader = dataset.get_data_loader(split='val')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with torch.no_grad():
    for batch_idx, data in enumerate(val_loader):
        
        data = batch_to_device(data, device, non_blocking=True)
        image = data["image"]
        _, _, _, outputs, feature = net(image)

        result = {"feature": feature}
        batch, channel, row, col = outputs[0].shape
        T = data["target"].copy()
        T["lcoff"] = T["lcoff"].permute(1, 0, 2, 3)

        for stack, output in enumerate(outputs):

            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()

            L = OrderedDict()
            Acc = OrderedDict()
            heatmap = {}
            lcmap, L["lcmap"] = lcmap_head(output, T["lcmap"])
            lcoff, L["lcoff"] = lcoff_head(output, T["lcoff"], mask=T["lcmap"])
            heatmap["lcmap"] = lcmap
            heatmap["lcoff"] = lcoff

            lleng, L["lleng"] = lleng_head(output, T["lleng"], mask=T["lcmap"])
            angle, L["angle"] = angle_head(output, T["angle"], mask=T["lcmap"])
            heatmap["lleng"] = lleng
            heatmap["angle"] = angle

            if stack == 0:
                result["heatmaps"] = heatmap

        H = result["heatmaps"]
        for i in range(image.shape[0]):
            index = batch_idx * eval_batch_size + i

            npz_dict = {}
            for k, v in H.items():
                if v is not None:
                    npz_dict[k] = v[i].cpu().numpy()
            np.savez(
                f"{npz}/{index:06}.npz",
                **npz_dict,
            )

