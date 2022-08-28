#!/usr/bin/env python3
"""Train L-CNN
Usage:
    test.py [options] <yaml-config> <ckpt> <dataname> <datadir>
    test.py (-h | --help )

Arguments:
   <yaml-config>                   Path to the yaml hyper-parameter file
   <ckpt>                          Path to ckpt
   <dataname>                      Dataset name
   <datadir>                       Dataset dir

Options:
   -h --help                       Show this screen.
   -d --devices <devices>          Comma seperated GPU devices [default: 0]
   -i --identifier <identifier>    Folder identifier [default: default-lr]
"""

import os
import random
import numpy as np
import torch

from JointPL.datasets.wireframe_line import Wireframe_line
from JointPL.utils.experiments import load_experiment
from JointPL.utils.line_parsing import OneStageLineParsing


def main():

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")

    # 1. dataset

    conf = {
        'val_batch_size': 1,
    }
    wireframe_dataset = Wireframe_line(conf)
    val_loader = wireframe_dataset.get_data_loader(split='val')

    # 2. model
    test_conf = {'name': 'joint_pl_model'}
    model = load_experiment('joint_pl_pretrained_model', test_conf)
    net = model.keypoint_net

    net.eval()
    net.training = True
    index = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):

            image = data["image"]
            _, _, desc, outputs, feature = net(image)

            output = outputs[0]
            head_off = [2, 4, 5, 6]
            heatmap = {}
            heatmap["lcmap"] = output[:, 0:head_off[0]].softmax(1)[:, 1]
            heatmap["lcoff"] = output[:, head_off[0]: head_off[1]].sigmoid() - 0.5
            heatmap["lleng"] = output[:, head_off[1]: head_off[2]].sigmoid()
            heatmap["angle"] = output[:, head_off[2]: head_off[3]].sigmoid()

            parsing = True
            if parsing:
                lines, scores = [], []
                for k in range(output.shape[0]):
                    line, score = OneStageLineParsing.fclip_torch(
                        lcmap=heatmap["lcmap"][k],
                        lcoff=heatmap["lcoff"][k],
                        lleng=heatmap["lleng"][k],
                        angle=heatmap["angle"][k],
                        delta=0.8,
                        resolution=128
                    )
                    lines.append(line[None])
                    scores.append(score[None])

                heatmap["lines"] = torch.cat(lines)
                heatmap["score"] = torch.cat(scores)

            result = {'heatmaps': heatmap}

            H = result["heatmaps"]
            torch.save(desc, '../assets/line_match_data/desc{}.pt'.format(index))
            lines = H['lines'].squeeze()
            torch.save(4*lines, '../assets/line_match_data/line{}.pt'.format(index))

            index += 1


if __name__ == "__main__":
    main()