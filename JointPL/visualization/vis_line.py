import os
import random
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib as mpl
from ..utils.tensor import batch_to_device
from ..datasets.wireframe_line import Wireframe_line
from ..utils.experiments import load_experiment
from ..utils.line_parsing import OneStageLineParsing
from ..settings import EXPER_PATH

mpl.use("Agg")


_PLOT_nlines = 100
_PLOT = True
PLTOPTS = {"color": "#33FFFF", "s": 1.2, "edgecolors": "none", "zorder": 5}
cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.9, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def imshow(im):
    sizes = im.shape
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.xlim([-0.5, sizes[1] - 0.5])
    plt.ylim([sizes[0] - 0.5, -0.5])
    plt.imshow(im)


def c(x):
    return sm.to_rgba(x)


def main():

    batch_size = 1
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
        'train_batch_size': 1,
    }
    wireframe_dataset = Wireframe_line(conf)
    val_loader = wireframe_dataset.get_data_loader(split='val')

    # 2. model
    test_conf = {'name': 'joint_pl_model'}
    model = load_experiment('joint_pl_pretrained_model', test_conf)
    net = model.keypoint_net
    net = net.cuda()

    outdir = os.path.join(EXPER_PATH, 'joint_pl_pretrained_model')
    os.makedirs(f"{outdir}/viz_line", exist_ok=True)

    net.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):

            data = batch_to_device(data, device, non_blocking=True)
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

            for i in range(image.shape[0]):
                index = batch_idx * batch_size + i

                if _PLOT:
                    lines, score = H["lines"][i].cpu().numpy() * 4, H["score"][i].cpu().numpy()
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)

                    iname = val_loader.dataset._get_im_name(index)
                    im = io.imread(iname)
                    imshow(im)
                    for (a, b), s in zip(lines[:_PLOT_nlines], score[:_PLOT_nlines]):
                        plt.plot([a[1], b[1]], [a[0], b[0]], color="orange", linewidth=0.5, zorder=s)
                        plt.scatter(a[1], a[0], **PLTOPTS)
                        plt.scatter(b[1], b[0], **PLTOPTS)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())

                    plt.savefig(f"{outdir}/viz_line/{index:06}.pdf", bbox_inches="tight", pad_inches=0.0, dpi=3000)
                    plt.close()


if __name__ == "__main__":

    main()
