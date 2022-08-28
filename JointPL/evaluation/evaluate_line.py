import glob
import numpy as np
from tqdm import tqdm
from ..utils.line_parsing import line_parsing_from_npz
import argparse
from ..settings import EXPER_PATH, DATA_PATH
import os.path as osp


def ap(tp, fp):
    recall = tp
    precision = tp / np.maximum(tp + fp, 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    return np.sum((recall[i + 1] - recall[i]) * precision[i + 1])


def msTPFP_hit(line_pred, line_gt, threshold):
    diff = ((line_pred[:, None, :, None] - line_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )
    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)
    hit = np.zeros(len(line_gt), np.bool)
    tp = np.zeros(len(line_pred), np.float)
    fp = np.zeros(len(line_pred), np.float)
    for i in range(len(line_pred)):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp, hit


def line_center_score(path, GT, threshold=5):
    preds = sorted(glob.glob(path))
    gts = sorted(glob.glob(GT))

    n_gt = 0
    n_pt = 0
    tps, fps, scores = [], [], []

    for pred_name, gt_name in tqdm(zip(preds, gts)):
        with np.load(gt_name) as fgt:
            gt_line = fgt["lpos"][:, :, :2]

        line, score = line_parsing_from_npz(
            pred_name,
            delta=0.8, nlines=1000,
            s_nms=0, resolution=128
        )
        line = line * (128 / 128)

        n_gt += len(gt_line)
        n_pt += len(line)
        tp, fp, hit = msTPFP_hit(line, gt_line, threshold)
        tps.append(tp)
        fps.append(fp)
        scores.append(score)

    tps = np.concatenate(tps)
    fps = np.concatenate(fps)

    scores = np.concatenate(scores)
    index = np.argsort(-scores)
    lcnn_tp = np.cumsum(tps[index]) / n_gt
    lcnn_fp = np.cumsum(fps[index]) / n_gt

    return ap(lcnn_tp, lcnn_fp)


def sAP_s1(path, GT):
    sAP = [5, 10, 15]
    print(f"Working on {path}")
    print("sAP: ", sAP)
    return [100 * line_center_score(f"{path}/*.npz", GT, t) for t in sAP]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script for generating npz file of line detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model", required=True, type=str, help="which model used to be evaluated")
    parser.add_argument("--dataset", required=True, type=str, help="evaluate on which dataset")
    parser.add_argument("--file", required=True, type=str, help="best or last file to be evaluated")

    args = parser.parse_args()


    if args.dataset == "Wireframe":
        GT_huang = osp.join(DATA_PATH, 'wireframe1_datarota_3w/valid/*_label.npz')
    else:
        GT_huang = osp.join(DATA_PATH, 'york/valid/*_label.npz')

    GT = GT_huang

    if args.file == "best":
        if args.dataset == "Wireframe":
            score = sAP_s1(osp.join(EXPER_PATH, args.model, "npz_best"), GT)
        else:
            score = sAP_s1(osp.join(EXPER_PATH, args.model, "npz_york_best"), GT)
    else:
        if args.dataset == "Wireframe":
            score = sAP_s1(osp.join(EXPER_PATH, args.model, "npz_last"), GT)
        else:
            score = sAP_s1(osp.join(EXPER_PATH, args.model, "npz_york_last"), GT)
            
    print(score)
