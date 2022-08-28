import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
import torch.nn.functional as F
from ..evaluation.descriptor_evaluation import (compute_homography,
                                                compute_matching_score)
from ..evaluation.detector_evaluation import compute_repeatability
from ..utils.image import to_color_normalized, to_gray_normalized


def evaluate_keypoint_net(data_loader, keypoint_net, model, output_shape=(320, 240), top_k=300, use_color_kp2d=True):
    """Keypoint net evaluation script.
    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        Dataset loader.
    keypoint_net: torch.nn.module
        Keypoint network.
    output_shape: tuple
        Original image shape.
    top_k: int
        Number of keypoints to use to compute metrics, selected based on probability.
    use_color: bool
        Use color or grayscale images.
    """
    if model == 'pretrained_kp2d' or model == 'joint_model':
        keypoint_net.eval()
        keypoint_net.training = False
    elif model == 'sift':
        sift = cv2.SIFT_create(nfeatures=top_k)

    conf_threshold = 0.0
    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []

    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="evaluate_keypoint_net"):
            if model == 'pretrained_kp2d':
                if use_color_kp2d:
                    image = to_color_normalized(sample['image'].cuda())
                    warped_image = to_color_normalized(sample['warped_image'].cuda())
                else:
                    image = to_gray_normalized(sample['image'].cuda())
                    warped_image = to_gray_normalized(sample['warped_image'].cuda())

                score_1, coord_1, desc1 = keypoint_net(image)
                score_2, coord_2, desc2 = keypoint_net(warped_image)
                B, C, Hc, Wc = desc1.shape

                # Scores & Descriptors
                score_1 = torch.cat([coord_1, score_1], dim=1).view(3, -1).t().cpu().numpy()
                score_2 = torch.cat([coord_2, score_2], dim=1).view(3, -1).t().cpu().numpy()
                desc1 = desc1.view(C, Hc, Wc).view(C, -1).t().cpu().numpy()
                desc2 = desc2.view(C, Hc, Wc).view(C, -1).t().cpu().numpy()

            elif model == 'joint_model':

                image = sample['image'].cuda()
                warped_image = sample['warped_image'].cuda()

                score_1, coord_1, desc1, _, _ = keypoint_net(image)
                score_2, coord_2, desc2, _, _ = keypoint_net(warped_image)
                B, C, Hc, Wc = desc1.shape

                # Scores & Descriptors
                score_1 = torch.cat([coord_1, score_1], dim=1).view(3, -1).t().cpu().numpy()
                score_2 = torch.cat([coord_2, score_2], dim=1).view(3, -1).t().cpu().numpy()
                desc1 = desc1.view(C, Hc, Wc).view(C, -1).t().cpu().numpy()
                desc2 = desc2.view(C, Hc, Wc).view(C, -1).t().cpu().numpy()

            elif model == 'superpoint':
                image = sample['image'].squeeze().numpy()
                warped_image = sample['warped_image'].squeeze().numpy()

                keypoints1, desc1, _ = keypoint_net.run(image)
                keypoints2, desc2, _ = keypoint_net.run(warped_image)

                score_1 = keypoints1.transpose()
                score_2 = keypoints2.transpose()
                desc1 = desc1.transpose()
                desc2 = desc2.transpose()
            elif model == 'sift':
                image = sample['image'].squeeze().numpy()
                warped_image = sample['warped_image'].squeeze().numpy()

                image8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                keypoints1, desc1 = sift.detectAndCompute(image8bit, None)
                keypoints1 = [[k.pt[0], k.pt[1], k.response] for k in keypoints1]
                score_1 = np.array(keypoints1)

                warp_image8bit = cv2.normalize(warped_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                keypoints2, desc2 = sift.detectAndCompute(warp_image8bit, None)
                keypoints2 = [[k.pt[0], k.pt[1], k.response] for k in keypoints2]
                score_2 = np.array(keypoints2)
            
            # Filter based on confidence threshold
            desc1 = desc1[score_1[:, 2] > conf_threshold, :]
            desc2 = desc2[score_2[:, 2] > conf_threshold, :]
            score_1 = score_1[score_1[:, 2] > conf_threshold, :]
            score_2 = score_2[score_2[:, 2] > conf_threshold, :]

            # Prepare data for eval
            data = {'image': sample['image'].numpy().squeeze(),
                    'image_shape': output_shape,
                    'warped_image': sample['warped_image'].numpy().squeeze(),
                    'homography': sample['homography'].squeeze().numpy(),
                    'prob': score_1,
                    'warped_prob': score_2,
                    'desc': desc1,
                    'warped_desc': desc2}

            # Compute repeatabilty and localization error
            _, _, rep, loc_err = compute_repeatability(data, keep_k_points=top_k, distance_thresh=3)
            repeatability.append(rep)
            localization_err.append(loc_err)

            # Compute correctness
            c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
            correctness1.append(c1)
            correctness3.append(c2)
            correctness5.append(c3)

            # Compute matching score
            mscore = compute_matching_score(data, keep_k_points=top_k)
            MScore.append(mscore)

    return np.mean(repeatability), np.mean(localization_err), \
           np.mean(correctness1), np.mean(correctness3), np.mean(correctness5), np.mean(MScore)


def sample_feat_by_coord(x, coord_n, norm=False):
    '''
    sample from normalized coordinates
    :param x: feature map [batch_size, n_dim, h, w]
    :param coord_n: normalized coordinates, [batch_size, n_pts, 2]
    :param norm: if l2 normalize features
    :return: the extracted features, [batch_size, n_pts, n_dim]
    '''
    feat = F.grid_sample(x, coord_n.unsqueeze(2)).squeeze(-1)
    if norm:
        feat = F.normalize(feat)
    feat = feat.transpose(1, 2)
    return feat
