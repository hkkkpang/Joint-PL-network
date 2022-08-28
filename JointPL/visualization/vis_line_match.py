import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from JointPL.utils.line_matching import WunschLineMatcher
from JointPL.utils.line_match_util import plot_images, plot_lines, plot_color_line_matches

# Initialize the line matcher
config = {

    'line_matcher_cfg': {
        'cross_check': True,
        'num_samples': 5,
        'min_dist_pts': 8,
        'top_k_candidates': 10,
        'grid_size': 4
    }
}

# Read and pre-process the images
scale_factor = 1  # we recommend resizing the images to a resolution in the range 400~800 pixels
img1 = '../assets/line_match_data/00036228_0.png'
img1 = cv2.imread(img1, 0)
img1 = cv2.resize(img1, (img1.shape[1] // scale_factor, img1.shape[0] // scale_factor),
                  interpolation = cv2.INTER_AREA)
img1 = (img1 / 255.).astype(float)
torch_img1 = torch.tensor(img1, dtype=torch.float)[None, None]
img2 = '../assets/line_match_data/00036228_1.png'
img2 = cv2.imread(img2, 0)
img2 = cv2.resize(img2, (img2.shape[1] // scale_factor, img2.shape[0] // scale_factor),
                  interpolation = cv2.INTER_AREA)
img2 = (img2 / 255.).astype(float)
torch_img2 = torch.tensor(img2, dtype=torch.float)[None, None]

# Match the lines

line_seg1 = torch.load('../assets/line_match_data/line0.pt')
line_seg1 = np.float64(line_seg1.numpy())
line_seg1 = line_seg1[:150, :, :]

line_seg2 = torch.load('../assets/line_match_data/line1.pt')
line_seg2 = np.float64(line_seg2.numpy())
line_seg2 = line_seg2[:150, :, :]

desc1 = torch.load('../assets/line_match_data/desc0.pt')
desc2 = torch.load('../assets/line_match_data/desc1.pt')

# Match the lines in both images
matcher = WunschLineMatcher(**config["line_matcher_cfg"])
matches = matcher.forward(line_seg1, line_seg2, desc1, desc2)

outputs = {"line_segments": [line_seg1, line_seg2],
           "matches": matches}

line_seg1 = outputs["line_segments"][0]
line_seg2 = outputs["line_segments"][1]
matches = outputs["matches"]

valid_matches = matches != -1
match_indices = matches[valid_matches]
matched_lines1 = line_seg1[valid_matches][:, :, ::-1]
matched_lines2 = line_seg2[match_indices][:, :, ::-1]

plot_images([img1, img2], ['Image 1 - detected lines', 'Image 2 - detected lines'])
plot_lines([line_seg1[:, :, ::-1], line_seg2[:, :, ::-1]], ps=3, lw=2)
plot_images([img1, img2], ['Image 1 - matched lines', 'Image 2 - matched lines'])
plot_color_line_matches([matched_lines1, matched_lines2], lw=2)
plt.show()
