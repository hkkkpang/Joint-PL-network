import torch
import torch.nn.functional as F

def normalize(coord, h, w):
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).float()
    coord_norm = (coord - c) / c
    return coord_norm


def denormalize(coord_norm, h, w):
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    coord = coord_norm * c + c
    return coord


def ind2coord(ind, width):
    ind = ind.unsqueeze(-1)
    x = ind % width
    y = ind // width
    coord = torch.cat((x, y), -1).float()
    return coord


def gen_grid(h_min, h_max, w_min, w_max, len_h, len_w):
    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w), torch.linspace(h_min, h_max, len_h)])
    grid = torch.stack((x, y), -1).transpose(0, 1).reshape(-1, 2).float().to(
        torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return grid


def sample_feat_by_coord(x, coord_n, norm=False):
    feat = F.grid_sample(x, coord_n.unsqueeze(2)).squeeze(-1)
    if norm:
        feat = F.normalize(feat)
    feat = feat.transpose(1, 2)
    return feat


def compute_prob(feat1, feat2):

    sim = feat1.bmm(feat2.transpose(1, 2))
    prob = F.softmax(sim, dim=-1)  # Bxmxn
    return prob


def get_1nn_coord(feat1, featmap2):

    batch_size, d, h, w = featmap2.shape
    feat2_flatten = featmap2.reshape(batch_size, d, h * w).transpose(1, 2)  # Bx(hw)xd

    sim = feat1.bmm(feat2_flatten.transpose(1, 2))
    ind2_1nn = torch.max(sim, dim=-1)[1]

    coord2 = ind2coord(ind2_1nn, w)
    coord2_n = normalize(coord2, h, w)
    return coord2_n


def get_expected_correspondence_locs(feat1, featmap2, with_std=False):

    B, d, h2, w2 = featmap2.size()
    grid_n = gen_grid(-1, 1, -1, 1, h2, w2)
    featmap2_flatten = featmap2.reshape(B, d, h2 * w2).transpose(1, 2)  # BX(hw)xd
    prob = compute_prob(feat1, featmap2_flatten)  # Bxnx(hw)

    grid_n = grid_n.unsqueeze(0).unsqueeze(0)  # 1x1x(hw)x2
    expected_coord_n = torch.sum(grid_n * prob.unsqueeze(-1), dim=2)  # Bxnx2

    if with_std:
        # convert to normalized scale [-1, 1]
        var = torch.sum(grid_n ** 2 * prob.unsqueeze(-1), dim=2) - expected_coord_n ** 2  # Bxnx2
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # Bxn
        return expected_coord_n, std
    else:
        return expected_coord_n


def get_expected_correspondence_within_window(feat1, featmap2, coord2_n, with_std=False):

    batch_size, n_dim, h2, w2 = featmap2.shape
    n_pts = coord2_n.shape[1]
    grid_n = gen_grid(h_min=-0.125, h_max=0.125,
                               w_min=-0.125, w_max=0.125,
                               len_h=int(0.125*h2), len_w=int(0.125*w2))

    grid_n_ = grid_n.repeat(batch_size, 1, 1, 1)  # Bx1xhwx2
    coord2_n_grid = coord2_n.unsqueeze(-2) + grid_n_  # Bxnxhwx2
    feat2_win = F.grid_sample(featmap2, coord2_n_grid, padding_mode='zeros').permute(0, 2, 3, 1)  # Bxnxhwxd
  
    feat1 = feat1.unsqueeze(-2)

    prob = compute_prob(feat1.reshape(batch_size * n_pts, -1, n_dim),
                                 feat2_win.reshape(batch_size * n_pts, -1, n_dim)).reshape(batch_size, n_pts, -1)

    expected_coord2_n = torch.sum(coord2_n_grid * prob.unsqueeze(-1), dim=2)  # Bxnx2

    if with_std:
        var = torch.sum(coord2_n_grid ** 2 * prob.unsqueeze(-1), dim=2) - expected_coord2_n ** 2  # Bxnx2
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # Bxn
        return expected_coord2_n, std
    else:
        return expected_coord2_n


def set_weight(std, regularizer=0.0):

    inverse_std = 1. / torch.clamp(std+regularizer, min=1e-10)
    weight = inverse_std / torch.mean(inverse_std)
    weight = weight.detach()  # Bxn

    return weight
