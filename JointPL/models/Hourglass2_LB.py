import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.image import image_grid


class BottleneckLine(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckLine, self).__init__()

        # self.ks = (M.line_kernel, 1)
        # self.padding = (int(M.line_kernel / 2), 0)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.conv2D = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        # self.conv2v = nn.Conv2d(planes, planes, kernel_size=(M.line_kernel, 1), padding=(int(M.line_kernel / 2), 0))
        # self.conv2h = nn.Conv2d(planes, planes, kernel_size=(1, M.line_kernel), padding=(0, int(M.line_kernel / 2)))
        self.mode = ["v", "h"]
        self.conv2, self.merge = self.build_line_layers(planes)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck2D.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def build_line_layers(self, planes):
        layer = []
        if "s" in self.mode:
            layer.append(nn.Conv2d(planes, planes, kernel_size=3, padding=1))

        if "v" in self.mode:
            layer.append(nn.Conv2d(planes, planes, kernel_size=(7, 1), padding=(int(7 / 2), 0)))

        if "h" in self.mode:
            layer.append(nn.Conv2d(planes, planes, kernel_size=(1, 7), padding=(0, int(7 / 2))))

        assert len(layer) > 0

        ll = len(self.mode)
        merge = nn.MaxPool3d((ll, 1, 1), stride=(ll, 1, 1))

        return nn.ModuleList(layer), merge

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)

        tt = torch.cat([torch.unsqueeze(conv(out), 2) for conv in self.conv2], dim=2)

        out = self.merge(tt)
        out = torch.squeeze(out, 2)

        # print(out.size())
        # exit()

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck2D.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class):
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)
        heads = []
        heads_size = [2, 2, 1, 1]
        heads_net = ["mask", "mask", "mask", "mask"]
        for k, (output_channels, net) in enumerate(zip(heads_size, heads_net)):
            if net == "raw":
                heads.append(
                    nn.Sequential(
                        nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(m, output_channels, kernel_size=1),
                    )
                )
                print(f"{k}-th head, head type {net}, head output {output_channels}")
            elif net == "mask":
                heads.append(
                    nn.Sequential(
                        nn.Conv2d(input_channels, 256, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, m, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(m, output_channels, kernel_size=1),
                    )
                )
                print(f"{k}-th head, head type {net}, head output {output_channels}")

        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


class HourglassNet2_LB(nn.Module):
    """Hourglass model from Newell et al ECCV 2016"""

    def __init__(self):
        super(HourglassNet2_LB, self).__init__()

        block2D = Bottleneck2D

        self.training = True

        self.inplanes = 64
        self.num_feats = self.inplanes * block2D.expansion
        self.num_stacks = 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block2D, self.inplanes, 1)
        self.layer2 = self._make_residual(block2D, self.inplanes, 1)
        self.layer3 = self._make_residual(block2D, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        head = lambda c_in, c_out: MultitaskHead(c_in, c_out)

        # build hourglass modules
        ch = self.num_feats * block2D.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        merge_fc = []
        for i in range(2):
            sub_hg, sub_res, sub_fc = [], [], []

            sub_hg.append(Hourglass(BottleneckLine, 1, self.num_feats, 4))
            sub_res.append(self._make_residual(BottleneckLine, self.num_feats, 1))
            sub_fc.append(self._make_fc(ch, ch))
            hg.append(nn.ModuleList(sub_hg))
            res.append(nn.ModuleList(sub_res))
            fc.append(nn.ModuleList(sub_fc))
            merge_fc.append(self._make_fc(int(ch*len(sub_fc)), ch))

            score.append(head(ch, 6))
            if i < 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1))
                score_.append(nn.Conv2d(6, ch, kernel_size=1))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.merge_fc = nn.ModuleList(merge_fc)

        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

        self.use_color = True
        self.with_drop = True
        self.do_cross = True
        self.do_upsample = True

        if self.do_cross is False:
            self.cross_ratio = 1.0

        self.cross_ratio = 2.0
        self.bn_momentum = 0.1

        self.conv_a = torch.nn.Sequential(torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          torch.nn.BatchNorm2d(256, momentum=0.1))
        self.conv_b = torch.nn.Sequential(torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          torch.nn.BatchNorm2d(256, momentum=0.1))

        if self.with_drop:
            self.dropout = torch.nn.Dropout2d(0.2)
        else:
            self.dropout = None

        self.cell = 8
        self.upsample = torch.nn.PixelShuffle(upscale_factor=2)

        self.conv_skip = torch.nn.Conv2d(256, 128, kernel_size=1)

        c4, c5, d1 = 256, 256, 512

        # Score Head.
        self.convDa = torch.nn.Sequential(torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1, bias=False),
                                          torch.nn.BatchNorm2d(c4, momentum=self.bn_momentum))
        self.convDb = torch.nn.Conv2d(c5, 1, kernel_size=3, stride=1, padding=1)

        # Location Head.
        self.convPa = torch.nn.Sequential(torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1, bias=False),
                                          torch.nn.BatchNorm2d(c4, momentum=self.bn_momentum))
        self.convPb = torch.nn.Conv2d(c5, 2, kernel_size=3, stride=1, padding=1)

        # Desc Head.
        self.convFa = torch.nn.Sequential(torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1, bias=False),
                                          torch.nn.BatchNorm2d(c4, momentum=self.bn_momentum))
        self.convFb = torch.nn.Sequential(torch.nn.Conv2d(c5, d1, kernel_size=3, stride=1, padding=1, bias=False),
                                          torch.nn.BatchNorm2d(d1, momentum=self.bn_momentum))
        self.convFaa = torch.nn.Sequential(torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1, bias=False),
                                           torch.nn.BatchNorm2d(c5, momentum=self.bn_momentum))
        self.convFbb = torch.nn.Conv2d(c5, 256, kernel_size=3, stride=1, padding=1)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        bn = nn.BatchNorm2d(outplanes)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):

        B, _, H, W = x.shape
        out = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            feat = []
            for j in range(len(self.hg[i])):
                y = self.hg[i][j](x)
                y = self.res[i][j](y)
                y = self.fc[i][j](y)
                feat.append(y)

            y = self.merge_fc[i](torch.cat(feat, dim=1))
            score = self.score[i](y)
            out.append(score)

            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        skip = self.conv_skip(y)
        x = self.maxpool(skip)
        x = self.relu(self.conv_a(x))
        x = self.relu(self.conv_b(x))

        if self.dropout:
            x = self.dropout(x)

        B, _, Hc, Wc = x.shape

        score_ = self.relu(self.convDa(x))
        if self.dropout:
            score_ = self.dropout(score_)
        score_ = self.convDb(score_).sigmoid()

        border_mask = torch.ones(B, Hc, Wc)
        border_mask[:, 0] = 0
        border_mask[:, Hc - 1] = 0
        border_mask[:, :, 0] = 0
        border_mask[:, :, Wc - 1] = 0
        border_mask = border_mask.unsqueeze(1)
        score_ = score_ * border_mask.to(score_.device)

        center_shift = self.relu(self.convPa(x))
        if self.dropout:
            center_shift = self.dropout(center_shift)
        center_shift = self.convPb(center_shift).tanh()

        step = (self.cell - 1) / 2.
        center_base = image_grid(B, Hc, Wc,
                                 dtype=center_shift.dtype,
                                 device=center_shift.device,
                                 ones=False, normalized=False).mul(self.cell) + step

        coord_un = center_base.add(center_shift.mul(self.cross_ratio * step))
        coord = coord_un.clone()
        coord[:, 0] = torch.clamp(coord_un[:, 0], min=0, max=W - 1)
        coord[:, 1] = torch.clamp(coord_un[:, 1], min=0, max=H - 1)

        feat = self.relu(self.convFa(x))
        if self.dropout:
            feat = self.dropout(feat)
        if self.do_upsample:
            feat = self.upsample(self.convFb(feat))
            feat = torch.cat([feat, skip], dim=1)
        feat = self.relu(self.convFaa(feat))
        feat = self.convFbb(feat)

        if self.training is False:
            coord_norm = coord[:, :2].clone()
            coord_norm[:, 0] = (coord_norm[:, 0] / (float(W - 1) / 2.)) - 1.
            coord_norm[:, 1] = (coord_norm[:, 1] / (float(H - 1) / 2.)) - 1.
            coord_norm = coord_norm.permute(0, 2, 3, 1)

            feat = torch.nn.functional.grid_sample(feat, coord_norm, align_corners=True)

            dn = torch.norm(feat, p=2, dim=1)  # Compute the norm.
            feat = feat.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        return score_, coord, feat, out[::-1], y