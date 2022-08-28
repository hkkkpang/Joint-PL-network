import argparse
import torch
import os

from torch.utils.data import DataLoader
from ..datasets.wireframe_evaluation import Wireframe_evaluation
from ..utils.experiments import load_experiment
from ..datasets.patches_dataset import PatchesDataset
from ..evaluation.evaluate import evaluate_keypoint_net
from ..models.keypoint_net import KeypointNet
from ..assets.superpoint.demo_superpoint import SuperPointFrontend


def load_SP_net(conf_thresh=0.0, cuda=torch.cuda.is_available(), nms_dist=4, nn_thresh=0.7):
    weights_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        './superpoint/superpoint_v1.pth')
    kp_net = SuperPointFrontend(
        weights_path, nms_dist=nms_dist, conf_thresh=conf_thresh,
        nn_thresh=nn_thresh, cuda=cuda)
    return kp_net


def main():
    parser = argparse.ArgumentParser(
        description='Script for KeyPointNet testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model", required=True, type=str, help="which model used to be evaluated")
    parser.add_argument("--dataset", required=True, type=str, help="evaluate on which dataset")

    args = parser.parse_args()

    if args.model == 'pretrained_kp2d':
        checkpoint = torch.load('/cluster/scratch/hapang/pretrained_models_KP2d/v4.ckpt')
        model_args = checkpoint['config']['model']['params']

        # Check model type
        if 'keypoint_net_type' in checkpoint['config']['model']['params']:
            net_type = checkpoint['config']['model']['params']
        else:
            net_type = KeypointNet  # default when no type is specified

        # Create and load keypoint net
        if net_type is KeypointNet:
            net = KeypointNet(use_color=model_args['use_color'],
                              do_upsample=model_args['do_upsample'],
                              do_cross=model_args['do_cross'])

        net.load_state_dict(checkpoint['state_dict'])
        net = net.cuda()
        net.eval()

    elif args.model == 'superpoint':
        net = load_SP_net()
    
    elif args.model == 'sift':
        net = None
    
    elif args.model == 'joint_model':
        test_conf = {'name': 'joint_pl_model'}
        model = load_experiment('joint_pl_pretrained_model', test_conf)
        net = model.keypoint_net
        net = net.cuda()

    eval_params = [{'res': (256, 256), 'top_k': 300, }]
    eval_params += [{'res': (512, 512), 'top_k': 1000, }]

    for params in eval_params:

        if args.model == 'pretrained_kp2d' or args.model == 'joint_model':
            color = True
        else:
            color = False

        if args.dataset == 'HPatches':
            hp_dataset = PatchesDataset(root_dir='/cluster/scratch/hapang/HPatches', use_color=color,
                                        output_shape=params['res'], type='a')
            data_loader = DataLoader(hp_dataset,
                                     batch_size=1,
                                     pin_memory=False,
                                     shuffle=False,
                                     num_workers=8,
                                     worker_init_fn=None,
                                     sampler=None)

        else:
            conf = {
                'test_batch_size': 1,
                'shape': params['res'],
                'use_color': color
            }
            wireframe_dataset = Wireframe_evaluation(conf)
            data_loader = wireframe_dataset.get_data_loader(split='test', shuffle=False)

        rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net(
            data_loader,
            net,
            model=args.model,
            output_shape=params['res'],
            top_k=params['top_k'],
            use_color_kp2d=True)

        print('Repeatability {0:.3f}'.format(rep))
        print('Localization Error {0:.3f}'.format(loc))
        print('Correctness d1 {:.3f}'.format(c1))
        print('Correctness d3 {:.3f}'.format(c3))
        print('Correctness d5 {:.3f}'.format(c5))
        print('MScore {:.3f}'.format(mscore))


if __name__ == '__main__':
    main()
