"""
Entry point training and testing iAPR
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from datasets.RelPoseDataset import RelPoseDataset
from datasets.KNNCameraPoseDataset import KNNCameraPoseDataset
from models.pose_losses import CameraPoseLoss
from os.path import join
from models.relformer.RelFormer import RelFormer, RelFormer2, BrRwlFormer
from models.DeltaNet import DeltaNet, BaselineRPR, DeltaNetEquiv, TDeltaNet, MSDeltaNet
import sys
import torch
import pandas as pd
import torchvision
from PIL import Image
from skimage.io import imread
import transforms3d as t3d

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def get_knn_indices(query, db):
    distances = torch.linalg.norm(db-query, axis=1)
    return torch.argsort(distances)


def convert_to_quat(rot_repr_type, est_rel_poses):
    if rot_repr_type != 'q' and rot_repr_type != '10d':
        if rot_repr_type == '6d':
            rot_repr = est_rel_poses[:, 3:]
            rot_repr = utils.compute_rotation_matrix_from_ortho6d(rot_repr)
        elif rot_repr_type == '9d':
            # apply SVD orthogonalization to get the rotation matrix
            rot_repr = utils.symmetric_orthogonalization(est_rel_poses[:, 3:])
        quaternions = utils.compute_quaternions_from_rotation_matrices(rot_repr)
        est_rel_poses = torch.cat((est_rel_poses[:, :3], quaternions), dim=1)
    return est_rel_poses

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def read_extrinsic_mat_from_file(file):
    # Read the text file
    with open(file, 'r') as file:
        lines = file.readlines()

    # Extract the values from each line and convert them to float
    values = [list(map(float, line.strip().split())) for line in lines]

    # Convert the values to a NumPy array
    array = np.array(values)

    # Extract the rotation matrix (3x3)
    rotation_matrix = array[:3, :3]

    # Extract the translation vector (1x3)
    translation_vector = array[:3, 3]

    # Convert the rotation matrix to a quaternion
    qw = np.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2
    qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * qw)
    qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * qw)
    qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * qw)

    # Create the array in the desired format (x, y, z, qw, qx, qy, qz)
    result_array = np.array([translation_vector[0], translation_vector[1], translation_vector[2], qw, qx, qy, qz])

    return result_array

def compute_rel_pose(pose_ref, pose_query):
    t1 = pose_ref[:3]
    q1 = pose_ref[3:]
    rot1 = t3d.quaternions.quat2mat(q1 / np.linalg.norm(q1))

    t2 = pose_query[:3]
    q2 = pose_query[3:]
    rot2 = t3d.quaternions.quat2mat(q2 / np.linalg.norm(q2))

    t_rel = t2 - t1
    rot_rel = np.dot(np.linalg.inv(rot1), rot2)
    q_rel = t3d.quaternions.mat2quat(rot_rel)
    rel_pose = np.zeros((1, 7))
    rel_pose[i, :3]  = t_rel
    rel_pose[i, 3:] = q_rel
    return rel_pose

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", help="train or eval", default='train')
    arg_parser.add_argument("--dataset_path", help="path to the physical location of the dataset", default="/nfstemp/Datasets/7Scenes/")
    arg_parser.add_argument("--rpr_backbone_path", help="path to the backbone path", default="models/backbones/efficient-net-b0.pth")
    arg_parser.add_argument("--labels_file", help="pairs file", default="datasets/7Scenes/7scenes_training_pairs.csv")
    arg_parser.add_argument("--test_labels_file", help="pairs file", default="datasets/7Scenes_test/NN_7scenes_fire.csv")
    arg_parser.add_argument("--config_file", help="path to configuration file", default="config/7scenes_config_deltanet_transformer_encoder_6d.json")
    arg_parser.add_argument("--checkpoint_path", help="path to a pre-trained RPR model")
    arg_parser.add_argument("--test_dataset_id", default="7scenes", help="test set id for testing on all scenes, options: 7scene OR cambridge")
    arg_parser.add_argument("--knn_len", help="knn_len", type=int, default="1")
    arg_parser.add_argument("--is_knn", help="is_knn", type=int, default="0")
    arg_parser.add_argument("--gpu", help="gpu id", default="0")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {} experiment for RelFormer".format(args.mode))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))
    logging.info("Running with command line params:\n{}".format(
        "{}".format(' '.join(sys.argv[:]))))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.fdeterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = 'cuda:' + args.gpu
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    rot_repr_type = config.get('rot_repr_type')
    if rot_repr_type is not None and rot_repr_type != "q":
        if rot_repr_type == '6d':
            config['rot_dim'] = 6
        elif rot_repr_type == '9d':
            config['rot_dim'] = 9
        elif rot_repr_type == '10d':
            config['rot_dim'] = 4 # we output quaternions
        else:
            raise NotImplementedError(rot_repr_type)
    else:
        config["rot_dim"] = 4
        config["rot_repr_type"] = 'q'
        rot_repr_type = 'q'

    arch = config.get("arch")
    is_multi_scale = False
    if arch == "relformer2":
        model = RelFormer2(config, args.rpr_backbone_path).to(device)
    elif arch == "relformer":
        model = RelFormer(config, args.rpr_backbone_path).to(device)
    elif arch == "b-relformer":
        model = BrRwlFormer(config, args.rpr_backbone_path).to(device)
    elif arch == "deltanet":
        model = DeltaNet(config, args.rpr_backbone_path).to(device)
        # support freeze
        estimate_position_with_prior = config.get("position_with_prior")
        estimate_rotation_with_prior = config.get("rotation_with_prior")
        freeze = False
        if estimate_rotation_with_prior:
            freeze = True
            # exclude rotation-related
            freeze_exclude_phrase = ["_q."] # freeze backbone and all position-related modules
        elif estimate_position_with_prior:
            freeze = True
            # exclude position-related
            freeze_exclude_phrase = ["_x."] # freeze backbone and all rotation-related modules
        if freeze:
            for name, parameter in model.named_parameters():
                freeze_param = True
                for phrase in freeze_exclude_phrase:
                    if phrase in name:
                        freeze_param = False
                        break
                if freeze_param:
                    parameter.requires_grad_(False)
                else:
                    print(name)

    elif arch == "baseline":
        model = BaselineRPR(args.rpr_backbone_path).to(device)
    elif arch == "deltanetequiv":
        model = DeltaNetEquiv(config).to(device)
    elif arch == "tdeltanet":
        model = TDeltaNet(config, args.rpr_backbone_path).to(device)
    elif arch == "msdeltanet":
        model = MSDeltaNet(config, args.rpr_backbone_path).to(device)
        is_multi_scale = True
        assert rot_repr_type == '6d' or rot_repr_type == 'q'
    else:
        raise NotImplementedError(arch)
    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id), strict=False)
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    eval_reductions = config.get("reduction_eval")
    train_reductions = config.get("reduction")
    if args.mode == 'train':
        print("Invalid mode")
        exit(0)

    else: # Test
        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        transform = utils.test_transforms.get('none')
        # transform = utils.test_transforms.get('baseline')
        if args.is_knn:
            test_dataset = KNNCameraPoseDataset(args.dataset_path, args.test_labels_file, args.refs_file, args.test_knn_file, transform, args.knn_len)
        else:
            test_dataset = RelPoseDataset(args.dataset_path, args.test_labels_file, transform)

        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(test_dataset, **loader_params)
        pose_stats = np.zeros((len(dataloader.dataset), 3))
        with torch.no_grad():
            i = 0
            minibatch = next(iter(dataloader))

            for k, v in minibatch.items():
                minibatch[k] = v.to(device)

            # ref_pose_array   = minibatch['ref_pose'].to(dtype=torch.float32).detach()
            # query_pose_array = minibatch['query_pose'].to(device).to(dtype=torch.float32).detach()
            # query_img = minibatch.get('query')
            # ref_img = minibatch.get('ref')

            # print("Ref position: {}".format(ref_pose_array[:, 0:3]))
            # print("Query position: {}".format(query_pose_array[:, 0:3]))

            ref_seq   = 1
            ref_index = 0
            ref_path  = args.dataset_path + "pumpkin/seq-" + str(ref_seq).zfill(2) + "/frame-" + str(ref_index).zfill(6)
            ref_img_own = transform(imread(ref_path + ".color.png")).to(device)
            ref_pose = read_extrinsic_mat_from_file(ref_path + ".pose.txt")

            query_seq   = 8
            query_index = 452
            query_path  = args.dataset_path + "pumpkin/seq-" + str(query_seq).zfill(2) + "/frame-" + str(query_index).zfill(6)
            query_img_own = transform(imread(query_path + ".color.png")).to(device)
            query_pose = read_extrinsic_mat_from_file(query_path + ".pose.txt")

            # Expand dimension (similar to unqueeze)
            query_img_own = query_img_own[None, :]
            ref_img_own   = ref_img_own[None, :]

            custom_batch = {'query': query_img_own,
                            'ref': ref_img_own}
            
            # Visualize images
            trans_to_pil = torchvision.transforms.ToPILImage(mode='RGB')
            out_query = trans_to_pil(custom_batch.get('query')[0])
            out_ref = trans_to_pil(custom_batch.get('ref')[0])
            get_concat_h(out_query, out_ref).show()
            
            # Forward pass to predict the initial pose guess
            t0 = time.time()
            res = model(custom_batch)
            est_rel_pose = res['rel_pose']
            torch.cuda.synchronize()
            tn = time.time()

            pred_rel_pose = convert_to_quat(rot_repr_type, est_rel_pose)
            gt_rel_pose = compute_rel_pose(ref_pose, query_pose)
            gt_rel_pose_tensor = torch.tensor(gt_rel_pose).to(device, dtype=torch.float32)
            
            posit_err, orient_err = utils.pose_err(est_rel_pose, gt_rel_pose_tensor)

            # pred_rel_position = pred_rel_pose[:, 0:3].detach().cpu().numpy()
            # print("Predicted relative position: {}".format(pred_rel_position))
            # pred_rel_q = pred_rel_pose[:, 3:].detach().cpu().numpy()
            # print("Predicted relative orientation: {}".format(pred_rel_q))

            # Collect statistics
            pose_stats[i, 0] = posit_err.item()
            pose_stats[i, 1] = orient_err.item()
            pose_stats[i, 2] = (tn - t0)*1000

            msg = "Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                pose_stats[i, 0],  pose_stats[i, 1],  pose_stats[i, 2])

            logging.info(msg)