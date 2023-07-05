#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from message_filters import Subscriber, ApproximateTimeSynchronizer
import json

import sys
sys.path.append("/home/aiiacvmllab/Projects/metric_learning_repos/relformer")

from models.relformer.RelFormer import RelFormer, RelFormer2, BrRwlFormer
from models.DeltaNet import DeltaNet, BaselineRPR, DeltaNetEquiv, TDeltaNet, MSDeltaNet

import os
import torch
import numpy as np
import time
from util import utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class RelformerTransformer():
    def __init__(self, node_name):
        rospy.init_node(node_name, anonymous=False)

        config_file       = rospy.get_param('~config_file', default="../config/7scenes_config_relformer.json")
        gpu               = rospy.get_param('~gpu', default="0")
        checkpoint        = rospy.get_param('~checkpoint', default="~/Documents/checkpoints/relformer/relformer_checkpoints/relformer_DeltanetEnc_6d_all.pth")
        rpr_backbone_path = rospy.get_param('~rpr_backbone_path', default="../models/backbones/efficient-net-b0.pth")
        
        query_image_topic = rospy.get_param('~query_image_topic', default="query_image")
        ref_image_topic   = rospy.get_param('~ref_image_topic', default="ref_image")

        # TODO. Define output format

        # Read configuration
        with open(config_file, "r") as read_file:
            config = json.load(read_file)
        rospy.loginfo("Running with configuration:\n{}".format(
            '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

        # Set the seeds and the device
        use_cuda = torch.cuda.is_available()
        self.device_id = 'cpu'
        torch_seed = 0
        numpy_seed = 2
        torch.manual_seed(torch_seed)
        if use_cuda:
            torch.backends.cudnn.fdeterministic = True
            torch.backends.cudnn.benchmark = False
            self.device_id = 'cuda:' + str(gpu)
        np.random.seed(numpy_seed)
        self.device = torch.device(self.device_id)

        # Load and setup model
        self.model = None
        self.model = self.__setup_model(config, rpr_backbone_path, checkpoint)
        self.model.eval()
        rospy.loginfo("Succesfully loaded model")

        # Set the dataset and data loader
        self.transform = utils.test_transforms.get('none')
        # self.transform = utils.test_transforms.get('baseline')

        # Setup publishers and subscribers
        self.__bridge = CvBridge()
        self.__query_image_sub = Subscriber(query_image_topic, Image)
        self.__ref_image_sub   = Subscriber(ref_image_topic, Image)

        # Synchronize the image and posearray messages based on their timestamps
        self.__ts = ApproximateTimeSynchronizer([self.__query_image_sub, self.__ref_image_sub], queue_size=1, slop=0.1)
        self.__ts.registerCallback(self.__inference_cb)

        rospy.loginfo("Succesfully configured subs/pubs")
        
    def __inference_cb(self, query_img_msg, ref_img_msg):
        
        rospy.loginfo("Received new info. Starting inference ...")

        batch = self.__create_batch_from_images_msgs(query_img_msg, ref_img_msg)

        self.infer_from_batch(batch)

    def __create_batch_from_images_msgs(self, query_msg, target_msg):
        # Setup initial image
        cv_query_img  = self.__bridge.imgmsg_to_cv2(query_msg, desired_encoding="passthrough")
        cv_query_img  = self.transform(cv_query_img)
        cv_ref_img = self.__bridge.imgmsg_to_cv2(target_msg, desired_encoding="passthrough")
        cv_ref_img = self.transform(cv_ref_img)

        # Expand dimension (similar to unqueeze)
        cv_query_img = cv_query_img[None, :]
        cv_ref_img   = cv_ref_img[None, :]

        batch = {'query': cv_query_img,
                 'query_pose': None,
                 'ref': cv_ref_img,
                 'ref_pose': None,
                 'rel_pose': None}
        return batch

    def __setup_model(self, config, rpr_backbone_path, checkpoint):
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
            model = RelFormer2(config, rpr_backbone_path).to(self.device)
        elif arch == "relformer":
            model = RelFormer(config, rpr_backbone_path).to(self.device)
        elif arch == "b-relformer":
            model = BrRwlFormer(config, rpr_backbone_path).to(self.device)
        elif arch == "deltanet":
            model = DeltaNet(config, rpr_backbone_path).to(self.device)
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
            model = BaselineRPR(rpr_backbone_path).to(self.device)
        elif arch == "deltanetequiv":
            model = DeltaNetEquiv(config).to(self.device)
        elif arch == "tdeltanet":
            model = TDeltaNet(config, rpr_backbone_path).to(self.device)
        elif arch == "msdeltanet":
            model = MSDeltaNet(config, rpr_backbone_path).to(self.device)
            is_multi_scale = True
            assert rot_repr_type == '6d' or rot_repr_type == 'q'
        else:
            raise NotImplementedError(arch)
        
        model.load_state_dict(torch.load(checkpoint, map_location=self.device_id), strict=False)
        rospy.loginfo("Initializing from checkpoint: {}".format(checkpoint))

        return model

    def infer_from_batch(self, batch):
        with torch.no_grad():
            # Move tensors to GPU
            for k, v in batch.items():
                if v is not None:
                    batch[k] = torch.tensor(v).to(self.device)

            # Forward pass to predict the initial pose guess
            t0 = time.time()
            res = self.model(batch)
            est_rel_pose = res['rel_pose']
            torch.cuda.synchronize()
            tn = time.time()

            est_rel_position = est_rel_pose[0][0:3]
            est_rel_ori = est_rel_pose[0][3:]

            rospy.loginfo("Estimate relative position: ({:.3f}, {:.3f}, {:.3f}) and orientation: ({:.3f}, {:.3f}, {:.3f}, {:.3f}) inferred in {:.2f}[ms]"
                          .format(est_rel_position[0], est_rel_position[1], est_rel_position[2],
                                  est_rel_ori[0], est_rel_ori[1], est_rel_ori[2], est_rel_ori[3],
                                  (tn - t0)*1000))

if __name__ == '__main__':
    try:
        _ = RelformerTransformer('relformer_transformer_inference_node')
        rospy.spin()
    except rospy.ROSInterruptException:
        pass