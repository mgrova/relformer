import random
from skimage.io import imread
from torch.utils.data import Dataset
import numpy as np
import transforms3d as t3d
import os
import glob
import itertools

class UnrealCustomRelPoseDataset(Dataset):
    def __init__(self, data_path, transform=None):

        self.img_path1, self.img_path2, self.poses1, self.poses2, self.rel_poses =\
            self.__read_data_and_generate_pairs(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.img_path1)

    def __getitem__(self, idx):
        img1 = imread(self.img_path1[idx])
        img2 = imread(self.img_path2[idx])
        pose1 = self.poses1[idx]
        pose2 = self.poses2[idx]
        rel_pose = self.rel_poses[idx]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # randomly flip images in an image pair
        if random.uniform(0, 1) > 0.5:
            img1, img2 = img2, img1
            pose1, pose2 = pose2, pose1
            rel_pose[:3] = -rel_pose[:3]
            rel_pose[3:] = [rel_pose[3], -rel_pose[4], -rel_pose[5], -rel_pose[6]]

        #todo add positive and negative

        return {'query': img1,
                'ref': img2,
                'query_pose': pose1,
                'ref_pose': pose2,
                'rel_pose':rel_pose}
    
    def grouper(self, iterable, n, fillvalue=None):
        "Collect data into fixed-length chunks or blocks"
        # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
        args = [iter(iterable)] * n
        return itertools.zip_longest(fillvalue=fillvalue, *args)

    def __read_data_and_generate_pairs(self, data_path):
        current_path = os.path.join(data_path, "train")
        scene_paths = glob.glob(os.path.join(current_path, "*"))

        image_pose_pairs = []
        for scene_path in scene_paths:
            if (not os.path.isdir(scene_path)):
                continue

            poses  = self.__read_poses_from_file(os.path.join(scene_path, 'poses.txt'))

            im_root = os.path.join(scene_path, 'rgb')
            frames = sorted([os.path.join(im_root, fname) for fname in os.listdir(im_root) if fname.endswith(".png")])

            for img, pose in zip(frames, poses):
                image_pose_pairs.append([img, pose])

        # Shuffle the elements randomly
        random.shuffle(image_pose_pairs)
        res = list(self.grouper(image_pose_pairs, 2))
        
        img_paths1 = []
        img_paths2 = []
        poses1 = []
        poses2 = []
        rel_poses = []
        for pair1,  pair2 in res:
            pair1_img  = pair1[0]
            pair1_pose = np.array(pair1[1][1:8])

            pair2_img  = pair2[0]
            pair2_pose = np.array(pair2[1][1:8])

            img_paths1.append(pair1_img)
            img_paths2.append(pair2_img)
            poses1.append(pair1_pose)
            poses2.append(pair2_pose)

            x_rel, q_rel = self.__compute_rel_pose(pair1_pose, pair2_pose)
            rel_pose = np.zeros(7)
            rel_pose[:3]  = x_rel
            rel_pose[3:] = q_rel
            rel_poses.append(rel_pose)

            # print(f"pair1_img: {pair1_img}")
            # print(f"pair2_img: {pair2_img}")
            # print(f"pair1_pose: {pair1_pose}")
            # print(f"pair2_pose: {pair2_pose}")
            # print(f"rel_pose: {rel_pose}")

        return img_paths1, img_paths2, poses1, poses2, rel_poses
    
    # format -> id x y z qw qx qy qz
    def __read_poses_from_file(self, filename):
        if not os.path.isfile(filename):
            raise Exception("File {} does not exist".format(filename))
        
        with open(filename, 'r') as file:
            lines = file.readlines()
            poses = []
            for line in lines:
                if line.startswith('#'):
                    continue
                else:
                    pose = line.split(' ')
                    pose = [float(el) for el in pose]
                    poses.append(pose)
            return poses

    def __compute_rel_pose(self, p1, p2):
        t1 = p1[:3]
        q1 = p1[3:]  #Expected qw, qx, qy, qz
        rot1 = t3d.quaternions.quat2mat(q1 / np.linalg.norm(q1))

        t2 = p2[:3]
        q2 = p2[3:]
        rot2 = t3d.quaternions.quat2mat(q2 / np.linalg.norm(q2))

        t_rel = t2 - t1
        rot_rel = np.dot(np.linalg.inv(rot1), rot2)
        q_rel = t3d.quaternions.mat2quat(rot_rel)
        return t_rel, q_rel