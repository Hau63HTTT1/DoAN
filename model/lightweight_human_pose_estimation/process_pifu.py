import torch
import cv2
import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from model.lightweight_human_pose_estimation.models.with_mobilenet import PoseEstimationWithMobileNet
from model.lightweight_human_pose_estimation.modules.keypoints import extract_keypoints, group_keypoints
from model.lightweight_human_pose_estimation.modules.load_state import load_state
from model.lightweight_human_pose_estimation.modules.pose import Pose, track_poses
from model.lightweight_human_pose_estimation import demo
import shutil

def process_image(image_path, output_dir, net):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        

    image_name = os.path.basename(image_path)
    new_image_path = os.path.join(output_dir, image_name)
    shutil.copy2(image_path, new_image_path)
    

    rect_path = new_image_path.replace('.%s' % (image_name.split('.')[-1]), '_rect.txt')
    
    stride = 8
    upsample_ratio = 4
    height_size = 512
    num_keypoints = Pose.num_kpts
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    orig_img = img.copy()
    heatmaps, pafs, scale, pad = demo.infer_fast(net, img, height_size, stride, upsample_ratio, cpu=False)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

    rects = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        valid_keypoints = []
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                valid_keypoints.append([pose_keypoints[kpt_id, 0], pose_keypoints[kpt_id, 1]])
        valid_keypoints = np.array(valid_keypoints)

      
        if len(valid_keypoints) > 0:
            pmin = valid_keypoints.min(0)
            pmax = valid_keypoints.max(0)

          
            center = (0.5 * (pmax[:2] + pmin[:2])).astype(np.int32)

         
            height_extension_factor = 2.0

           
            width = pmax[0] - pmin[0]
            height = pmax[1] - pmin[1]
            radius = int(0.65 * max(width, height))

           
            has_nose = pose_entries[n][0] != -1.0
            has_eyes = pose_entries[n][14] != -1.0 or pose_entries[n][15] != -1.0
            has_ears = pose_entries[n][16] != -1.0 or pose_entries[n][17] != -1.0

            if has_nose or has_eyes or has_ears:
                head_keypoints = []
                for kpt_id in [0, 14, 15, 16, 17]:  
                    if pose_entries[n][kpt_id] != -1.0:
                        head_keypoints.append([pose_keypoints[kpt_id, 0], pose_keypoints[kpt_id, 1]])

                if head_keypoints:
                    head_points = np.array(head_keypoints)
                    head_top = head_points.min(0)[1]
                    head_extension = int((pmax[1] - head_top) * 1.2)
                    center[1] = center[1] - int(0.35 * radius)
                    radius = max(radius, int(1.4 * height))

            elif pose_entries[n][10] != -1.0 or pose_entries[n][13] != -1.0:
                shoulders_y = []
                if pose_entries[n][5] != -1.0:  
                    shoulders_y.append(pose_keypoints[5, 1])
                if pose_entries[n][2] != -1.0: 
                    shoulders_y.append(pose_keypoints[2, 1])

                if shoulders_y:
                    avg_shoulder_y = sum(shoulders_y) / len(shoulders_y)
                    head_height = height * 0.5
                    center[1] = int(avg_shoulder_y - head_height * 0.8)
                    radius = int(1.5 * max(width, height_extension_factor * height))
                else:
                    center = (0.5 * (pmax[:2] + pmin[:2])).astype(np.int32)
                    center[1] -= int(0.4 * height)
                    radius = int(1.0 * max(width, height_extension_factor * height))

            elif pose_entries[n][8] != -1.0 and pose_entries[n][11] != -1.0:
                center = (0.5 * (pose_keypoints[8] + pose_keypoints[11])).astype(np.int32)
                base_radius = np.sqrt(((center[None,:] - valid_keypoints)**2).sum(1)).max(0)
                radius = int(1.8 * base_radius)
                center[1] -= int(0.3 * radius)

            else:
                center = np.array([img.shape[1]//2, img.shape[0]//2], dtype=np.int32)
                radius = max(img.shape[1]//2, int(img.shape[0] * 0.6))

          
            x1 = max(0, center[0] - radius)
            extra_top_padding = int(0.2 * radius)
            y1 = max(0, center[1] - radius - extra_top_padding)
            width = min(2*radius, img.shape[1] - x1)
            height = min(2*radius + extra_top_padding, img.shape[0] - y1)

            rects.append([x1, y1, width, height])
        else:
            
            rects.append([0, 0, img.shape[1], img.shape[0]])

    np.savetxt(rect_path, np.array(rects), fmt='%d')
    return rect_path, new_image_path

def init_pose_model():
    net = PoseEstimationWithMobileNet()
    checkpoint_path = os.path.join(project_root, 'checkpoint_iter_370000.pth')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)
    return net.cuda()