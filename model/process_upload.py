import os
import torch
import cv2
import numpy as np
import shutil
from .lightweight_human_pose_estimation.models.with_mobilenet import PoseEstimationWithMobileNet
from .lightweight_human_pose_estimation.modules.keypoints import extract_keypoints, group_keypoints
from .lightweight_human_pose_estimation.modules.load_state import load_state
from .lightweight_human_pose_estimation.modules.pose import Pose
from .lightweight_human_pose_estimation import demo

class ImageProcessor:
    def __init__(self):
        self.net = None
        self.initialize_model()

    def initialize_model(self):
        """Initialize the pose estimation model"""
        if self.net is None:
            self.net = PoseEstimationWithMobileNet()
            checkpoint_path = os.path.join(os.path.dirname(__file__), 
                                         'lightweight_human_pose_estimation', 
                                         'checkpoint_iter_370000.pth')
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            load_state(self.net, checkpoint)
            self.net = self.net.cuda()
            self.net.eval()

    def process_image(self, image_path):

        try:
            output_dir = os.path.join('model', 'PiFu-singleview', 'sample_images')

            os.makedirs(output_dir, exist_ok=True)

            image_name = os.path.basename(image_path)
            new_image_path = os.path.join(output_dir, image_name)
            shutil.copy2(image_path, new_image_path)

            rect_path = new_image_path.replace('.%s' % (image_name.split('.')[-1]), '_rect.txt')

            img = cv2.imread(new_image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not read image")
            
            height_size = 512
            stride = 8
            upsample_ratio = 4

            heatmaps, pafs, scale, pad = demo.infer_fast(self.net, img, height_size, stride, upsample_ratio, cpu=False)
            
            total_keypoints_num = 0
            all_keypoints_by_type = []
            num_keypoints = Pose.num_kpts
            
            for kpt_idx in range(num_keypoints):
                total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
            
            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
            
            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
                all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
            

            rects = self._calculate_rects(img, pose_entries, all_keypoints, num_keypoints)
            

            np.savetxt(rect_path, rects, fmt='%d')
            
            return new_image_path, rect_path
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise

    def _calculate_rects(self, img, pose_entries, all_keypoints, num_keypoints):
        """Calculate rectangles for the image based on pose keypoints"""
        rects = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
                
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            valid_keypoints = []
            
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                    valid_keypoints.append([pose_keypoints[kpt_id, 0], pose_keypoints[kpt_id, 1]])
                    
            valid_keypoints = np.array(valid_keypoints)
            
            if len(valid_keypoints) > 0:
                rect = self._calculate_single_rect(img, valid_keypoints, pose_entries[n], pose_keypoints)
                rects.append(rect)
        

        if not rects:
            rects.append([0, 0, img.shape[1], img.shape[0]])

        rects = np.array(rects, dtype=np.int32)

        rects[:, 0] = np.clip(rects[:, 0], 0, img.shape[1])  
        rects[:, 1] = np.clip(rects[:, 1], 0, img.shape[0])  
        rects[:, 2] = np.clip(rects[:, 2], 1, img.shape[1] - rects[:, 0])  
        rects[:, 3] = np.clip(rects[:, 3], 1, img.shape[0] - rects[:, 1])  
        
        return rects

    def _calculate_single_rect(self, img, valid_keypoints, pose_entry, pose_keypoints):
        """Calculate rectangle for a single pose"""
        pmin = valid_keypoints.min(0)
        pmax = valid_keypoints.max(0)
        center = (0.5 * (pmax[:2] + pmin[:2])).astype(np.int32)
        
        width = max(1, pmax[0] - pmin[0])  
        height = max(1, pmax[1] - pmin[1]) 
        radius = int(0.65 * max(width, height))
        
        # Handle head keypoints
        has_nose = pose_entry[0] != -1.0
        has_eyes = pose_entry[14] != -1.0 or pose_entry[15] != -1.0
        has_ears = pose_entry[16] != -1.0 or pose_entry[17] != -1.0
        
        if has_nose or has_eyes or has_ears:
            center[1] = center[1] - int(0.35 * radius)
            radius = max(radius, int(1.4 * height))
        
        
        x1 = max(0, center[0] - radius)
        extra_top_padding = int(0.2 * radius)
        y1 = max(0, center[1] - radius - extra_top_padding)
        width = min(max(1, 2*radius), img.shape[1] - x1)  
        height = min(max(1, 2*radius + extra_top_padding), img.shape[0] - y1)  
        
        return [x1, y1, width, height]


image_processor = ImageProcessor()