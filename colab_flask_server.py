from flask import Flask, request, send_file, jsonify
from flask_ngrok import run_with_ngrok
import os
import torch
import cv2
import numpy as np
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose
import demo
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)
run_with_ngrok(app)

# Global variables for models and paths
UPLOAD_FOLDER = '/content/pifuhd/sample_images'
RESULTS_FOLDER = '/content/pifuhd/results/pifuhd_final/recon'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def get_rect(net, images, height_size):
    net = net.eval()
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    
    for image in images:
        rect_path = image.replace('.%s' % (image.split('.')[-1]), '_rect.txt')
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = demo.infer_fast(net, img, height_size, stride, upsample_ratio, cpu=False)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        # Use the same rectangle detection logic as in your original code
        rects = []
        if len(pose_entries) == 0:
            rects.append([0, 0, img.shape[1], img.shape[0]])
        else:
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
                    pmin = valid_keypoints.min(0)
                    pmax = valid_keypoints.max(0)
                    center = (0.5 * (pmax[:2] + pmin[:2])).astype(np.int32)
                    radius = int(1.45 * max(pmax[0] - pmin[0], pmax[1] - pmin[1]))
                    
                    x1 = center[0] - radius
                    y1 = center[1] - radius
                    rects.append([x1, y1, 2*radius, 2*radius])
                else:
                    rects.append([0, 0, img.shape[1], img.shape[0]])
        
        np.savetxt(rect_path, np.array(rects), fmt='%d')

def process_image(image_path):
    # Initialize pose estimation model
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load('/content/lightweight-human-pose-estimation.pytorch/checkpoint_iter_370000.pth', map_location='cpu')
    load_state(net, checkpoint)
    
    # Get rectangle coordinates
    get_rect(net.cuda(), [image_path], 512)
    
    # Run PiFuHD
    image_dir = os.path.dirname(image_path)
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Run the PiFuHD model
    os.system(f'cd /content/pifuhd && python -m apps.simple_test -r 256 --use_rect -i {image_dir}')
    
    # Get output paths
    obj_path = f'/content/pifuhd/results/pifuhd_final/recon/result_{file_name}_256.obj'
    
    return obj_path

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Process the image and get the 3D model
            obj_path = process_image(filepath)
            
            # Send the 3D model file back
            return send_file(obj_path, as_attachment=True)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run() 