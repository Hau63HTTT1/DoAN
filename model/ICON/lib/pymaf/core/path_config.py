
import os

# pymaf
pymaf_data_dir = os.path.join(os.path.dirname(__file__), "../../../data/HPS/pymaf_data")

SMPL_MEAN_PARAMS = os.path.join(pymaf_data_dir, "smpl_mean_params.npz")
SMPL_MODEL_DIR = os.path.join(pymaf_data_dir, "../../smpl_related/models/smpl")
MESH_DOWNSAMPLEING = os.path.join(pymaf_data_dir, "mesh_downsampling.npz")

CUBE_PARTS_FILE = os.path.join(pymaf_data_dir, "cube_parts.npy")
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(pymaf_data_dir, "J_regressor_extra.npy")
JOINT_REGRESSOR_H36M = os.path.join(pymaf_data_dir, "J_regressor_h36m.npy")
VERTEX_TEXTURE_FILE = os.path.join(pymaf_data_dir, "vertex_texture.npy")
SMPL_MEAN_PARAMS = os.path.join(pymaf_data_dir, "smpl_mean_params.npz")
CHECKPOINT_FILE = os.path.join(pymaf_data_dir, "pretrained_model/PyMAF_model_checkpoint.pt")

# pare
pare_data_dir = os.path.join(os.path.dirname(__file__), "../../../data/HPS/pare_data")
CFG = os.path.join(pare_data_dir, "pare/checkpoints/pare_w_3dpw_config.yaml")
CKPT = os.path.join(pare_data_dir, "pare/checkpoints/pare_w_3dpw_checkpoint.ckpt")

# hybrik
hybrik_data_dir = os.path.join(os.path.dirname(__file__), "../../../data/HPS/hybrik_data")
HYBRIK_CFG = os.path.join(hybrik_data_dir, "hybrik_config.yaml")
HYBRIK_CKPT = os.path.join(hybrik_data_dir, "pretrained_w_cam.pth")
