import os

# Training
GPU_ID = 0
DISPLAY_ID = GPU_ID * 10 + 10
NAME = 'pifu_demo/rp_Man'


BATCH_SIZE = 1
MLP_DIM = '257 1024 512 256 128 1'
MLP_DIM_COLOR = '513 1024 512 256 128 3'

VOL_RES = 256

CHECKPOINTS_NETG_PATH = './checkpoints/net_G'
CHECKPOINTS_NETC_PATH = './checkpoints/net_C'

TEST_FOLDER_PATH = './sample_images/rp_Man'


os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
cmd = f"python ./apps/eval.py --name {NAME} --batch_size {BATCH_SIZE} --mlp_dim {MLP_DIM} --mlp_dim_color {MLP_DIM_COLOR} --num_stack 4 --num_hourglass 2 --resolution {VOL_RES} --hg_down ave_pool --norm group --norm_color group --test_folder_path {TEST_FOLDER_PATH} --load_netG_checkpoint_path {CHECKPOINTS_NETG_PATH} --load_netC_checkpoint_path {CHECKPOINTS_NETC_PATH} --num_views 4"

os.system(cmd)
