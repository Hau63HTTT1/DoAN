import numpy as np


def keypoint_hflip(kp, img_width):

    if len(kp.shape) == 2:
        kp[:, 0] = (img_width - 1.) - kp[:, 0]
    elif len(kp.shape) == 3:
        kp[:, :, 0] = (img_width - 1.) - kp[:, :, 0]
    return kp


def convert_kps(joints2d, src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()

    out_joints2d = np.zeros((joints2d.shape[0], len(dst_names), joints2d.shape[-1]))

    for idx, jn in enumerate(dst_names):
        if jn in src_names:
            out_joints2d[:, idx] = joints2d[:, src_names.index(jn)]

    return out_joints2d


def get_perm_idxs(src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()
    idxs = [src_names.index(h) for h in dst_names if h in src_names]
    return idxs


def get_mpii3d_test_joint_names():
    return [
        'headtop',   
        'neck',
        'rshoulder',    
        'relbow',   
        'rwrist',    
        'lshoulder',    
        'lelbow',    
        'lwrist',    
        'rhip',    
        'rknee',    
        'rankle',    
        'lhip',    
        'lknee',    
        'lankle',    
        'hip',    
        'Spine (H36M)',    
        'Head (H36M)',    
    ]


def get_mpii3d_joint_names():
    return [
        'spine3',   
        'spine4',   
        'spine2',    
        'Spine (H36M)',  
        'hip',    
        'neck',    
        'Head (H36M)',    
        "headtop",   
        'left_clavicle',   
        "lshoulder",   
        "lelbow",    
        "lwrist",    
        'left_hand',    
        'right_clavicle',    
        'rshoulder',    
        'relbow',    
        'rwrist',   
        'right_hand',   
        'lhip',    
        'lknee',  
        'lankle',    
        'left_foot',  
        'left_toe',   
        "rhip",    
        "rknee",    
        "rankle",    
        'right_foot',    
        'right_toe'    
    ]


def get_insta_joint_names():
    return [
        'OP RHeel',
        'OP RKnee',
        'OP RHip',
        'OP LHip',
        'OP LKnee',
        'OP LHeel',
        'OP RWrist',
        'OP RElbow',
        'OP RShoulder',
        'OP LShoulder',
        'OP LElbow',
        'OP LWrist',
        'OP Neck',
        'headtop',
        'OP Nose',
        'OP LEye',
        'OP REye',
        'OP LEar',
        'OP REar',
        'OP LBigToe',
        'OP RBigToe',
        'OP LSmallToe',
        'OP RSmallToe',
        'OP LAnkle',
        'OP RAnkle',
    ]


def get_mmpose_joint_names():

    return [
        'OP Nose',    
        'OP LEye',    
        'OP REye',   
        'OP LEar',   
        'OP REar',    
        'OP LShoulder',   
        'OP RShoulder',    
        'OP LElbow',   
        'OP RElbow',  
        'OP LWrist',   
        'OP RWrist',   
        'OP LHip',    
        'OP RHip',  
        'OP LKnee',  
        'OP RKnee', 
        'OP LAnkle',  
        'OP RAnkle',    
        'OP LBigToe',    
        'OP LSmallToe',   
        'OP LHeel',    
        'OP RBigToe',    
        'OP RSmallToe', 
        'OP RHeel',    
    ]


def get_insta_skeleton():
    return np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 10],
            [2, 8],
            [3, 9],
            [10, 11],
            [8, 12],
            [9, 12],
            [12, 13],
            [12, 14],
            [14, 15],
            [14, 16],
            [15, 17],
            [16, 18],
            [0, 20],
            [20, 22],
            [5, 19],
            [19, 21],
            [5, 23],
            [0, 24],
        ]
    )


def get_staf_skeleton():
    return np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 5],
            [5, 6],
            [6, 7],
            [1, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [8, 12],
            [12, 13],
            [13, 14],
            [0, 15],
            [0, 16],
            [15, 17],
            [16, 18],
            [2, 9],
            [5, 12],
            [1, 19],
            [20, 19],
        ]
    )


def get_staf_joint_names():
    return [
        'OP Nose',
        'OP Neck',
        'OP RShoulder',
        'OP RElbow',
        'OP RWrist',
        'OP LShoulder',
        'OP LElbow',
        'OP LWrist',
        'OP MidHip',
        'OP RHip',
        'OP RKnee',
        'OP RAnkle',
        'OP LHip',
        'OP LKnee',
        'OP LAnkle',
        'OP REye',
        'OP LEye',
        'OP REar',
        'OP LEar',
        'Neck (LSP)',
        'Top of Head (LSP)',
    ]


def get_spin_op_joint_names():
    return [
        'OP Nose',
        'OP Neck',
        'OP RShoulder',
        'OP RElbow',
        'OP RWrist',
        'OP LShoulder',
        'OP LElbow',
        'OP LWrist',
        'OP MidHip',
        'OP RHip',
        'OP RKnee',
        'OP RAnkle',
        'OP LHip',
        'OP LKnee',
        'OP LAnkle',
        'OP REye',
        'OP LEye',
        'OP REar',
        'OP LEar',
        'OP LBigToe',
        'OP LSmallToe',
        'OP LHeel',
        'OP RBigToe',
        'OP RSmallToe',
        'OP RHeel',
    ]


def get_openpose_joint_names():
    return [
        'OP Nose',
        'OP Neck',
        'OP RShoulder',
        'OP RElbow',
        'OP RWrist',
        'OP LShoulder',
        'OP LElbow',
        'OP LWrist',
        'OP MidHip',
        'OP RHip',
        'OP RKnee',
        'OP RAnkle',
        'OP LHip',
        'OP LKnee',
        'OP LAnkle',
        'OP REye',
        'OP LEye',
        'OP REar',
        'OP LEar',
        'OP LBigToe',
        'OP LSmallToe',
        'OP LHeel',
        'OP RBigToe',
        'OP RSmallToe',
        'OP RHeel',
    ]


def get_spin_joint_names():
    return [
        'OP Nose',
        'OP Neck',
        'OP RShoulder',
        'OP RElbow',
        'OP RWrist',
        'OP LShoulder',
        'OP LElbow',
        'OP LWrist',
        'OP MidHip',
        'OP RHip',
        'OP RKnee',
        'OP RAnkle',
        'OP LHip',
        'OP LKnee',
        'OP LAnkle',
        'OP REye',
        'OP LEye',
        'OP REar',
        'OP LEar',
        'OP LBigToe',
        'OP LSmallToe',
        'OP LHeel',
        'OP RBigToe',
        'OP RSmallToe',
        'OP RHeel',
        'rankle',
        'rknee',
        'rhip',
        'lhip',
        'lknee',
        'lankle',
        'rwrist',
        'relbow',
        'rshoulder',
        'lshoulder',
        'lelbow',
        'lwrist',
        'neck',
        'headtop',
        'hip',
        'thorax',
        'Spine (H36M)',
        'Jaw (H36M)',
        'Head (H36M)',
        'nose',
        'leye',
        'reye',
        'lear',
        'rear',
    ]


def get_muco3dhp_joint_names():
    return [
        'headtop', 'thorax', 'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow', 'lwrist',
        'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle', 'hip', 'Spine (H36M)', 'Head (H36M)',
        'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe'
    ]


def get_h36m_joint_names():
    return [
        'hip',
        'lhip',
        'lknee',
        'lankle',
        'rhip',
        'rknee',
        'rankle',
        'Spine (H36M)',
        'neck',
        'Head (H36M)',
        'headtop',
        'lshoulder',
        'lelbow',
        'lwrist',
        'rshoulder',
        'relbow',
        'rwrist',
    ]


def get_spin_skeleton():
    return np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 5],
            [5, 6],
            [6, 7],
            [1, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [8, 12],
            [12, 13],
            [13, 14],
            [0, 15],
            [0, 16],
            [15, 17],
            [16, 18],
            [21, 19],
            [19, 20],
            [14, 21],
            [11, 24],
            [24, 22],
            [22, 23],
            [0, 38],
        ]
    )


def get_openpose_skeleton():
    return np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 5],
            [5, 6],
            [6, 7],
            [1, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [8, 12],
            [12, 13],
            [13, 14],
            [0, 15],
            [0, 16],
            [15, 17],
            [16, 18],
            [21, 19],
            [19, 20],
            [14, 21],
            [11, 24],
            [24, 22],
            [22, 23],
        ]
    )


def get_posetrack_joint_names():
    return [
        "nose", "neck", "headtop", "lear", "rear", "lshoulder", "rshoulder", "lelbow", "relbow",
        "lwrist", "rwrist", "lhip", "rhip", "lknee", "rknee", "lankle", "rankle"
    ]


def get_posetrack_original_kp_names():
    return [
        'nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear', 'left_shoulder',
        'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
        'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]


def get_pennaction_joint_names():
    return [
        "headtop",
        "lshoulder",
        "rshoulder",
        "lelbow",
        "relbow",
        "lwrist",
        "rwrist",
        "lhip",
        "rhip",
        "lknee",
        "rknee",
        "lankle",
        "rankle"
    ]


def get_common_joint_names():
    return [
        "rankle",
        "rknee",
        "rhip",
        "lhip",
        "lknee",
        "lankle",
        "rwrist",
        "relbow",
        "rshoulder",
        "lshoulder",
        "lelbow",
        "lwrist",
        "neck",
        "headtop",
    ]


def get_common_paper_joint_names():
    return [
        "Right Ankle",
        "Right Knee",
        "Right Hip",
        "Left Hip",
        "Left Knee",
        "Left Ankle",
        "Right Wrist",
        "Right Elbow",
        "Right Shoulder",
        "Left Shoulder",
        "Left Elbow",
        "Left Wrist",
        "Neck",
        "Head",
    ]


def get_common_skeleton():
    return np.array(
        [
            [0, 1],
            [1, 2],
            [3, 4],
            [4, 5],
            [6, 7],
            [7, 8],
            [8, 2],
            [8, 9],
            [9, 3],
            [2, 3],
            [8, 12],
            [9, 10],
            [12, 9],
            [10, 11],
            [12, 13],
        ]
    )


def get_coco_joint_names():
    return [
        "nose",
        "leye",
        "reye",
        "lear",
        "rear",
        "lshoulder",
        "rshoulder",
        "lelbow",
        "relbow",
        "lwrist",
        "rwrist",
        "lhip",
        "rhip",
        "lknee",
        "rknee",
        "lankle",
        "rankle",
    ]


def get_ochuman_joint_names():
    return [
        'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle',
        'lhip', 'lknee', 'lankle', 'headtop', 'neck', 'rear', 'lear', 'nose', 'reye', 'leye'
    ]


def get_crowdpose_joint_names():
    return [
        'lshoulder', 'rshoulder', 'lelbow', 'relbow', 'lwrist', 'rwrist', 'lhip', 'rhip', 'lknee',
        'rknee', 'lankle', 'rankle', 'headtop', 'neck'
    ]


def get_coco_skeleton():
    return np.array(
        [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
            [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
        ]
    )


def get_mpii_joint_names():
    return [
        "rankle",
        "rknee",
        "rhip",
        "lhip",
        "lknee",
        "lankle",
        "hip",
        "thorax",
        "neck",
        "headtop",
        "rwrist",
        "relbow",
        "rshoulder",
        "lshoulder",
        "lelbow",
        "lwrist",
    ]


def get_mpii_skeleton():
    return np.array(
        [
            [0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [7, 12],
            [12, 11], [11, 10], [7, 13], [13, 14], [14, 15]
        ]
    )


def get_aich_joint_names():
    return [
        "rshoulder",
        "relbow",
        "rwrist",
        "lshoulder",
        "lelbow",
        "lwrist",
        "rhip",
        "rknee",
        "rankle",
        "lhip",
        "lknee",
        "lankle",
        "headtop",
        "neck",
    ]


def get_aich_skeleton():
    return np.array(
        [
            [0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13], [13, 0],
            [13, 3], [0, 6], [3, 9]
        ]
    )


def get_3dpw_joint_names():
    return [
        "nose",
        "thorax",
        "rshoulder",
        "relbow",
        "rwrist",
        "lshoulder",
        "lelbow",
        "lwrist",
        "rhip",
        "rknee",
        "rankle",
        "lhip",
        "lknee",
        "lankle",
    ]


def get_3dpw_skeleton():
    return np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [2, 8], [5, 11], [8, 11],
            [8, 9], [9, 10], [11, 12], [12, 13]
        ]
    )


def get_smplcoco_joint_names():
    return [
        "rankle",
        "rknee",
        "rhip",
        "lhip",
        "lknee",
        "lankle",
        "rwrist",
        "relbow",
        "rshoulder",
        "lshoulder",
        "lelbow",
        "lwrist",
        "neck",
        "headtop",
        "nose",
        "leye",
        "reye",
        "lear",
        "rear",
    ]


def get_smplcoco_skeleton():
    return np.array(
        [
            [0, 1],
            [1, 2],
            [3, 4],
            [4, 5],
            [6, 7],
            [7, 8],
            [8, 12],
            [12, 9],
            [9, 10],
            [10, 11],
            [12, 13],
            [14, 15],
            [15, 17],
            [16, 18],
            [14, 16],
            [8, 2],
            [9, 3],
            [2, 3],
        ]
    )


def get_smpl_joint_names():
    return [
        'hips',
        'leftUpLeg',
        'rightUpLeg',
        'spine',
        'leftLeg',
        'rightLeg',
        'spine1',
        'leftFoot',
        'rightFoot',
        'spine2',
        'leftToeBase',
        'rightToeBase',
        'neck',
        'leftShoulder',
        'rightShoulder',
        'head',
        'leftArm',
        'rightArm',
        'leftForeArm',
        'rightForeArm',
        'leftHand',
        'rightHand',
        'leftHandIndex1',
        'rightHandIndex1',
    ]


def get_smpl_paper_joint_names():
    return [
        'Hips',
        'Left Hip',
        'Right Hip',
        'Spine',
        'Left Knee',
        'Right Knee',
        'Spine_1',
        'Left Ankle',
        'Right Ankle',
        'Spine_2',
        'Left Toe',
        'Right Toe',
        'Neck',
        'Left Shoulder',
        'Right Shoulder',
        'Head',
        'Left Arm',
        'Right Arm',
        'Left Elbow',
        'Right Elbow',
        'Left Hand',
        'Right Hand',
        'Left Thumb',
        'Right Thumb',
    ]


def get_smpl_neighbor_triplets():
    return [
        [0, 1, 2],
        [1, 4, 0],
        [2, 0, 5],
        [3, 0, 6],
        [4, 7, 1],
        [5, 2, 8],
        [6, 3, 9],
        [7, 10, 4],
        [8, 5, 11],
        [9, 13, 14],
        [10, 7, 4],
        [11, 8, 5],
        [12, 9, 15],
        [13, 16, 9],
        [14, 9, 17],
        [15, 9, 12],
        [16, 18, 13],
        [17, 14, 19],
        [18, 20, 16],
        [19, 17, 21],
        [20, 22, 18],
        [21, 19, 23],
        [22, 20, 18],
        [23, 19, 21],
    ]


def get_smpl_skeleton():
    return np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 4],
            [2, 5],
            [3, 6],
            [4, 7],
            [5, 8],
            [6, 9],
            [7, 10],
            [8, 11],
            [9, 12],
            [9, 13],
            [9, 14],
            [12, 15],
            [13, 16],
            [14, 17],
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
            [20, 22],
            [21, 23],
        ]
    )


def map_spin_joints_to_smpl():
    return [
        [(39, 27, 28), 0],
        [(28, ), 1],
        [(27, ), 2],
        [(41, 27, 28, 39), 3],
        [(29, ), 4],
        [(26, ), 5],
        [(
            41,
            40,
            33,
            34,
        ), 6],
        [(30, ), 7],
        [(25, ), 8],
        [(40, 33, 34), 9],
        [(30, ), 10],
        [(25, ), 11],
        [(37, 42, 33, 34), 12],
        [(34, ), 13],
        [(33, ), 14],
        [(
            33,
            34,
            38,
            43,
            44,
            45,
            46,
            47,
            48,
        ), 15],
        [(34, ), 16],
        [(33, ), 17],
        [(35, ), 18],
        [(32, ), 19],
        [(36, ), 20],
        [(31, ), 21],
        [(36, ), 22],
        [(31, ), 23],
    ]


def map_smpl_to_common():
    return [
        [(11, 8), 0],
        [(5, ), 1],
        [(2, ), 2],
        [(1, ), 3],
        [(4, ), 4],
        [(10, 7), 5],
        [(21, 23), 6],
        [(18, ), 7],
        [(17, 14), 8],
        [(16, 13), 9],
        [(19, ), 10],
        [(20, 22), 11],
        [(0, 3, 6, 9, 12), 12],
        [(15, ), 13],
    ]


def relation_among_spin_joints():
    return [
        [(), 25],
        [(), 26],
        [(39, ), 27],
        [(39, ), 28],
        [(), 29],
        [(), 30],
        [(), 31],
        [(), 32],
        [(), 33],
        [(), 34],
        [(), 35],
        [(), 36],
        [(
            40,
            42,
            44,
            43,
            38,
            33,
            34,
        ), 37],
        [(
            43,
            44,
            45,
            46,
            47,
            48,
            33,
            34,
        ), 38],
        [(
            27,
            28,
        ), 39],
        [(
            27,
            28,
            37,
            41,
            42,
        ), 40],
        [(
            27,
            28,
            39,
            40,
        ), 41],
        [(
            37,
            38,
            44,
            45,
            46,
            47,
            48,
        ), 42],
        [(
            44,
            45,
            46,
            47,
            48,
            38,
            42,
            37,
            33,
            34,
        ), 43],
        [(44, 45, 46, 47, 48, 38, 42, 37, 33, 34), 44],
        [(44, 45, 46, 47, 48, 38, 42, 37, 33, 34), 45],
        [(44, 45, 46, 47, 48, 38, 42, 37, 33, 34), 46],
        [(44, 45, 46, 47, 48, 38, 42, 37, 33, 34), 47],
        [(44, 45, 46, 47, 48, 38, 42, 37, 33, 34), 48],
    ]
