
import os
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from skimage.io import imread
import cv2

from .models.encoders import ResnetEncoder, MLP, HRNEncoder
from .models.moderators import TempSoftmaxFusion
from .models.SMPLX import SMPLX
from .utils import util
from .utils import rotation_converter as converter
from .utils import tensor_cropper
from .utils.config import cfg


class PIXIE(object):
    def __init__(self, config=None, device="cuda:0"):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config

        self.device = device

        self.param_list_dict = {}
        for lst in self.cfg.params.keys():
            param_list = cfg.params.get(lst)
            self.param_list_dict[lst] = {i: cfg.model.get("n_" + i) for i in param_list}


        self._create_model()

        self._setup_cropper()

    def forward(self, data):


        param_dict = self.encode(
            {"body": {
                "image": data
            }},
            threthold=True,
            keep_local=True,
            copy_and_paste=False,
        )
        opdict = self.decode(param_dict["body"], param_type="body")

        return opdict

    def _setup_cropper(self):
        self.Cropper = {}
        for crop_part in ["head", "hand"]:
            data_cfg = self.cfg.dataset[crop_part]
            scale_size = (data_cfg.scale_min + data_cfg.scale_max) * 0.5
            self.Cropper[crop_part] = tensor_cropper.Cropper(
                crop_size=data_cfg.image_size,
                scale=[scale_size, scale_size],
                trans_scale=0,
            )

    def _create_model(self):
        self.model_dict = {}

        self.Encoder = {}
        for key in self.cfg.network.encoder.keys():
            if self.cfg.network.encoder.get(key).type == "resnet50":
                self.Encoder[key] = ResnetEncoder().to(self.device)
            elif self.cfg.network.encoder.get(key).type == "hrnet":
                self.Encoder[key] = HRNEncoder().to(self.device)
            self.model_dict[f"Encoder_{key}"] = self.Encoder[key].state_dict()


        self.Regressor = {}
        for key in self.cfg.network.regressor.keys():
            n_output = sum(self.param_list_dict[f"{key}_list"].values())
            channels = ([2048] + self.cfg.network.regressor.get(key).channels + [n_output])
            if self.cfg.network.regressor.get(key).type == "mlp":
                self.Regressor[key] = MLP(channels=channels).to(self.device)
            self.model_dict[f"Regressor_{key}"] = self.Regressor[key].state_dict()


        self.Extractor = {}
        for key in self.cfg.network.extractor.keys():
            channels = [2048] + self.cfg.network.extractor.get(key).channels + [2048]
            if self.cfg.network.extractor.get(key).type == "mlp":
                self.Extractor[key] = MLP(channels=channels).to(self.device)
            self.model_dict[f"Extractor_{key}"] = self.Extractor[key].state_dict()

  
        self.Moderator = {}
        for key in self.cfg.network.moderator.keys():
            share_part = key.split("_")[0]
            detach_inputs = self.cfg.network.moderator.get(key).detach_inputs
            detach_feature = self.cfg.network.moderator.get(key).detach_feature
            channels = [2048 * 2] + self.cfg.network.moderator.get(key).channels + [2]
            self.Moderator[key] = TempSoftmaxFusion(
                detach_inputs=detach_inputs,
                detach_feature=detach_feature,
                channels=channels,
            ).to(self.device)
            self.model_dict[f"Moderator_{key}"] = self.Moderator[key].state_dict()

 
        self.smplx = SMPLX(self.cfg.model).to(self.device)
        self.part_indices = self.smplx.part_indices

      
        model_path = self.cfg.pretrained_modelpath
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            for key in self.model_dict.keys():
                util.copy_state_dict(self.model_dict[key], checkpoint[key])
        else:
            print(f"pixie trained model path: {model_path} does not exist!")
            exit()
       
        for module in [self.Encoder, self.Regressor, self.Moderator, self.Extractor]:
            for net in module.values():
                net.eval()

    def decompose_code(self, code, num_dict):
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
        return code_dict

    def part_from_body(self, image, part_key, points_dict, crop_joints=None):
        """crop part(head/left_hand/right_hand) out from body data, joints also change accordingly"""
        assert part_key in ["head", "left_hand", "right_hand"]
        assert "smplx_kpt" in points_dict.keys()
        if part_key == "head":
            # use face 68 kpts for cropping head image
            indices_key = "face"
        elif part_key == "left_hand":
            indices_key = "left_hand"
        elif part_key == "right_hand":
            indices_key = "right_hand"

        # get points for cropping
        part_indices = self.part_indices[indices_key]
        if crop_joints is not None:
            points_for_crop = crop_joints[:, part_indices]
        else:
            points_for_crop = points_dict["smplx_kpt"][:, part_indices]

    
        cropper_key = "hand" if "hand" in part_key else part_key
        points_scale = image.shape[-2:]
        cropped_image, tform = self.Cropper[cropper_key].crop(image, points_for_crop, points_scale)

        cropped_points_dict = {}
        for points_key in points_dict.keys():
            points = points_dict[points_key]
            cropped_points = self.Cropper[cropper_key].transform_points(
                points, tform, points_scale, normalize=True
            )
            cropped_points_dict[points_key] = cropped_points
        return cropped_image, cropped_points_dict

    @torch.no_grad()
    def encode(
        self,
        data,
        threthold=True,
        keep_local=True,
        copy_and_paste=False,
        body_only=False,
    ):

        for key in data.keys():
            assert key in ["body", "head", "hand"]

        feature = {}
        param_dict = {}

    
        for key in data.keys():
            part = key
        
            feature[key] = {}
            feature[key][part] = self.Encoder[part](data[key]["image"])

    
            if key == "head" or key == "hand":
    
                part_dict = self.decompose_code(
                    self.Regressor[part](feature[key][part]),
                    self.param_list_dict[f"{part}_list"],
                )
              
                feature[key][f"{key}_share"] = feature[key][key]
                share_dict = self.decompose_code(
                    self.Regressor[f"{part}_share"](feature[key][f"{part}_share"]),
                    self.param_list_dict[f"{part}_share_list"],
                )
           
                param_dict[key] = {**share_dict, **part_dict}

    
            if key == "body":
                fusion_weight = {}
                f_body = feature["body"]["body"]
               
                for part_name in ["head", "left_hand", "right_hand"]:
                    feature["body"][f"{part_name}_share"] = self.Extractor[f"{part_name}_share"](
                        f_body
                    )

              
                if (
                    "head_image" not in data[key].keys() or
                    "left_hand_image" not in data[key].keys() or
                    "right_hand_image" not in data[key].keys()
                ):

                    body_dict = self.decompose_code(
                        self.Regressor[part](feature[key][part]),
                        self.param_list_dict[part + "_list"],
                    )
               
                    head_share_dict = self.decompose_code(
                        self.Regressor["head" + "_share"](feature[key]["head" + "_share"]),
                        self.param_list_dict["head" + "_share_list"],
                    )
            
                    right_hand_share_dict = self.decompose_code(
                        self.Regressor["hand" + "_share"](feature[key]["right_hand" + "_share"]),
                        self.param_list_dict["hand" + "_share_list"],
                    )
                
                    left_hand_share_dict = self.decompose_code(
                        self.Regressor["hand" + "_share"](feature[key]["left_hand" + "_share"]),
                        self.param_list_dict["hand" + "_share_list"],
                    )
            
                    left_hand_share_dict["left_hand_pose"] = left_hand_share_dict.pop(
                        "right_hand_pose"
                    )
                    left_hand_share_dict["left_wrist_pose"] = left_hand_share_dict.pop(
                        "right_wrist_pose"
                    )
                    param_dict[key] = {
                        **body_dict,
                        **head_share_dict,
                        **left_hand_share_dict,
                        **right_hand_share_dict,
                    }
                    if body_only:
                        param_dict["moderator_weight"] = None
                        return param_dict
                    prediction_body_only = self.decode(param_dict[key], param_type="body")
                   
                    for part_name in ["head", "left_hand", "right_hand"]:
                        part = part_name.split("_")[-1]
                        points_dict = {
                            "smplx_kpt": prediction_body_only["smplx_kpt"],
                            "trans_verts": prediction_body_only["transformed_vertices"],
                        }
                        image_hd = torchvision.transforms.Resize(1024)(data["body"]["image"])
                        cropped_image, cropped_joints_dict = self.part_from_body(
                            image_hd, part_name, points_dict
                        )
                        data[key][part_name + "_image"] = cropped_image

        
                for part_name in ["head", "left_hand", "right_hand"]:
                    part = part_name.split("_")[-1]
                    cropped_image = data[key][part_name + "_image"]
                 
                    if part_name == "left_hand":
                        cropped_image = torch.flip(cropped_image, dims=(-1, ))
                
                    f_part = self.Encoder[part](cropped_image)
                    part_dict = self.decompose_code(
                        self.Regressor[part](f_part),
                        self.param_list_dict[f"{part}_list"],
                    )
                    part_share_dict = self.decompose_code(
                        self.Regressor[f"{part}_share"](f_part),
                        self.param_list_dict[f"{part}_share_list"],
                    )
                    param_dict["body_" + part_name] = {**part_dict, **part_share_dict}

             
                    f_body_out, f_part_out, f_weight = self.Moderator[f"{part}_share"](
                        feature["body"][f"{part_name}_share"], f_part, work=True
                    )
                    if copy_and_paste:
   
                        feature["body"][f"{part_name}_share"] = f_part
                    elif threthold and part == "hand":
  
                        part_w = f_weight[:, [1]]
                        part_w[part_w > 0.7] = 1.0
                        f_body_out = (
                            feature["body"][f"{part_name}_share"] * (1.0 - part_w) + f_part * part_w
                        )
                        feature["body"][f"{part_name}_share"] = f_body_out
                    else:
                        feature["body"][f"{part_name}_share"] = f_body_out
                    fusion_weight[part_name] = f_weight
                param_dict["moderator_weight"] = fusion_weight

           
                head_share_dict = self.decompose_code(
                    self.Regressor["head" + "_share"](feature[key]["head" + "_share"]),
                    self.param_list_dict["head" + "_share_list"],
                )
               
                right_hand_share_dict = self.decompose_code(
                    self.Regressor["hand" + "_share"](feature[key]["right_hand" + "_share"]),
                    self.param_list_dict["hand" + "_share_list"],
                )
   
                left_hand_share_dict = self.decompose_code(
                    self.Regressor["hand" + "_share"](feature[key]["left_hand" + "_share"]),
                    self.param_list_dict["hand" + "_share_list"],
                )

                left_hand_share_dict["left_hand_pose"] = left_hand_share_dict.pop("right_hand_pose")
                left_hand_share_dict["left_wrist_pose"] = left_hand_share_dict.pop(
                    "right_wrist_pose"
                )
                param_dict["body"] = {
                    **body_dict,
                    **head_share_dict,
                    **left_hand_share_dict,
                    **right_hand_share_dict,
                }
     
                param_dict["body"]["tex"] = param_dict["body_head"]["tex"]
                param_dict["body"]["light"] = param_dict["body_head"]["light"]

                if keep_local:
                    param_dict[key]["exp"] = param_dict["body_head"]["exp"]
                    param_dict[key]["right_hand_pose"] = param_dict["body_right_hand"][
                        "right_hand_pose"]
                    param_dict[key]["left_hand_pose"] = param_dict["body_left_hand"][
                        "right_hand_pose"]

        return param_dict

    def convert_pose(self, param_dict, param_type):

        assert param_type in ["body", "head", "hand"]

     
        for key in param_dict:
            if "pose" in key and "jaw" not in key:
                param_dict[key] = converter.batch_cont2matrix(param_dict[key])
        if param_type == "body" or param_type == "head":
            param_dict["jaw_pose"] = converter.batch_euler2matrix(param_dict["jaw_pose"]
                                                                 )[:, None, :, :]


        if param_type == "head":
            batch_size = param_dict["shape"].shape[0]
            param_dict["abs_head_pose"] = param_dict["head_pose"].clone()
            param_dict["global_pose"] = param_dict["head_pose"]
            param_dict["partbody_pose"] = self.smplx.body_pose.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            )[:, :self.param_list_dict["body_list"]["partbody_pose"]]
            param_dict["neck_pose"] = self.smplx.neck_pose.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            )
            param_dict["left_wrist_pose"] = self.smplx.neck_pose.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            )
            param_dict["left_hand_pose"] = self.smplx.left_hand_pose.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            )
            param_dict["right_wrist_pose"] = self.smplx.neck_pose.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            )
            param_dict["right_hand_pose"] = self.smplx.right_hand_pose.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            )
        elif param_type == "hand":
            batch_size = param_dict["right_hand_pose"].shape[0]
            param_dict["abs_right_wrist_pose"] = param_dict["right_wrist_pose"].clone()
            dtype = param_dict["right_hand_pose"].dtype
            device = param_dict["right_hand_pose"].device
            x_180_pose = (torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(1, 1, 1))
            x_180_pose[0, 2, 2] = -1.0
            x_180_pose[0, 1, 1] = -1.0
            param_dict["global_pose"] = x_180_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            param_dict["shape"] = self.smplx.shape_params.expand(batch_size, -1)
            param_dict["exp"] = self.smplx.expression_params.expand(batch_size, -1)
            param_dict["head_pose"] = self.smplx.head_pose.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            )
            param_dict["neck_pose"] = self.smplx.neck_pose.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            )
            param_dict["jaw_pose"] = self.smplx.jaw_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            param_dict["partbody_pose"] = self.smplx.body_pose.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            )[:, :self.param_list_dict["body_list"]["partbody_pose"]]
            param_dict["left_wrist_pose"] = self.smplx.neck_pose.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            )
            param_dict["left_hand_pose"] = self.smplx.left_hand_pose.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            )
        elif param_type == "body":
            batch_size = param_dict["shape"].shape[0]
            param_dict["abs_head_pose"] = param_dict["head_pose"].clone()
            param_dict["abs_right_wrist_pose"] = param_dict["right_wrist_pose"].clone()
            param_dict["abs_left_wrist_pose"] = param_dict["left_wrist_pose"].clone()
            param_dict["left_wrist_pose"] = util.flip_pose(param_dict["left_wrist_pose"])
            param_dict["left_hand_pose"] = util.flip_pose(param_dict["left_hand_pose"])
        else:
            exit()

        return param_dict

    def decode(self, param_dict, param_type):
        if "jaw_pose" in param_dict.keys() and len(param_dict["jaw_pose"].shape) == 2:
            self.convert_pose(param_dict, param_type)
        elif param_dict["right_wrist_pose"].shape[-1] == 6:
            self.convert_pose(param_dict, param_type)

        partbody_pose = param_dict["partbody_pose"]
        param_dict["body_pose"] = torch.cat(
            [
                partbody_pose[:, :11],
                param_dict["neck_pose"],
                partbody_pose[:, 11:11 + 2],
                param_dict["head_pose"],
                partbody_pose[:, 13:13 + 4],
                param_dict["left_wrist_pose"],
                param_dict["right_wrist_pose"],
            ],
            dim=1,
        )
        if param_type == "head" or param_type == "body":
            param_dict["body_pose"] = self.smplx.pose_abs2rel(
                param_dict["global_pose"], param_dict["body_pose"], abs_joint="head"
            )
        if param_type == "hand" or param_type == "body":
            param_dict["body_pose"] = self.smplx.pose_abs2rel(
                param_dict["global_pose"],
                param_dict["body_pose"],
                abs_joint="left_wrist",
            )
            param_dict["body_pose"] = self.smplx.pose_abs2rel(
                param_dict["global_pose"],
                param_dict["body_pose"],
                abs_joint="right_wrist",
            )

        if self.cfg.model.check_pose:
            for pose_ind in [14]:  
                curr_pose = param_dict["body_pose"][:, pose_ind]
                euler_pose = converter._compute_euler_from_matrix(curr_pose)
                for i, max_angle in enumerate([20, 70, 10]):
                    euler_pose_curr = euler_pose[:, i]
                    euler_pose_curr[euler_pose_curr != torch.clamp(
                        euler_pose_curr,
                        min=-max_angle * np.pi / 180,
                        max=max_angle * np.pi / 180,
                    )] = 0.0
                param_dict["body_pose"][:, pose_ind] = converter.batch_euler2matrix(euler_pose)


        verts, landmarks, joints = self.smplx(
            shape_params=param_dict["shape"],
            expression_params=param_dict["exp"],
            global_pose=param_dict["global_pose"],
            body_pose=param_dict["body_pose"],
            jaw_pose=param_dict["jaw_pose"],
            left_hand_pose=param_dict["left_hand_pose"],
            right_hand_pose=param_dict["right_hand_pose"],
        )
        smplx_kpt3d = joints.clone()


        cam = param_dict[param_type + "_cam"]
        trans_verts = util.batch_orth_proj(verts, cam)
        predicted_landmarks = util.batch_orth_proj(landmarks, cam)[:, :, :2]
        predicted_joints = util.batch_orth_proj(joints, cam)[:, :, :2]

        prediction = {
            "vertices": verts,
            "transformed_vertices": trans_verts,
            "face_kpt": predicted_landmarks,
            "smplx_kpt": predicted_joints,
            "smplx_kpt3d": smplx_kpt3d,
            "joints": joints,
            "cam": param_dict[param_type + "_cam"],
        }


        prediction["face_kpt"] = torch.cat(
            [prediction["face_kpt"][:, -17:], prediction["face_kpt"][:, :-17]], dim=1
        )

        prediction.update(param_dict)

        return prediction

    def decode_Tpose(self, param_dict):
        """return body mesh in T pose, support body and head param dict only"""
        verts, _, _ = self.smplx(
            shape_params=param_dict["shape"],
            expression_params=param_dict["exp"],
            jaw_pose=param_dict["jaw_pose"],
        )
        return verts
