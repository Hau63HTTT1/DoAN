
import yaml
import os.path as osp
import torch
import numpy as np
from ..dataset.mesh_util import *
from ..net.geometry import orthogonal
from pytorch3d.renderer.mesh import rasterize_meshes
from .render_utils import Pytorch3dRasterizer
from pytorch3d.structures import Meshes
import cv2
from PIL import Image
from tqdm import tqdm
import os
from termcolor import colored

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.cloud_io import atomic_save
from pytorch_lightning.utilities import rank_zero_warn


def rename(old_dict, old_name, new_name):
    new_dict = {}
    for key, value in zip(old_dict.keys(), old_dict.values()):
        new_key = key if key != old_name else new_name
        new_dict[new_key] = old_dict[key]
    return new_dict


class SubTrainer(pl.Trainer):
    def save_checkpoint(self, filepath, weights_only=False):
        _checkpoint = self.checkpoint_connector.dump_checkpoint(weights_only)

        del_keys = []
        for key in _checkpoint["state_dict"].keys():
            for ig_key in ["normal_filter", "voxelization", "reconEngine"]:
                if ig_key in key:
                    del_keys.append(key)
        for key in del_keys:
            del _checkpoint["state_dict"][key]

        if self.is_global_zero:
            if self.training_type_plugin:
                checkpoint = self.training_type_plugin.on_save(_checkpoint)
            try:
                atomic_save(checkpoint, filepath)
            except AttributeError as err:
                if LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:
                    del checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]
                rank_zero_warn(
                    "Warning, `hyper_parameters` dropped from checkpoint."
                    f" An attribute is not picklable {err}"
                )
                atomic_save(checkpoint, filepath)


def load_networks(cfg, model, mlp_path, normal_path):

    model_dict = model.state_dict()
    main_dict = {}
    normal_dict = {}

    if os.path.exists(mlp_path) and mlp_path.endswith("ckpt"):
        main_dict = torch.load(mlp_path,
                               map_location=torch.device(f"cuda:{cfg.gpus[0]}"))["state_dict"]

        main_dict = {
            k: v
            for k, v in main_dict.items() if k in model_dict and v.shape == model_dict[k].shape and
            ("reconEngine" not in k) and ("normal_filter" not in k) and ("voxelization" not in k)
        }
        print(colored(f"Resume MLP weights from {mlp_path}", "green"))

    if os.path.exists(normal_path) and normal_path.endswith("ckpt"):
        normal_dict = torch.load(normal_path,
                                 map_location=torch.device(f"cuda:{cfg.gpus[0]}"))["state_dict"]

        for key in normal_dict.keys():
            normal_dict = rename(normal_dict, key, key.replace("netG", "netG.normal_filter"))

        normal_dict = {
            k: v
            for k, v in normal_dict.items() if k in model_dict and v.shape == model_dict[k].shape
        }
        print(colored(f"Resume normal model from {normal_path}", "green"))

    model_dict.update(main_dict)
    model_dict.update(normal_dict)
    model.load_state_dict(model_dict)

    del main_dict
    del normal_dict
    del model_dict
    torch.cuda.empty_cache()


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1], sample_tensor.shape[2],
        sample_tensor.shape[3]
    )
    return sample_tensor


def gen_mesh_eval(opt, net, cuda, data, resolution=None):
    resolution = opt.resolution if resolution is None else resolution
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter(image_tensor)

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        verts, faces, _, _ = reconstruction_faster(
            net, cuda, calib_tensor, resolution, b_min, b_max, use_octree=False
        )

    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')
        verts, faces = None, None
    return verts, faces


def gen_mesh(opt, net, cuda, data, save_path, resolution=None):
    resolution = opt.resolution if resolution is None else resolution
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter(image_tensor)

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (
                np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5
            )[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:, :, ::-1])).save(save_img_path)

        verts, faces, _, _ = reconstruction_faster(
            net, cuda, calib_tensor, resolution, b_min, b_max
        )
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        xyz_tensor = net.projection(verts_tensor, calib_tensor[:1])
        uv = xyz_tensor[:, :2, :]
        color = netG.index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5
        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')
        verts, faces, color = None, None, None
    return verts, faces, color


def gen_mesh_color(opt, netG, netC, cuda, data, save_path, use_octree=True):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    netG.filter(image_tensor)
    netC.filter(image_tensor)
    netC.attach(netG.get_im_feat())

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (
                np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5
            )[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:, :, ::-1])).save(save_img_path)

        verts, faces, _, _ = reconstruction_faster(
            netG, cuda, calib_tensor, opt.resolution, b_min, b_max, use_octree=use_octree
        )

        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        verts_tensor = reshape_sample_tensor(verts_tensor, opt.num_views)
        color = np.zeros(verts.shape)
        interval = 10000
        for i in range(len(color) // interval):
            left = i * interval
            right = i * interval + interval
            if i == len(color) // interval - 1:
                right = -1
            netC.query(verts_tensor[:, :, left:right], calib_tensor)
            rgb = netC.get_preds()[0].detach().cpu().numpy() * 0.5 + 0.5
            color[left:right] = rgb.T

        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')
        verts, faces, color = None, None, None
    return verts, faces, color


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def compute_acc(pred, gt, thresh=0.5):

    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt

def calc_error(opt, net, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        erorr_arr, IOU_arr, prec_arr, recall_arr = [], [], [], []
        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]

            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            sample_tensor = data['samples'].to(device=cuda).unsqueeze(0)
            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)
            label_tensor = data['labels'].to(device=cuda).unsqueeze(0)

            res, error = net.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor)

            IOU, prec, recall = compute_acc(res, label_tensor)

            erorr_arr.append(error.item())
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())

    return np.average(erorr_arr), np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)


def calc_error_color(opt, netG, netC, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        error_color_arr = []

        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]

            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            color_sample_tensor = data['color_samples'].to(device=cuda).unsqueeze(0)

            if opt.num_views > 1:
                color_sample_tensor = reshape_sample_tensor(color_sample_tensor, opt.num_views)

            rgb_tensor = data['rgbs'].to(device=cuda).unsqueeze(0)

            netG.filter(image_tensor)
            _, errorC = netC.forward(
                image_tensor,
                netG.get_im_feat(),
                color_sample_tensor,
                calib_tensor,
                labels=rgb_tensor
            )

            error_color_arr.append(errorC.item())

    return np.average(error_color_arr)



def query_func(opt, netG, features, points, proj_matrix=None):

    assert len(points) == 1
    samples = points.repeat(opt.num_views, 1, 1)
    samples = samples.permute(0, 2, 1)

    if proj_matrix is not None:
        samples = orthogonal(samples, proj_matrix)

    calib_tensor = torch.stack([torch.eye(4).float()], dim=0).type_as(samples)

    preds = netG.query(
        features=features, points=samples, calibs=calib_tensor, regressor=netG.if_regressor
    )

    if type(preds) is list:
        preds = preds[0]

    return preds


def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)


def in1d(ar1, ar2):
    mask = ar2.new_zeros((max(ar1.max(), ar2.max()) + 1, ), dtype=torch.bool)
    mask[ar2.unique()] = True
    return mask[ar1]


def get_visibility(xy, z, faces):

    xyz = torch.cat((xy, -z), dim=1)
    xyz = (xyz + 1.0) / 2.0
    faces = faces.long()

    rasterizer = Pytorch3dRasterizer(image_size=2**12)
    meshes_screen = Meshes(verts=xyz[None, ...], faces=faces[None, ...])
    raster_settings = rasterizer.raster_settings

    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        meshes_screen,
        image_size=raster_settings.image_size,
        blur_radius=raster_settings.blur_radius,
        faces_per_pixel=raster_settings.faces_per_pixel,
        bin_size=raster_settings.bin_size,
        max_faces_per_bin=raster_settings.max_faces_per_bin,
        perspective_correct=raster_settings.perspective_correct,
        cull_backfaces=raster_settings.cull_backfaces,
    )

    vis_vertices_id = torch.unique(faces[torch.unique(pix_to_face), :])
    vis_mask = torch.zeros(size=(z.shape[0], 1))
    vis_mask[vis_vertices_id] = 1.0

    return vis_mask


def batch_mean(res, key):

    return torch.stack([x[key] if isinstance(x, dict) else batch_mean(x, key) for x in res]).mean()


def tf_log_convert(log_dict):
    new_log_dict = log_dict.copy()
    for k, v in log_dict.items():
        new_log_dict[k.replace("_", "/")] = v
        del new_log_dict[k]

    return new_log_dict


def bar_log_convert(log_dict, name=None, rot=None):
    from decimal import Decimal

    new_log_dict = {}

    if name is not None:
        new_log_dict['name'] = name[0]
    if rot is not None:
        new_log_dict['rot'] = rot[0]

    for k, v in log_dict.items():
        color = "yellow"
        if 'loss' in k:
            color = "red"
            k = k.replace("loss", "L")
        elif 'acc' in k:
            color = "green"
            k = k.replace("acc", "A")
        elif 'iou' in k:
            color = "green"
            k = k.replace("iou", "I")
        elif 'prec' in k:
            color = "green"
            k = k.replace("prec", "P")
        elif 'recall' in k:
            color = "green"
            k = k.replace("recall", "R")

        if 'lr' not in k:
            new_log_dict[colored(k.split("_")[1], color)] = colored(f"{v:.3f}", color)
        else:
            new_log_dict[colored(k.split("_")[1], color)] = colored(f"{Decimal(str(v)):.1E}", color)

    if 'loss' in new_log_dict.keys():
        del new_log_dict['loss']

    return new_log_dict


def accumulate(outputs, rot_num, split):

    hparam_log_dict = {}

    metrics = outputs[0].keys()
    datasets = split.keys()

    for dataset in datasets:
        for metric in metrics:
            keyword = f"{dataset}-{metric}"
            if keyword not in hparam_log_dict.keys():
                hparam_log_dict[keyword] = 0
            for idx in range(split[dataset][0] * rot_num, split[dataset][1] * rot_num):
                hparam_log_dict[keyword] += outputs[idx][metric]
            hparam_log_dict[keyword] /= (split[dataset][1] - split[dataset][0]) * rot_num

    print(colored(hparam_log_dict, "green"))

    return hparam_log_dict


def calc_error_N(outputs, targets):
    outputs = -outputs.permute(0, 2, 1).reshape(-1, 1)
    targets = targets.reshape(-1, 3)[:, 2:3]
    with_normals = targets.sum(dim=1).abs() > 0.0

    grad_loss = ((outputs[with_normals].norm(2, dim=-1) - 1)**2).mean()
    normal_loss = (outputs - targets)[with_normals].abs().norm(2, dim=1).mean()

    return grad_loss * 0.0 + normal_loss


def calc_knn_acc(preds, carn_verts, labels, pick_num):

    N_knn_full = labels.shape[1]
    preds = preds.permute(0, 2, 1).reshape(-1, 3)
    labels = labels.permute(0, 2, 1).reshape(-1, N_knn_full)
    labels = labels[:, :pick_num]

    dist = torch.cdist(preds, carn_verts, p=2)
    knn = dist.topk(k=pick_num, dim=1, largest=False)[1]
    cat_mat = torch.sort(torch.cat((knn, labels), dim=1))[0]
    bool_col = torch.zeros_like(cat_mat)[:, 0]
    for i in range(pick_num * 2 - 1):
        bool_col += cat_mat[:, i] == cat_mat[:, i + 1]
    acc = (bool_col > 0).sum() / len(bool_col)

    return acc


def calc_acc_seg(output, target, num_multiseg):
    from pytorch_lightning.metrics import Accuracy
    return Accuracy()(output.reshape(-1, num_multiseg).cpu(), target.flatten().cpu())


def add_watermark(imgs, titles):

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (350, 50)
    bottomRightCornerOfText = (800, 50)
    fontScale = 1
    fontColor = (1.0, 1.0, 1.0)
    lineType = 2

    for i in range(len(imgs)):

        title = titles[i + 1]
        cv2.putText(imgs[i], title, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        if i == 0:
            cv2.putText(
                imgs[i], str(titles[i][0]), bottomRightCornerOfText, font, fontScale, fontColor,
                lineType
            )

    result = np.concatenate(imgs, axis=0).transpose(2, 0, 1)

    return result


def make_test_gif(img_dir):

    if img_dir is not None and len(os.listdir(img_dir)) > 0:
        for dataset in os.listdir(img_dir):
            for subject in sorted(os.listdir(osp.join(img_dir, dataset))):
                img_lst = []
                im1 = None
                for file in sorted(os.listdir(osp.join(img_dir, dataset, subject))):
                    if file[-3:] not in ['obj', 'gif']:
                        img_path = os.path.join(img_dir, dataset, subject, file)
                        if im1 == None:
                            im1 = Image.open(img_path)
                        else:
                            img_lst.append(Image.open(img_path))

                print(os.path.join(img_dir, dataset, subject, "out.gif"))
                im1.save(
                    os.path.join(img_dir, dataset, subject, "out.gif"),
                    save_all=True,
                    append_images=img_lst,
                    duration=500,
                    loop=0
                )


def export_cfg(logger, cfg):

    cfg_export_file = osp.join(
        logger.save_dir, logger.name, f"version_{logger.version}", "cfg.yaml"
    )

    if not osp.exists(cfg_export_file):
        os.makedirs(osp.dirname(cfg_export_file), exist_ok=True)
        with open(cfg_export_file, "w+") as file:
            _ = yaml.dump(cfg, file)
