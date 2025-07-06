import numpy as np
import torch
import torch.nn.functional as F
import math
from collections import OrderedDict
import os
from scipy.ndimage import morphology
import PIL.Image as pil_img
from skimage.io import imsave
import cv2
import pickle




def generate_triangles(h, w, mask=None):
    '''
    quad layout:
        0 1 ... w-1
        w w+1
        .
        w*h
    '''
    triangles = []
    margin = 0
    for x in range(margin, w - 1 - margin):
        for y in range(margin, h - 1 - margin):
            triangle0 = [y * w + x, y * w + x + 1, (y + 1) * w + x]
            triangle1 = [y * w + x + 1, (y + 1) * w + x + 1, (y + 1) * w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:, [0, 2, 1]]
    return triangles


def face_vertices(vertices, faces):

    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + \
        (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    return vertices[faces.long()]


def vertex_normals(vertices, faces):
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None
                                                                         ]    # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    normals.index_add_(
        0, faces[:, 1].long(),
        torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]
        )
    )
    normals.index_add_(
        0, faces[:, 2].long(),
        torch.cross(
            vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]
        )
    )
    normals.index_add_(
        0, faces[:, 0].long(),
        torch.cross(
            vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]
        )
    )

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    return normals


def batch_orth_proj(X, camera):

    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:, :, 2:]], 2)
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn


DIM_FLIP = np.array([1, -1, -1], dtype=np.float32)
DIM_FLIP_TENSOR = torch.tensor([1, -1, -1], dtype=torch.float32)


def flip_pose(pose_vector, pose_format='rot-mat'):
    if pose_format == 'aa':
        if torch.is_tensor(pose_vector):
            dim_flip = DIM_FLIP_TENSOR
        else:
            dim_flip = DIM_FLIP
        return (pose_vector.reshape(-1, 3) * dim_flip).reshape(-1)
    elif pose_format == 'rot-mat':
        rot_mats = pose_vector.reshape(-1, 9).clone()

        rot_mats[:, [1, 2, 3, 6]] *= -1
        return rot_mats.view_as(pose_vector)
    else:
        raise ValueError(f'Unknown rotation format: {pose_format}')



def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)

    gauss = torch.stack([torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()


def get_gaussian_kernel(kernel_size: int, sigma: float):

    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )
    window_1d = gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(kernel_size, sigma):

    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError("kernel_size must be a tuple of length two. Got {}".format(kernel_size))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}".format(sigma))
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


def gaussian_blur(x, kernel_size=(5, 5), sigma=(1.3, 1.3)):
    b, c, h, w = x.shape
    kernel = get_gaussian_kernel2d(kernel_size, sigma).to(x.device).to(x.dtype)
    kernel = kernel.repeat(c, 1, 1, 1)
    padding = [(k - 1) // 2 for k in kernel_size]
    return F.conv2d(x, kernel, padding=padding, stride=1, groups=c)


def _compute_binary_kernel(window_size):

    window_range = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


def median_blur(x, kernel_size=(3, 3)):
    b, c, h, w = x.shape
    kernel = _compute_binary_kernel(kernel_size).to(x.device).to(x.dtype)
    kernel = kernel.repeat(c, 1, 1, 1)
    padding = [(k - 1) // 2 for k in kernel_size]
    features = F.conv2d(x, kernel, padding=padding, stride=1, groups=c)
    features = features.view(b, c, -1, h, w)
    median = torch.median(features, dim=2)[0]
    return median


def get_laplacian_kernel2d(kernel_size: int):

    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}".format(kernel_size))

    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size**2
    kernel_2d: torch.Tensor = kernel
    return kernel_2d


def laplacian(x):
    b, c, h, w = x.shape
    kernel_size = 3
    kernel = get_laplacian_kernel2d(kernel_size).to(x.device).to(x.dtype)
    kernel = kernel.repeat(c, 1, 1, 1)
    padding = (kernel_size - 1) // 2
    return F.conv2d(x, kernel, padding=padding, stride=1, groups=c)




def copy_state_dict(cur_state_dict, pre_state_dict, prefix='', load_name=None):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        if load_name is not None:
            if load_name not in k:
                continue
        v = _get_params(k)
        try:
            if v is None:
                continue
            cur_state_dict[k].copy_(v)
        except:
            continue


def dict2obj(d):

    if not isinstance(d, dict):
        return d

    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o





def remove_module(state_dict):

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]   
        new_state_dict[name] = v
    return new_state_dict


def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]]
    return image.astype(np.uint8).copy()


def dict_tensor2npy(tensor_dict):
    npy_dict = {}
    for key in tensor_dict:
        npy_dict[key] = tensor_dict[key][0].cpu().numpy()
    return npy_dict


def load_config(cfg_file):
    import yaml
    with open(cfg_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def move_dict_to_device(dict, device, tensor2float=False):
    for k, v in dict.items():
        if isinstance(v, torch.Tensor):
            if tensor2float:
                dict[k] = v.float().to(device)
            else:
                dict[k] = v.to(device)


def write_obj(
    obj_name,
    vertices,
    faces,
    colors=None,
    texture=None,
    uvcoords=None,
    uvfaces=None,
    inverse_face_order=False,
    normal_map=None,
):

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
    mtl_name = obj_name.replace('.obj', '.mtl')
    texture_name = obj_name.replace('.obj', '.png')
    material_name = 'FaceTexture'

    faces = faces.copy()
 
    faces += 1
    if inverse_face_order:
        faces = faces[:, [2, 1, 0]]
        if uvfaces is not None:
            uvfaces = uvfaces[:, [2, 1, 0]]

 
    with open(obj_name, 'w') as f:
        if texture is not None:
            f.write('mtllib %s\n\n' % os.path.basename(mtl_name))


        if colors is None:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        else:
            for i in range(vertices.shape[0]):
                f.write(
                    'v {} {} {} {} {} {}\n'.format(
                        vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1],
                        colors[i, 2]
                    )
                )

  
        if texture is None:
            for i in range(faces.shape[0]):
                f.write('f {} {} {}\n'.format(faces[i, 0], faces[i, 1], faces[i, 2]))
        else:
            for i in range(uvcoords.shape[0]):
                f.write('vt {} {}\n'.format(uvcoords[i, 0], uvcoords[i, 1]))
            f.write('usemtl %s\n' % material_name)
     
            uvfaces = uvfaces + 1
            for i in range(faces.shape[0]):
                f.write(
                    'f {}/{} {}/{} {}/{}\n'.format(
                        faces[i, 0], uvfaces[i, 0], faces[i, 1], uvfaces[i, 1], faces[i, 2],
                        uvfaces[i, 2]
                    )
                )
        
            with open(mtl_name, 'w') as f:
                f.write('newmtl %s\n' % material_name)
                s = 'map_Kd {}\n'.format(os.path.basename(texture_name))    # map to image
                f.write(s)

                if normal_map is not None:
                    if torch.is_tensor(normal_map):
                        normal_map = normal_map.detach().cpu().numpy().squeeze()

                    normal_map = np.transpose(normal_map, (1, 2, 0))
                    name, _ = os.path.splitext(obj_name)
                    normal_name = f'{name}_normals.png'
                    f.write(f'disp {normal_name}')

                    out_normal_map = normal_map / (
                        np.linalg.norm(normal_map, axis=-1, keepdims=True) + 1e-9
                    )
                    out_normal_map = (out_normal_map + 1) * 0.5

                    cv2.imwrite(normal_name, (out_normal_map * 255).astype(np.uint8)[:, :, ::-1])

            cv2.imwrite(texture_name, texture)


def save_pkl(savepath, params, ind=0):
    out_data = {}
    for k, v in params.items():
        if torch.is_tensor(v):
            out_data[k] = v[ind].detach().cpu().numpy()
        else:
            out_data[k] = v
  
    with open(savepath, 'wb') as f:
        pickle.dump(out_data, f, protocol=2)





def load_obj(obj_filename):

    with open(obj_filename, 'r') as f:
        lines = [line.strip() for line in f]

    verts, uvcoords = [], []
    faces, uv_faces = [], []

    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("v "):  
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("vt "):    # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            uvcoords.append(tx)
        elif line.startswith("f "):   
    
            face = tokens[1:]
            face_list = [f.split("/") for f in face]
            for vert_props in face_list:
       
                faces.append(int(vert_props[0]))
                if len(vert_props) > 1:
                    if vert_props[1] != "":
     
                        uv_faces.append(int(vert_props[1]))

    verts = torch.tensor(verts, dtype=torch.float32)
    uvcoords = torch.tensor(uvcoords, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.long)
    faces = faces.reshape(-1, 3) - 1
    uv_faces = torch.tensor(uv_faces, dtype=torch.long)
    uv_faces = uv_faces.reshape(-1, 3) - 1
    return (verts, uvcoords, faces, uv_faces)


# ---------------------------------- visualization
def draw_rectangle(img, bbox, bbox_color=(255, 255, 255), thickness=3, is_opaque=False, alpha=0.5):


    output = img.copy()
    if not is_opaque:
        cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, thickness)
    else:
        overlay = img.copy()

        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, -1)
    

    return output


def plot_bbox(image, bbox):

    image = cv2.rectangle(
        image.copy(), (bbox[1], bbox[0]), (bbox[3], bbox[2]), [0, 255, 0], thickness=3
    )
    return image


end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1


def plot_kpts(image, kpts, color='r'):

    kpts = kpts.copy().astype(np.int32)
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    image = image.copy()
    kpts = kpts.copy()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        if kpts.shape[1] == 4:
            if kpts[i, 3] > 0.5:
                c = (0, 255, 0)
            else:
                c = (0, 0, 255)
        image = cv2.circle(image, (st[0], st[1]), 1, c, 2)
        if i in end_list:
            continue
        ed = kpts[i + 1, :2]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)

    return image


def plot_verts(image, kpts, color='r'):

    kpts = kpts.copy().astype(np.int32)
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (0, 0, 255)
    elif color == 'y':
        c = (0, 255, 255)
    image = image.copy()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        image = cv2.circle(image, (st[0], st[1]), 1, c, 5)

    return image


def tensor_vis_landmarks(images, landmarks, gt_landmarks=None, color='g', isScale=True):
    vis_landmarks = []
    images = images.cpu().numpy()
    predicted_landmarks = landmarks.detach().cpu().numpy()
    if gt_landmarks is not None:
        gt_landmarks_np = gt_landmarks.detach().cpu().numpy()
    for i in range(images.shape[0]):
        image = images[i]
        image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]].copy()
        image = (image * 255)
        if isScale:
            predicted_landmark = predicted_landmarks[i] * \
                image.shape[0]/2 + image.shape[0]/2
        else:
            predicted_landmark = predicted_landmarks[i]
        if predicted_landmark.shape[0] == 68:
            image_landmarks = plot_kpts(image, predicted_landmark, color)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(
                    image_landmarks, gt_landmarks_np[i] * image.shape[0] / 2 + image.shape[0] / 2,
                    'r'
                )
        else:
            image_landmarks = plot_verts(image, predicted_landmark, color)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(
                    image_landmarks, gt_landmarks_np[i] * image.shape[0] / 2 + image.shape[0] / 2,
                    'r'
                )
        vis_landmarks.append(image_landmarks)

    vis_landmarks = np.stack(vis_landmarks)
    vis_landmarks = torch.from_numpy(
        vis_landmarks[:, :, :, [2, 1, 0]].transpose(0, 3, 1, 2)
    ) / 255.   
    return vis_landmarks
