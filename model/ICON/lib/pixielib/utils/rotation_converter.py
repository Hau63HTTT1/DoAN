import torch
import torch.nn.functional as F
import numpy as np

pi = torch.Tensor([3.14159265358979323846])


def rad2deg(tensor):

    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    return 180. * tensor / pi.to(tensor.device).type(tensor.dtype)


def deg2rad(tensor):

    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.




def euler_to_quaternion(r):
    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]

    z = z / 2.0
    y = y / 2.0
    x = x / 2.0
    cz = torch.cos(z)
    sz = torch.sin(z)
    cy = torch.cos(y)
    sy = torch.sin(y)
    cx = torch.cos(x)
    sx = torch.sin(x)
    quaternion = torch.zeros_like(r.repeat(1, 2))[..., :4].to(r.device)
    quaternion[..., 0] += cx * cy * cz - sx * sy * sz
    quaternion[..., 1] += cx * sy * sz + cy * cz * sx
    quaternion[..., 2] += cx * cz * sy - sx * cy * sz
    quaternion[..., 3] += cx * cy * sz + sx * cz * sy
    return quaternion


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):

    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(rotation_matrix.shape)
        )

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1], t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2]
        ], -1
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] + rmat_t[:, 1, 0], t1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1]
        ], -1
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2
        ], -1
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [
            t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1], rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0]
        ], -1
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1.float()
    mask_c1 = mask_d2 * (1 - mask_d0_d1.float())
    mask_c2 = (1 - mask_d2.float()) * mask_d0_nd1
    mask_c3 = (1 - mask_d2.float()) * (1 - mask_d0_nd1.float())
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(
        t0_rep * mask_c0 + t1_rep * mask_c1 +    
        t2_rep * mask_c2 + t3_rep * mask_c3
    )    
    q *= 0.5
    return q


def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    
    if not torch.is_tensor(angle_axis):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(angle_axis)))

    if not angle_axis.shape[-1] == 3:
        raise ValueError(
            "Input must be a tensor of shape Nx3 or 3. Got {}".format(angle_axis.shape)
        )
    a0: torch.Tensor = angle_axis[..., 0:1]
    a1: torch.Tensor = angle_axis[..., 1:2]
    a2: torch.Tensor = angle_axis[..., 2:3]
    theta_squared: torch.Tensor = a0 * a0 + a1 * a1 + a2 * a2

    theta: torch.Tensor = torch.sqrt(theta_squared)
    half_theta: torch.Tensor = theta * 0.5

    mask: torch.Tensor = theta_squared > 0.0
    ones: torch.Tensor = torch.ones_like(half_theta)

    k_neg: torch.Tensor = 0.5 * ones
    k_pos: torch.Tensor = torch.sin(half_theta) / theta
    k: torch.Tensor = torch.where(mask, k_pos, k_neg)
    w: torch.Tensor = torch.where(mask, torch.cos(half_theta), ones)

    quaternion: torch.Tensor = torch.zeros_like(angle_axis)
    quaternion[..., 0:1] += a0 * k
    quaternion[..., 1:2] += a1 * k
    quaternion[..., 2:3] += a2 * k
    return torch.cat([w, quaternion], dim=-1)




def quaternion_to_rotation_matrix(quat):

    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy, w2 - x2 + y2 - z2,
            2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
        ],
        dim=1
    ).view(B, 3, 3)
    return rotMat


def quaternion_to_angle_axis(quaternion: torch.Tensor):
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape)
        )
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta)
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * \
        torch.ones_like(sin_theta).to(quaternion.device)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion).to(quaternion.device)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


_AXIS_TO_IND = {'x': 0, 'y': 1, 'z': 2}


def _elementary_basis_vector(axis):
    b = torch.zeros(3)
    b[_AXIS_TO_IND[axis]] = 1
    return b


def _compute_euler_from_matrix(dcm, seq='xyz', extrinsic=False):
    orig_device = dcm.device
    dcm = dcm.to('cpu')
    seq = seq.lower()

    if extrinsic:
        seq = seq[::-1]

    if dcm.ndim == 2:
        dcm = dcm[None, :, :]
    num_rotations = dcm.shape[0]

    device = dcm.device
    n1 = _elementary_basis_vector(seq[0])
    n2 = _elementary_basis_vector(seq[1])
    n3 = _elementary_basis_vector(seq[2])


    sl = torch.dot(torch.cross(n1, n2), n3)
    cl = torch.dot(n1, n3)

    offset = torch.atan2(sl, cl)
    c = torch.stack((n2, torch.cross(n1, n2), n1)).type(dcm.dtype).to(device)

  
    rot = torch.tensor([
        [1, 0, 0],
        [0, cl, sl],
        [0, -sl, cl],
    ]).type(dcm.dtype)
    res = torch.einsum('ij,...jk->...ik', c, dcm)
    dcm_transformed = torch.einsum('...ij,jk->...ik', res, c.T @ rot)


    angles = torch.zeros((num_rotations, 3), dtype=dcm.dtype, device=device)

    positive_unity = dcm_transformed[:, 2, 2] > 1
    negative_unity = dcm_transformed[:, 2, 2] < -1
    dcm_transformed[positive_unity, 2, 2] = 1
    dcm_transformed[negative_unity, 2, 2] = -1
    angles[:, 1] = torch.acos(dcm_transformed[:, 2, 2])


    eps = 1e-7
    safe1 = (torch.abs(angles[:, 1]) >= eps)
    safe2 = (torch.abs(angles[:, 1] - np.pi) >= eps)

    angles[:, 1] += offset


    safe_mask = torch.logical_and(safe1, safe2)
    angles[safe_mask,
           0] = torch.atan2(dcm_transformed[safe_mask, 0, 2], -dcm_transformed[safe_mask, 1, 2])
    angles[safe_mask,
           2] = torch.atan2(dcm_transformed[safe_mask, 2, 0], dcm_transformed[safe_mask, 2, 1])
    if extrinsic:

        angles[~safe_mask, 0] = 0

        angles[~safe1, 2] = torch.atan2(
            dcm_transformed[~safe1, 1, 0] - dcm_transformed[~safe1, 0, 1],
            dcm_transformed[~safe1, 0, 0] + dcm_transformed[~safe1, 1, 1]
        )

        angles[~safe2, 2] = -torch.atan2(
            dcm_transformed[~safe2, 1, 0] + dcm_transformed[~safe2, 0, 1],
            dcm_transformed[~safe2, 0, 0] - dcm_transformed[~safe2, 1, 1]
        )
    else:

        angles[~safe_mask, 2] = 0
  
        angles[~safe1, 0] = torch.atan2(
            dcm_transformed[~safe1, 1, 0] - dcm_transformed[~safe1, 0, 1],
            dcm_transformed[~safe1, 0, 0] + dcm_transformed[~safe1, 1, 1]
        )

        angles[~safe2, 0] = torch.atan2(
            dcm_transformed[~safe2, 1, 0] + dcm_transformed[~safe2, 0, 1],
            dcm_transformed[~safe2, 0, 0] - dcm_transformed[~safe2, 1, 1]
        )


    if seq[0] == seq[2]:

        adjust_mask = torch.logical_or(angles[:, 1] < 0, angles[:, 1] > np.pi)
    else:

        adjust_mask = torch.logical_or(angles[:, 1] < -np.pi / 2, angles[:, 1] > np.pi / 2)

    adjust_mask = torch.logical_and(adjust_mask, safe_mask)

    angles[adjust_mask, 0] += np.pi
    angles[adjust_mask, 1] = 2 * offset - angles[adjust_mask, 1]
    angles[adjust_mask, 2] -= np.pi

    angles[angles < -np.pi] += 2 * np.pi
    angles[angles > np.pi] -= 2 * np.pi

    # Step 8
    if not torch.all(safe_mask):
        print(
            "Gimbal lock detected. Setting third angle to zero since"
            "it is not possible to uniquely determine all angles."
        )

    if extrinsic:

        angles = torch.flip(angles, dims=[
            -1,
        ])

    angles = angles.to(orig_device)
    return angles





def batch_euler2axis(r):
    return quaternion_to_angle_axis(euler_to_quaternion(r))


def batch_euler2matrix(r):
    return quaternion_to_rotation_matrix(euler_to_quaternion(r))


def batch_matrix2euler(rot_mats):

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] + rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def batch_matrix2axis(rot_mats):
    return quaternion_to_angle_axis(rotation_matrix_to_quaternion(rot_mats))


def batch_axis2matrix(theta):

    return quaternion_to_rotation_matrix(angle_axis_to_quaternion(theta))


def batch_axis2euler(theta):
    return batch_matrix2euler(batch_axis2matrix(theta))


def batch_axis2euler(r):
    return rot_mat_to_euler(batch_rodrigues(r))


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def batch_cont2matrix(module_input):

    batch_size = module_input.shape[0]
    reshaped_input = module_input.reshape(-1, 3, 2)


    b1 = F.normalize(reshaped_input[:, :, 0].clone(), dim=1)

    dot_prod = torch.sum(b1 * reshaped_input[:, :, 1].clone(), dim=1, keepdim=True)

    b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=1)

    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats.view(batch_size, -1, 3, 3)
