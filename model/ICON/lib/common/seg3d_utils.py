
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plot_mask2D(mask, title="", point_coords=None, figsize=10, point_marker_size=5):

    H, W = mask.shape
    plt.figure(figsize=(figsize, figsize))
    if title:
        title += ", "
    plt.title("{}resolution {}x{}".format(title, H, W), fontsize=30)
    plt.ylabel(H, fontsize=30)
    plt.xlabel(W, fontsize=30)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(mask.detach(), interpolation="nearest", cmap=plt.get_cmap('gray'))
    if point_coords is not None:
        plt.scatter(
            x=point_coords[0], y=point_coords[1], color="red", s=point_marker_size, clip_on=True
        )
    plt.xlim(-0.5, W - 0.5)
    plt.ylim(H - 0.5, -0.5)
    plt.show()


def plot_mask3D(
    mask=None, title="", point_coords=None, figsize=1500, point_marker_size=8, interactive=True
):

    import trimesh
    import vtkplotter
    from skimage import measure

    vp = vtkplotter.Plotter(title=title, size=(figsize, figsize))
    vis_list = []

    if mask is not None:
        mask = mask.detach().to("cpu").numpy()
        mask = mask.transpose(2, 1, 0)

        verts, faces, normals, values = measure.marching_cubes_lewiner(
            mask, 0.5, gradient_direction='ascent'
        )

        mesh = trimesh.Trimesh(verts, faces)
        mesh.visual.face_colors = [200, 200, 250, 100]
        vis_list.append(mesh)

    if point_coords is not None:
        point_coords = torch.stack(point_coords, 1).to("cpu").numpy()



        pc = vtkplotter.Points(point_coords, r=point_marker_size, c='red')
        vis_list.append(pc)

    vp.show(*vis_list, bg="white", axes=1, interactive=interactive, azimuth=30, elevation=30)


def create_grid3D(min, max, steps):
    if type(min) is int:
        min = (min, min, min)
    if type(max) is int:
        max = (max, max, max)
    if type(steps) is int:
        steps = (steps, steps, steps)
    arrangeX = torch.linspace(min[0], max[0], steps[0]).long()
    arrangeY = torch.linspace(min[1], max[1], steps[1]).long()
    arrangeZ = torch.linspace(min[2], max[2], steps[2]).long()
    gridD, girdH, gridW = torch.meshgrid([arrangeZ, arrangeY, arrangeX])
    coords = torch.stack([gridW, girdH, gridD])
    coords = coords.view(3, -1).t()
    return coords


def create_grid2D(min, max, steps):
    if type(min) is int:
        min = (min, min)
    if type(max) is int:
        max = (max, max)
    if type(steps) is int:
        steps = (steps, steps)
    arrangeX = torch.linspace(min[0], max[0], steps[0]).long()
    arrangeY = torch.linspace(min[1], max[1], steps[1]).long()
    girdH, gridW = torch.meshgrid([arrangeY, arrangeX])
    coords = torch.stack([gridW, girdH])
    coords = coords.view(2, -1).t()
    return coords


class SmoothConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size for smooth_conv must be odd: {3, 5, ...}"
        self.padding = (kernel_size - 1) // 2

        weight = torch.ones(
            (in_channels, out_channels, kernel_size, kernel_size), dtype=torch.float32
        ) / (kernel_size**2)
        self.register_buffer('weight', weight)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=self.padding)


class SmoothConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size for smooth_conv must be odd: {3, 5, ...}"
        self.padding = (kernel_size - 1) // 2

        weight = torch.ones(
            (in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype=torch.float32
        ) / (kernel_size**3)
        self.register_buffer('weight', weight)

    def forward(self, input):
        return F.conv3d(input, self.weight, padding=self.padding)


def build_smooth_conv3D(in_channels=1, out_channels=1, kernel_size=3, padding=1):
    smooth_conv = torch.nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding
    )
    smooth_conv.weight.data = torch.ones(
        (in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype=torch.float32
    ) / (kernel_size**3)
    smooth_conv.bias.data = torch.zeros(out_channels)
    return smooth_conv


def build_smooth_conv2D(in_channels=1, out_channels=1, kernel_size=3, padding=1):
    smooth_conv = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding
    )
    smooth_conv.weight.data = torch.ones(
        (in_channels, out_channels, kernel_size, kernel_size), dtype=torch.float32
    ) / (kernel_size**2)
    smooth_conv.bias.data = torch.zeros(out_channels)
    return smooth_conv


def get_uncertain_point_coords_on_grid3D(uncertainty_map, num_points, **kwargs):

    R, _, D, H, W = uncertainty_map.shape

    num_points = min(D * H * W, num_points)
    point_scores, point_indices = torch.topk(
        uncertainty_map.view(R, D * H * W), k=num_points, dim=1
    )
    point_coords = torch.zeros(R, num_points, 3, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = (point_indices % W).to(torch.float)
    point_coords[:, :, 1] = (point_indices % (H * W) // W).to(torch.float)
    point_coords[:, :, 2] = (point_indices // (H * W)).to(torch.float)
    print(f"resolution {D} x {H} x {W}", point_scores.min(), point_scores.max())
    return point_indices, point_coords


def get_uncertain_point_coords_on_grid3D_faster(uncertainty_map, num_points, clip_min):
    R, _, D, H, W = uncertainty_map.shape

    assert R == 1, "batchsize > 1 is not implemented!"
    uncertainty_map = uncertainty_map.view(D * H * W)
    indices = (uncertainty_map >= clip_min).nonzero().squeeze(1)
    num_points = min(num_points, indices.size(0))
    point_scores, point_indices = torch.topk(uncertainty_map[indices], k=num_points, dim=0)
    point_indices = indices[point_indices].unsqueeze(0)

    point_coords = torch.zeros(R, num_points, 3, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = (point_indices % W).to(torch.float)
    point_coords[:, :, 1] = (point_indices % (H * W) // W).to(torch.float)
    point_coords[:, :, 2] = (point_indices // (H * W)).to(torch.float)
    return point_indices, point_coords


def get_uncertain_point_coords_on_grid2D(uncertainty_map, num_points, **kwargs):

    R, _, H, W = uncertainty_map.shape

    num_points = min(H * W, num_points)
    point_scores, point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.long, device=uncertainty_map.device)
    point_coords[:, :, 0] = (point_indices % W).to(torch.long)
    point_coords[:, :, 1] = (point_indices // W).to(torch.long)
    return point_indices, point_coords


def get_uncertain_point_coords_on_grid2D_faster(uncertainty_map, num_points, clip_min):

    R, _, H, W = uncertainty_map.shape

    assert R == 1, "batchsize > 1 is not implemented!"
    uncertainty_map = uncertainty_map.view(H * W)
    indices = (uncertainty_map >= clip_min).nonzero().squeeze(1)
    num_points = min(num_points, indices.size(0))
    point_scores, point_indices = torch.topk(uncertainty_map[indices], k=num_points, dim=0)
    point_indices = indices[point_indices].unsqueeze(0)

    point_coords = torch.zeros(R, num_points, 2, dtype=torch.long, device=uncertainty_map.device)
    point_coords[:, :, 0] = (point_indices % W).to(torch.long)
    point_coords[:, :, 1] = (point_indices // W).to(torch.long)
    return point_indices, point_coords


def calculate_uncertainty(logits, classes=None, balance_value=0.5):

    if logits.shape[1] == 1:
        gt_class_logits = logits
    else:
        gt_class_logits = logits[torch.arange(logits.shape[0], device=logits.device),
                                 classes].unsqueeze(1)
    return -torch.abs(gt_class_logits - balance_value)
