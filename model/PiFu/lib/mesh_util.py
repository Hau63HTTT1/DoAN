from skimage import measure
import numpy as np
import torch
from .sdf import create_grid, eval_grid_octree, eval_grid
from skimage import measure


def reconstruction(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=10000, transform=None):
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, net.num_views, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        net.query(samples, calib_tensor)
        pred = net.get_preds()[0][0]
        pred_np = pred.detach().cpu().numpy()
        if np.random.random() < 0.01:
            print(f"Raw network output range: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
        
        if np.allclose(pred_np, 0) or np.allclose(pred_np, 1):
            print("Warning: Network is predicting all same values!")
        

        pred_np = pred_np - 0.5
        
    
        if np.random.random() < 0.01:
            print(f"Transformed prediction range: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
        
        return pred_np


    if use_octree:
        print("Using octree evaluation...")
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        print("Using regular grid evaluation...")
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)

    print(f"Final SDF shape: {sdf.shape}")
    print(f"Final SDF range: [{sdf.min():.4f}, {sdf.max():.4f}]")
    print(f"SDF unique values: {len(np.unique(sdf))}")


    try:
 
        sdf_min, sdf_max = np.min(sdf), np.max(sdf)
        print(f"SDF stats:")
        print(f"- Range: [{sdf_min:.4f}, {sdf_max:.4f}]")
        print(f"- Mean: {np.mean(sdf):.4f}")
        print(f"- Std: {np.std(sdf):.4f}")
        
        if sdf_min == sdf_max:
            raise ValueError("SDF has no variation - all values are the same!")
        
        level = 0.0
        print(f"Using surface level: {level:.4f}")
        
        try:
            verts, faces, normals, values = measure.marching_cubes(sdf, level)
        except AttributeError:
        
            verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, level)
            
        print(f"Marching cubes successful: {len(verts)} vertices, {len(faces)} faces")

        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T
        return verts, faces, normals, values
    except Exception as e:
        print('error cannot marching cubes:', str(e))
        print('SDF contains NaN:', np.isnan(sdf).any())
        print('SDF contains Inf:', np.isinf(sdf).any())
        print('SDF is all same value:', np.allclose(sdf, sdf[0,0,0]))
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()
