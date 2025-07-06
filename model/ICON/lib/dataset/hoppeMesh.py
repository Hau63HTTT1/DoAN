
import numpy as np
from scipy.spatial import cKDTree
import trimesh

import logging

logging.getLogger("trimesh").setLevel(logging.ERROR)


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def save_ply(mesh_path, points, rgb):
    to_save = np.concatenate([points, rgb * 255], axis=-1)
    return np.savetxt(
        mesh_path,
        to_save,
        fmt='%.6f %.6f %.6f %d %d %d',
        comments='',
        header=(
            'ply\nformat ascii 1.0\nelement vertex {:d}\n' +
            'property float x\nproperty float y\nproperty float z\n' +
            'property uchar red\nproperty uchar green\nproperty uchar blue\n' + 'end_header'
        ).format(points.shape[0])
    )


class HoppeMesh:
    def __init__(self, verts, faces, vert_normals, face_normals):

        self.verts = verts
        self.faces = faces
        self.vert_normals = vert_normals
        self.face_normals = face_normals

        self.kd_tree = cKDTree(self.verts)
        self.len = len(self.verts)

    def query(self, points):
        dists, idx = self.kd_tree.query(points, n_jobs=1)
        dirs = points - self.verts[idx]
        signs = (dirs * self.vert_normals[idx]).sum(axis=1)
        signs = (signs > 0) * 2 - 1
        return signs * dists

    def contains(self, points):

        labels = trimesh.Trimesh(vertices=self.verts, faces=self.faces).contains(points)
        return labels

    def export(self, path):
        if self.colors is not None:
            save_obj_mesh_with_color(path, self.verts, self.faces, self.colors[:, 0:3] / 255.0)
        else:
            save_obj_mesh(path, self.verts, self.faces)

    def export_ply(self, path):
        save_ply(path, self.verts, self.colors[:, 0:3] / 255.0)

    def triangles(self):
        return self.verts[self.faces]
