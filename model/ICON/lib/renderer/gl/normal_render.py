
import numpy as np
import math

from .framework import *
from .norm_render import NormRender


class NormalRender(NormRender):
    def __init__(self, width=1600, height=1200, name='Normal Renderer'):
        NormRender.__init__(self, width, height, name, program_files=['normal.vs', 'normal.fs'])

        self.norm_buffer = glGenBuffers(1)

        self.norm_data = None

    def set_normal_mesh(self, vertices, faces, norms, face_normals):
        NormRender.set_mesh(self, vertices, faces)

        self.norm_data = norms[face_normals.reshape([-1])]

        glBindBuffer(GL_ARRAY_BUFFER, self.norm_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.norm_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def euler_to_rot_mat(self, r_x, r_y, r_z):
        R_x = np.array(
            [[1, 0, 0], [0, math.cos(r_x), -math.sin(r_x)], [0, math.sin(r_x),
                                                             math.cos(r_x)]]
        )

        R_y = np.array(
            [[math.cos(r_y), 0, math.sin(r_y)], [0, 1, 0], [-math.sin(r_y), 0,
                                                            math.cos(r_y)]]
        )

        R_z = np.array(
            [[math.cos(r_z), -math.sin(r_z), 0], [math.sin(r_z), math.cos(r_z), 0], [0, 0, 1]]
        )

        R = np.dot(R_z, np.dot(R_y, R_x))

        return R

    def draw(self):
        self.draw_init()

        glUseProgram(self.program)
        glUniformMatrix4fv(self.model_mat_unif, 1, GL_FALSE, self.model_view_matrix.transpose())
        glUniformMatrix4fv(self.persp_mat_unif, 1, GL_FALSE, self.projection_matrix.transpose())

        # Handle vertex buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, self.vertex_dim, GL_DOUBLE, GL_FALSE, 0, None)

        # Handle normal buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.norm_buffer)

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, 0, None)

        glDrawArrays(GL_TRIANGLES, 0, self.n_vertices)

        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glUseProgram(0)

        self.draw_end()
