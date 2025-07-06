from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
import numpy as np
import open3d as o3d
from OpenGL.GL import *
from OpenGL.GLU import *

class ModelViewer(QOpenGLWidget):
    mesh_loaded = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh = None
        self.rotation = [0, 0, 0]
        self.translation = [0, 0, -5]
        self.scale = 1.0
        self.last_pos = None
        self.is_rotating = False
        self.display_list = None
        self.display_list_simple = None
        self.simplified_mesh = None
        self.render_quality = 'high' 
        self.needs_display_list_update = False
        
       
        self.material_ambient = [0.2, 0.2, 0.2, 1.0]
        self.material_diffuse = [0.7, 0.7, 0.8, 1.0]
        self.material_specular = [1.0, 1.0, 1.0, 1.0]
        self.material_shininess = 50.0
        
    
        self.light_position = [1.0, 1.0, 1.0, 0.0]
        self.light_ambient = [0.2, 0.2, 0.2, 1.0]
        self.light_diffuse = [0.8, 0.8, 0.8, 1.0]
        self.light_specular = [1.0, 1.0, 1.0, 1.0]
        
    
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_rotation)
        self.update_timer.setInterval(16)  
       
        self.target_rotation = [0, 0, 0]
        self.rotation_speed = [0, 0, 0]
        self.damping = 0.95

    def update_display_lists(self):
        """Update display lists if needed"""
        if not self.needs_display_list_update or self.mesh is None:
            return

        try:
  
            if self.display_list is not None:
                glDeleteLists(self.display_list, 1)
            if self.display_list_simple is not None:
                glDeleteLists(self.display_list_simple, 1)


            vertices = np.asarray(self.mesh.vertices)
            triangles = np.asarray(self.mesh.triangles)
            normals = np.asarray(self.mesh.vertex_normals)

            self.display_list = glGenLists(1)
            glNewList(self.display_list, GL_COMPILE)
            glBegin(GL_TRIANGLES)
            for triangle in triangles:
                for vertex_id in triangle:
                    normal = normals[vertex_id]
                    glNormal3f(*normal)
                    color = [(n + 1.0) / 2.0 for n in normal]
                    glColor3f(0.5 + color[0] * 0.2, 
                             0.6 + color[1] * 0.2,
                             0.7 + color[2] * 0.2)
                    vertex = vertices[vertex_id]
                    glVertex3f(*vertex)
            glEnd()
            glEndList()

  
            if self.simplified_mesh is None:
                self.simplified_mesh = self.mesh.simplify_quadric_decimation(
                    target_number_of_triangles=len(self.mesh.triangles) // 4
                )
                self.simplified_mesh.compute_vertex_normals()

            vertices = np.asarray(self.simplified_mesh.vertices)
            triangles = np.asarray(self.simplified_mesh.triangles)
            normals = np.asarray(self.simplified_mesh.vertex_normals)

            self.display_list_simple = glGenLists(1)
            glNewList(self.display_list_simple, GL_COMPILE)
            glBegin(GL_TRIANGLES)
            for triangle in triangles:
                for vertex_id in triangle:
                    normal = normals[vertex_id]
                    glNormal3f(*normal)
                    glColor3f(0.7, 0.7, 0.8)
                    vertex = vertices[vertex_id]
                    glVertex3f(*vertex)
            glEnd()
            glEndList()

            self.needs_display_list_update = False
        except Exception as e:
            print(f"Error creating display lists: {str(e)}")
            self.display_list = None
            self.display_list_simple = None

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        
        # Set up material properties
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, self.material_ambient)
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, self.material_diffuse)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, self.material_specular)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, self.material_shininess)
        
  
        glLightfv(GL_LIGHT0, GL_POSITION, self.light_position)
        glLightfv(GL_LIGHT0, GL_AMBIENT, self.light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, self.light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, self.light_specular)
        
   
        glShadeModel(GL_SMOOTH)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):

        self.update_display_lists()

     
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glBegin(GL_QUADS)
        glColor3f(0.2, 0.2, 0.3)  
        glVertex2f(-1.0, 1.0)
        glVertex2f(1.0, 1.0)
        glColor3f(0.1, 0.1, 0.15) 
        glVertex2f(1.0, -1.0)
        glVertex2f(-1.0, -1.0)
        glEnd()
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
    
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glClear(GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
      
        glLightfv(GL_LIGHT0, GL_POSITION, self.light_position)
        

        glTranslatef(*self.translation)
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)
        glScalef(self.scale, self.scale, self.scale)

        if self.display_list is not None:
            if self.is_rotating and self.display_list_simple is not None:
                glCallList(self.display_list_simple)
            else:
                glCallList(self.display_list)
        elif self.mesh is not None:
        
            vertices = np.asarray(self.mesh.vertices)
            triangles = np.asarray(self.mesh.triangles)
            normals = np.asarray(self.mesh.vertex_normals)
            
            glBegin(GL_TRIANGLES)
            for triangle in triangles:
                for vertex_id in triangle:
                    normal = normals[vertex_id]
                    glNormal3f(*normal)
                    glColor3f(0.7, 0.7, 0.8)
                    vertex = vertices[vertex_id]
                    glVertex3f(*vertex)
            glEnd()

    def update_rotation(self):
    
        if any(abs(s) > 0.01 for s in self.rotation_speed):
            for i in range(3):
                self.rotation[i] += self.rotation_speed[i]
                self.rotation_speed[i] *= self.damping
            self.update()
        else:
            self.update_timer.stop()
            self.is_rotating = False
            self.update()

    def mousePressEvent(self, event):
        self.last_pos = event.pos()
        if event.buttons() & Qt.LeftButton:
            self.is_rotating = True

    def mouseReleaseEvent(self, event):
        if self.is_rotating:
            self.update_timer.start()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return
            
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        
        if event.buttons() & Qt.LeftButton:
            self.rotation_speed[1] = dx * 0.5
            self.rotation_speed[0] = dy * 0.5
            self.rotation[1] += self.rotation_speed[1]
            self.rotation[0] += self.rotation_speed[0]
        elif event.buttons() & Qt.RightButton:
            self.translation[0] += dx * 0.01
            self.translation[1] -= dy * 0.01
            
        self.last_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.scale *= 1.1 if delta > 0 else 0.9
        self.update()

    def load_model(self, file_path):
        """Load and display a 3D model using Open3D"""
        try:
            
            extension = file_path.lower().split('.')[-1]
            
            if extension == 'obj':
                self.mesh = o3d.io.read_triangle_mesh(file_path)
            elif extension == 'stl':
                self.mesh = o3d.io.read_triangle_mesh(file_path)
            elif extension == 'ply':
                self.mesh = o3d.io.read_triangle_mesh(file_path)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
            
            if not self.mesh.has_vertex_normals():
                self.mesh.compute_vertex_normals()
                
        
            center = self.mesh.get_center()
            self.mesh.translate(-center)
            
            
            bbox = self.mesh.get_axis_aligned_bounding_box()
            scale = 2.0 / np.max(bbox.get_extent())
            self.mesh.scale(scale, center=[0, 0, 0])
            
       
            self.rotation = [30, 30, 0]  
            self.translation = [0, 0, -5]
            self.scale = 1.0
            self.rotation_speed = [0, 0, 0]
            
           
            self.needs_display_list_update = True
            self.simplified_mesh = None
            
            self.update()
            
            
            self.mesh_loaded.emit(True)
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            
            
            self.mesh_loaded.emit(False)
            
            return False

    def clear_model(self):
     
        self.mesh = None
        self.simplified_mesh = None
        if self.display_list is not None:
            glDeleteLists(self.display_list, 1)
            self.display_list = None
        if self.display_list_simple is not None:
            glDeleteLists(self.display_list_simple, 1)
            self.display_list_simple = None
        self.update() 