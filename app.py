import sys
from rabit_np import RaBitModel

import numpy as np
import openmesh as om

from viewerUI import Ui_RaBitViewer
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import *

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import ctypes
import math


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        # GUI
        self.ui = Ui_RaBitViewer()
        self.ui.setupUi(self)

        # rabit object
        datar = "./rabit_data/"
        datar_file = "shape/"
        beta_weight = np.load(datar + datar_file + "pcamat.npy")
        self.maxmin = np.load(datar + datar_file + "maxmin.npy")
        if beta_weight is None:
            self.pca_dim = 10
        else:
            self.pca_dim = beta_weight.shape[0]
        self.rabit = RaBitModel()

        # initialize the sliders
        self.poseSliders = []
        self.shapeSliders = []
        for item in self.ui.__dict__:
            if 'horizontalSlider' in item:
                self.poseSliders.append(self.ui.__dict__[item])
            if 'verticalSlider' in item:
                self.shapeSliders.append(self.ui.__dict__[item])
        self.position = 0

        # set events
        for i in range(len(self.poseSliders)):
            self.poseSliders[i].setMinimum(-8)
            self.poseSliders[i].setMaximum(8)
            self.poseSliders[i].valueChanged[int].connect(self.changevalue)

        for i in range(len(self.shapeSliders)):
            self.shapeSliders[i].setMinimum(-5)
            self.shapeSliders[i].setMaximum(5)
            self.shapeSliders[i].valueChanged[int].connect(self.changevalue)
        self.ui.pushButton.clicked.connect(self.pushButton_Click)

    def loadScene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        x, y, width, height = glGetDoublev(GL_VIEWPORT)
        gluPerspective(
            45,  # field of view in degrees
            width / float(height or 1),  # aspect ratio
            .25,  # near clipping plane
            200,  # far clipping plane
        )

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        angle = 3.14 / 4 + self.position * 3.14 * 2 / 17

        scale = 1
        x0 = math.cos(angle) * scale
        y0 = math.sin(angle) * scale
        z0 = 0

        x = x0
        y = 0.5**0.5 * y0
        z = 0.5**0.5 * y0

        gluLookAt(x, y, z, 0, 0, 0, 0, 1, 0)

    def paintGL(self):

        self.loadScene()

        vertices = self.rabit.verts[self.rabit.quads, :].reshape(-1).tolist()
        mesh = om.PolyMesh(points=self.rabit.verts, face_vertex_indices=self.rabit.quads.reshape(-1, 4))
        mesh.request_vertex_normals()
        mesh.update_vertex_normals()
        normals = mesh.vertex_normals()
        normals = normals[self.rabit.quads, :]
        normals = normals.reshape(-1)

        vertex_data = (ctypes.c_float * len(vertices))(*(vertices))
        vertex_size = normal_size = len(vertices) * 4
        vertex_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_size, vertex_data, GL_STATIC_DRAW)

        normal_data = (ctypes.c_float * len(normals))(*(normals))

        normal_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, normal_vbo)
        glBufferData(GL_ARRAY_BUFFER, normal_size, normal_data, GL_STATIC_DRAW)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, normal_vbo)
        glNormalPointer(GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_QUADS, 0, len(vertices) // 3)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        # self.setupRC()

    def setupViewer(self):
        self.ui.openGLWidget.initializeGL()
        self.ui.openGLWidget.paintGL = self.paintGL
        timer = QTimer(self)
        timer.timeout.connect(self.ui.openGLWidget.update)
        timer.start(5)

    def setupRC(self):
        # Light values and coordinates
        ambientLight = [0.4, 0.4, 0.4, 1.0]
        diffuseLight = [0.7, 0.7, 0.7, 1.0]
        specular = [0.9, 0.9, 0.9, 1.0]
        lightPos = [-0.03, 0.15, 0.1, 1.0]
        specref = [0.6, 0.6, 0.6, 1.0]

        glEnable(GL_DEPTH_TEST)    # Hidden surface removal
        glEnable(GL_CULL_FACE)    # Do not calculate inside of solid object
        glFrontFace(GL_CCW)

        glEnable(GL_LIGHTING)

        # Setup light 0
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientLight)
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight)
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular)

        # Position and turn on the light
        glLightfv(GL_LIGHT0, GL_POSITION, lightPos)
        glEnable(GL_LIGHT0)

        # Enable color tracking
        glEnable(GL_COLOR_MATERIAL)

        # Set Material properties to follow glColor values
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)

        # All materials hereafter have full specular reflectivity with a moderate shine
        glMaterialfv(GL_FRONT, GL_SPECULAR, specref)
        glMateriali(GL_FRONT, GL_SHININESS, 64)

    def changevalue(self):
        pose_vals = []
        shape_vals = []
        for i in range(23):
            vector = np.zeros(3)
            vector[0] = 3 * self.poseSliders[3 * i + 0].value() / 8
            vector[1] = 3 * self.poseSliders[3 * i + 1].value() / 8
            vector[2] = 3 * self.poseSliders[3 * i + 2].value() / 8
            pose_vals.append(vector)

        self.position = self.poseSliders[-1].value()
        pose_vals = np.array(pose_vals)
        for item in self.shapeSliders:
            shape_vals.append(item.value())

        shapeval_list = []
        for i in range(10):
            max1, min1 = self.maxmin[0][i], self.maxmin[1][i]
            d = (max1 - min1) / 11
            avg = (max1 + min1) / 2
            shapeval_list.append(shape_vals[i] * d + avg)

        if self.pca_dim > 10:
            shape_vals = np.append(shape_vals, [0] * (self.pca_dim - 10))
        trans_vals = np.zeros(3)
        self.rabit.set_params_UI(pose=pose_vals, beta=shape_vals, trans=trans_vals)

    # save the model
    def pushButton_Click(self):
        print("not support for now")
        output_path = self.ui.lineEdit.text()
        new_mesh = om.PolyMesh(points=self.rabit.verts, face_vertex_indices=self.rabit.quads.reshape(-1, 4))
        om.write_mesh(output_path, new_mesh)
        print('Saved to %s' % output_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.setupViewer()
    w.show()
    sys.exit(app.exec_())
