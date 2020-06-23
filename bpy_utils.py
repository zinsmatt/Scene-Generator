import bpy
import numpy as np
import os
import glob
import cv2

coco_folder = "/media/matt/4eb435a5-0096-4404-9393-8d75ca083559/matthieu/coco/train2017/"
background_images_filenames = glob.glob(os.path.join(coco_folder, "*.jpg"))

def get_calibration_matrix_K_from_blender(cam):
    """
        return the K matrix from Blender
    """
    camd = cam.data
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    s_u = resolution_x_in_px * scale / sensor_width_in_mm 
    s_v = resolution_y_in_px * scale / sensor_height_in_mm
    fx = f_in_mm * s_u
    fy = f_in_mm * s_u
    ppx = resolution_x_in_px*scale / 2
    ppy = resolution_y_in_px*scale / 2
    K = np.array([[fx, 0.0, ppx],
                  [0.0, fy, ppy],
                  [0.0, 0.0, 1.0]])
    return K


def clean_obj_lamp_and_mesh(context):
    """
        Clean the scene
    """
    scene = context.scene
    objs = bpy.data.objects
    meshes = bpy.data.meshes
    for obj in objs:
        if obj.type == "MESH" or obj.type == 'LAMP':
            bpy.context.collection.objects.unlink(obj)
            objs.remove(obj)
    for mesh in meshes:
        meshes.remove(mesh)


def load_mesh(mesh_filename):
    """
        Load mesh in Blender
    """
    name = os.path.splitext(os.path.basename(mesh_filename))[0]
    a=bpy.ops.import_scene.obj(filepath=mesh_filename, axis_forward="Y", axis_up="Z")
    return bpy.context.collection.objects[name]


def get_mesh_vertices(mesh):
    """ 
        Return the mesh vertices (Nx3 Numpy array)
    """
    v = mesh.data.vertices
    pts = []
    for i in range(len(v)):
        co = v[i].co
        pts.append([co.x, co.y, co.z])
    pts = np.vstack(pts)
    return pts

def get_bbox(mesh):
    """
        Return the mesh bbox (3D)
    """
    pts = get_mesh_vertices(mesh)
    minx, miny, minz = np.min(pts, axis=0)
    maxx, maxy, maxz = np.max(pts, axis=0)
    return minx, miny, minz, maxx, maxy, maxz


def get_object_pose(mesh):
    """
        Return the mesh pose (4x4 matrix)
    """
    m = mesh.matrix_world
    print("m = \n", m)
    M = np.eye(4, dtype=np.float)
    for i in range(4):
        for j in range(4):
            M[i, j] = m[i][j]
    return M


def parent_obj_to_camera(b_camera):
    """
        Create an empty with a parent link to camera
    """
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty
    bpy.context.collection.objects.link(b_empty)
    return b_empty


def create_camera_look_at_constraint(cam):
    """
        Force the camera to llok at an empty
        Returns this empty
    """
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty
    return b_empty


def get_random_background(filenames, W, H):
    """
        Get a random background from COCO
    """
    img = cv2.imread(filenames[np.random.randint(0, len(filenames)-1)])
    return cv2.resize(img, (W, H))

def compose_images(fg, bg):
    """
        Compose two images using alpha
    """
    mask = fg[:, :, 3]
    im = fg[:, :, :3]
    y, x = np.where(mask == 0)
    im[y, x, :] = bg[y, x, :]
    return im
