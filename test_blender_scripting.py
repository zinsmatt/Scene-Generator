import bpy
import os
import numpy as np
import cv2


def get_calibration_matrix_K_from_blender(cam):
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
    name = os.path.splitext(os.path.basename(mesh_filename))[0]
    a=bpy.ops.import_scene.obj(filepath=mesh_filename, axis_forward="Y", axis_up="Z")
    return bpy.context.collection.objects[name]


def get_mesh_vertices(mesh):
    v = mesh.data.vertices
    pts = []
    for i in range(len(v)):
        co = v[i].co
        pts.append([co.x, co.y, co.z])
    pts = np.vstack(pts)
    return pts

def get_bbox(mesh):
    pts = get_mesh_vertices(mesh)
    minx, miny, minz = np.min(pts, axis=0)
    maxx, maxy, maxz = np.max(pts, axis=0)
    return minx, miny, minz, maxx, maxy, maxz

def get_object_pose(mesh):
    m = mesh.matrix_world
    print("m = \n", m)
    M = np.eye(4, dtype=np.float)
    for i in range(4):
        for j in range(4):
            M[i, j] = m[i][j]
    return M

objs = bpy.data.objects
if "Cube" in objs:
    objs.remove(objs["Cube"], do_unlink=True)



clean_obj_lamp_and_mesh(bpy.context)


table = load_mesh("/home/matt/dev/Scene-Generator/data/table/table.obj")
mug = load_mesh("/home/matt/dev/Scene-Generator/data/objects/mug.obj")
banana = load_mesh("/home/matt/dev/Scene-Generator/data/objects/banana.obj")
scissors = load_mesh("/home/matt/dev/Scene-Generator/data/objects/scissors.obj")
cracker = load_mesh ("/home/matt/dev/Scene-Generator/data/objects/cracker.obj")
# cube = load_mesh("/home/matt/dev/Scene-Generator/data/objects/cube.obj")
# dode = load_mesh("/home/matt/dev/Scene-Generator/data/objects/dodecahedron.obj")

objects = [mug, banana, scissors, cracker]

bbox_mug = get_bbox(mug)
mug.location.z -= bbox_mug[2]
mug.location.x += 0.35
mug.location.y -= 0.2

bbox_banana = get_bbox(banana)
banana.location.z -= bbox_banana[2]
banana.location.x += 0.2
banana.location.y += 0.4


bbox_scissors = get_bbox(scissors)
scissors.location.z -= bbox_scissors[2]
scissors.location.x -= 0.2
scissors.location.y -= 0.5

bbox_cracker = get_bbox(cracker)
cracker.location.z -= bbox_cracker[2]


def parent_obj_to_camera(b_camera):
    # set the parenting to the origin
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty

    bpy.context.collection.objects.link(b_empty)
    return b_empty


WIDTH, HEIGHT = 640, 480


scene = bpy.context.scene
cam = scene.objects['Camera']
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty


light = scene.objects["Light"]
light.location.x = 1.5
light.location.y = 0.5
light.location.z = 3.2960827350616455
light.data.energy = 500.0
light.data.specular_factor = 0.5


angles = np.linspace(0, 360, 50)
d = 2
x = d * 1.5 * np.cos(np.deg2rad(angles))
y = d * np.sin(np.deg2rad(angles))
z = 1.5 * np.ones_like(x)
cam_positions = np.vstack((x, y, z)).T

with open("/home/matt/dev/Scene-Generator/camera_path.obj", "r") as fin:
    lines = fin.readlines()
    pts = []
    for l in lines:
        if len(l) > 0 and l[:2] == "v ":
            p = list(map(float, l[1:].split()))
            pts.append(p)
    cam_positions = np.vstack(pts)
print(cam_positions)


fp = "out/"

# configure Cycles engine and use GPU
bpy.context.scene.render.engine = "CYCLES"
bpy.context.preferences.addons['cycles'].preferences.get_devices()
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = "GPU"
bpy.context.scene.cycles.feature_set = "SUPPORTED"
bpy.context.preferences.addons['cycles'].preferences.devices[0].use = True


scene = bpy.context.scene
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.resolution_percentage = 100


K = get_calibration_matrix_K_from_blender(cam)


# mug points
bpy.context.view_layer.update()

all_pts_world_h = []
for o in objects:
    pts = get_mesh_vertices(o)
    pts_h = np.vstack((pts.T, np.ones((1, pts.shape[0]))))
    pts_world_h = get_object_pose(o) @ pts_h
    all_pts_world_h.append(pts_world_h.T)
all_pts_world_h = np.vstack(all_pts_world_h).T

import time
a = time.time()
for i in range(20):
    # cam.location.x = cam_positions[i, 0]
    # cam.location.y = cam_positions[i, 1]
    # cam.location.z = cam_positions[i, 2]

    az = np.deg2rad(np.random.randint(0, 360))
    elev = np.deg2rad(np.random.randint(10, 80))
    d = 1.0
    cam.location.x = d * np.cos(az) * np.cos(elev)
    cam.location.y = d * np.sin(az) * np.cos(elev)
    cam.location.z = d * np.sin(elev)

    b_empty.location.x = (np.random.rand() - 0.5) / 2
    b_empty.location.y = (np.random.rand() - 0.5) / 2

    bpy.context.view_layer.update()
    bpy.context.scene.render.filepath = os.path.join(fp, 'rendering_%03d.png' % i)
    bpy.ops.render.render(write_still=True)

    # transform points to cam
    cam_pose = get_object_pose(cam)
    correct = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
    cam_pose =  cam_pose @ correct
    T_to_cam = np.linalg.inv(cam_pose)
    pts_cam_h = T_to_cam @ all_pts_world_h

    # project points on image
    p_proj = K @ pts_cam_h[:3, :]
    p_proj /= p_proj[2, :]
    uvs = np.round(p_proj).astype(int)
    uvs[0, :] = np.clip(uvs[0, :], 0, WIDTH-1)
    uvs[1, :] = np.clip(uvs[1, :], 0, HEIGHT-1)

    img = np.zeros((HEIGHT, WIDTH), np.uint8)
    img[uvs[1, :], uvs[0, :]] = 255
    # cv2.imwrite(os.path.join(fp, "rendering_%03d_mask.png" % i), img)

    # bg = cv2.imread(os.path.join(fp, 'rendering_%03d.png' % i)).astype(float)
    # fg = np.dstack([img.astype(float)]*3)
    # res = 0.5 * bg + 0.5 * fg
    # res = res.astype(np.uint8)
    # cv2.imwrite(os.path.join(fp, 'mixed/rendering_%03d_mixed.png' % i), res)


b = time.time()
print("done in ", b-a)

print("banana = \n", get_object_pose(banana))
