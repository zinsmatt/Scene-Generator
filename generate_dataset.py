#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""


import bpy
from bpy_utils import * 
from mesh_io import read_obj
import numpy as np
import json

def get_object(name):
    return bpy.context.collection.objects[name]

def create_if_needed(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)



bpy.ops.wm.open_mainfile(filepath="scene.blend")


objects_names = ["airplane", "banana", "can", "cleaner", "cracker", "driller", "orange", "pitcher", "plate", "soccer"]
objects = []
# objects.append(get_object("cracker"))
for o in objects_names:
    objects.append(get_object(o))
    

WIDTH, HEIGHT = 1080, 768

scene = bpy.context.scene
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.resolution_percentage = 100


out_folder = "/home/mzins/dev/Scene-Generator/dataset"
mode = "train"
# mode = "test"
out_folder = os.path.join(out_folder, mode)
create_if_needed(out_folder)

file_out = os.path.join("annotations.json")

if mode =="train":
    trajectory_files = ["trajectories/traj_02.obj", "trajectories/traj_03.obj", "trajectories/traj_04.obj"]#, "trajectories/traj_05.obj", "trajectories/traj_06.obj"]
else:
    trajectory_files = ["trajectories/traj_04.obj"]
traj_points = []
for f in trajectory_files:
    pts, _ = read_obj(f)
    traj_points.append(pts[::3, :])
traj_points = np.vstack(traj_points)




scene = bpy.context.scene
cam = scene.objects['Camera']
b_empty = create_camera_look_at_constraint(cam)
b_empty.location.x = (np.random.rand() - 0.5) / 2
b_empty.location.y = (np.random.rand() - 0.5) / 2


light = scene.objects["Light"]
light.location.x = 1.5
light.location.y = 0.5
light.location.z = 3.6
light.data.energy = 500.0
light.data.specular_factor = 0.25





# configure Cycles engine and use GPU

bpy.context.scene.render.engine = "BLENDER_EEVEE"
bpy.context.scene.eevee.motion_blur_samples = 1
bpy.context.scene.eevee.volumetric_samples = 1
bpy.context.scene.eevee.taa_render_samples = 8
bpy.context.scene.eevee.taa_samples = 1
# bpy.context.scene.render.engine = "CYCLES"
# bpy.context.preferences.addons['cycles'].preferences.get_devices()
# bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
# bpy.context.scene.cycles.device = "GPU"
# bpy.context.scene.cycles.feature_set = "SUPPORTED"
# bpy.context.preferences.addons['cycles'].preferences.devices[0].use = True


scene = bpy.context.scene
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.resolution_percentage = 100
# bpy.context.scene.render.image_settings.color_mode ='RGBA'
# bpy.context.scene.render.film_transparent = True


K = get_calibration_matrix_K_from_blender(cam)


import time
a = time.time()

data = []
objects_categories = np.linspace(0, len(objects_names), len(objects_names), dtype=np.int32, endpoint=False).tolist()
for i in range(traj_points.shape[0]):
    cam.location.x = traj_points[i, 0]
    cam.location.y = traj_points[i, 1]
    cam.location.z = traj_points[i, 2]


    # b_empty.location.x = (np.random.rand() - 0.5) / 2
    # b_empty.location.y = (np.random.rand() - 0.5) / 2

    bpy.context.view_layer.update()
    bpy.context.scene.render.filepath = os.path.join(out_folder, 'rendering_%03d.png' % i)
    bpy.ops.render.render(write_still=True)
    
    img = cv2.imread(bpy.context.scene.render.filepath)
    
    # transform points to cam
    cam_pose = get_object_pose(cam)
    correct = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
    cam_pose =  cam_pose @ correct
    T_to_cam = np.linalg.inv(cam_pose)
    # pts_cam_h = T_to_cam @ all_pts_world_h

    objects_cam_points = []
    objects_cam_dists = []
    for o in objects:
        pts = get_mesh_vertices(o)
        pts_h = np.vstack((pts.T, np.ones((1, pts.shape[0]))))
        pts_world_h = get_object_pose(o) @ pts_h
        pts_cam_h = T_to_cam @ pts_world_h
        pts_cam = pts_cam_h[:3, :].T
        objects_cam_points.append(pts_cam)
        objects_cam_dists.append(np.linalg.norm(np.mean(pts_cam_h, axis=0)))

    objects_order = np.argsort(objects_cam_dists)
    buffer = np.zeros((HEIGHT, WIDTH), np.uint8)
    detections = []
    for oid in objects_order:
        cat = objects_categories[oid]
        pts = objects_cam_points[oid]
        p_proj = K @ pts.T
        p_proj /= p_proj[2, :]
    
        uvs = p_proj[:2, :]
        uvs = uvs.T

        minx, miny = np.min(np.floor(uvs), axis=0)
        maxx, maxy = np.max(np.ceil(uvs), axis=0)
        if minx < 0 or miny < 0 or maxx >= WIDTH or maxy >= HEIGHT:
            continue

        uvs[:, 0] = np.clip(uvs[:, 0], 0, WIDTH-1)
        uvs[:, 1] = np.clip(uvs[:, 1], 0, HEIGHT-1)
        minx, miny = np.min(np.floor(uvs), axis=0).astype(int)
        maxx, maxy = np.max(np.ceil(uvs), axis=0).astype(int)

        center = (int((minx + maxx) / 2), int((miny + maxy) / 2))
        if buffer[center[1], center[0]] == 0:
            annot = {}
            annot["bbox"] = list(map(int, [minx, miny, maxx, maxy]))
            annot["bbox_mode"] = 0
            annot["category_id"] = cat
            detections.append(annot)
        
        buffer[miny:maxy+1, minx:maxx+1] = 1



        cv2.rectangle(img, (int(minx), int(miny)), (int(maxx), int(maxy)), [0, 255, 0], 2)
    # cv2.imwrite(bpy.context.scene.render.filepath, img)
    cv2.imwrite("out/annot/img_%04d.png" % i, img)

    img_dict = {}
    img_dict["file_name"] = bpy.context.scene.render.filepath
    img_dict["height"] = HEIGHT
    img_dict["width"] = WIDTH
    img_dict["id"] = i
    img_dict["K"] = K.tolist()
    img_dict["rotation"] = T_to_cam[:3, :3].tolist()
    img_dict["translation"] = T_to_cam[:3, 3].tolist()

    img_dict["annotations"] = detections
    data.append(img_dict)
    

print("data = ", data)
with open(os.path.join(out_folder, "labels.json"), "w") as fout:
    json.dump(data, fout)

b = time.time()
print("done in ", b-a)
