import bpy
import os
import numpy as np
import cv2
import json
import glob

from bpy_utils import *


clean_obj_lamp_and_mesh(bpy.context)


mode = "train"
# mode = "test"

fp = "/home/matt/dev/Scene-Generator/output_data/" + mode


table = load_mesh("/home/matt/dev/Scene-Generator/data/table/table.obj")
mug = load_mesh("/home/matt/dev/Scene-Generator/data/objects/mug.obj")
banana = load_mesh("/home/matt/dev/Scene-Generator/data/objects/banana.obj")
scissors = load_mesh("/home/matt/dev/Scene-Generator/data/objects/scissors.obj")
cracker = load_mesh ("/home/matt/dev/Learning_Uncertainties/ellipse_from_object/data/cad_models/cracker_box/textured.obj")
# cube = load_mesh("/home/matt/dev/Scene-Generator/data/objects/cube.obj")
# dode = load_mesh("/home/matt/dev/Scene-Generator/data/objects/dodecahedron.obj")

objects_categories = [73, 41, 46, 76]
objects = [cracker, mug, banana, scissors]

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



WIDTH, HEIGHT = 640, 480


scene = bpy.context.scene
cam = scene.objects['Camera']
b_empty = create_camera_look_at_constraint(cam)
# cam_constraint = cam.constraints.new(type='TRACK_TO')
# cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
# cam_constraint.up_axis = 'UP_Y'
# b_empty = parent_obj_to_camera(cam)
# cam_constraint.target = b_empty


light = scene.objects["Light"]
light.location.x = 1.5
light.location.y = 0.5
light.location.z = 4.0
light.data.energy = 500.0
light.data.specular_factor = 0.25


# angles = np.linspace(0, 360, 50)
# d = 2
# x = d * 1.5 * np.cos(np.deg2rad(angles))
# y = d * np.sin(np.deg2rad(angles))
# z = 1.5 * np.ones_like(x)
# cam_positions = np.vstack((x, y, z)).T

# with open("/home/matt/dev/Scene-Generator/camera_path.obj", "r") as fin:
#     lines = fin.readlines()
#     pts = []
#     for l in lines:
#         if len(l) > 0 and l[:2] == "v ":
#             p = list(map(float, l[1:].split()))
#             pts.append(p)
#     cam_positions = np.vstack(pts)


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
bpy.context.scene.render.image_settings.color_mode ='RGBA'
bpy.context.scene.render.film_transparent = True


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

data = []
for i in range(200):
    # cam.location.x = cam_positions[i, 0]
    # cam.location.y = cam_positions[i, 1]
    # cam.location.z = cam_positions[i, 2]

    az = np.deg2rad(np.random.randint(0, 360))
    elev = np.deg2rad(np.random.randint(10, 80))
    d = 1.0 + float(np.random.randint(0, 400)) / 100
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



    # img = cv2.imread(bpy.context.scene.render.filepath)
    # cv2.rectangle(img, (int(minx), int(miny)), (int(maxx), int(maxy)), [0, 255, 0], 2)
    # cv2.imwrite(bpy.context.scene.render.filepath, img)
    # cv2.imwrite(os.path.join(fp, "rendering_%03d_mask.png" % i), img)


    img_dict = {}
    img_dict["file_name"] = bpy.context.scene.render.filepath
    img_dict["height"] = HEIGHT
    img_dict["width"] = WIDTH
    img_dict["id"] = i

    img_dict["annotations"] = detections
    data.append(img_dict)
    
    img = cv2.imread(bpy.context.scene.render.filepath, cv2.IMREAD_UNCHANGED)
    background = get_random_background(background_images_filenames, WIDTH, HEIGHT)

    img = compose_images(img, background)
    cv2.imwrite(bpy.context.scene.render.filepath, img)

    # fg = np.dstack([img.astype(float)]*3)
    # res = 0.5 * bg + 0.5 * fg
    # res = res.astype(np.uint8)
    # cv2.imwrite(os.path.join(fp, 'mixed/rendering_%03d_mixed.png' % i), res)

print("data = ", data)
with open(os.path.join(fp, "labels.json"), "w") as fout:
    json.dump(data, fout)

b = time.time()
print("done in ", b-a)

