import bpy
import sys
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot
from ellipsoids_3D_reconstruction import draw_ellipse_from_mat
from ellipsoid_tools import ellipsoid_phys_to_alg_mat
from ellipse_tools import decompose_ellipse
from mesh_io import read_obj, write_obj
import h5py
import matplotlib.pyplot as plt
from ellcv.types import Ellipse, Ellipsoid
from ellcv.visu import draw_ellipse

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

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty
    bpy.context.collection.objects.link(b_empty)
    return b_empty

def create_if_needed(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)


objs = bpy.data.objects
if "Cube" in objs:
    objs.remove(objs["Cube"], do_unlink=True)
clean_obj_lamp_and_mesh(bpy.context)



RENDER_ENGINE = "BLENDER_EEVEE"
# RENDER_ENGINE = "CYCLES"


USE_IMAGE_AS_BACKGROUND = False
MASK_BACKGROUND = False
BACKGROUND_COLOR = (0.0, 1.0, 0.0)

# output_folder = "/media/mzins/DATA1/single_object/driller_background_1"
output_folder = "/media/mzins/DATA1/single_object/driller_background_2"
output_folder = "/media/mzins/DATA1/single_object/cracker_background_1"
# output_folder = "/media/mzins/DATA1/single_object/cracker_background_2"

DO_CROP = True
#mode = "train"
# mode = "test"
for mode in ["train", "test"]:
    output_images_folder = os.path.join(output_folder, mode)
    output_masks_folder = os.path.join(output_folder, mode + "_masks")
    output_debug_folder = os.path.join(output_folder, "dbg")
    output_labels_folder = output_folder
    output_labels_name = "annotations_" + mode + ".h5"
    output_labels_cam_name = "cam_info_" + mode + ".h5"
    output_labels_box_name = "box_info_" + mode + ".h5"


    ELLIPSOID_MATRIX = "experiment_independence_background/ellipsoid_driller.txt"
    ELLIPSOID_MATRIX = "experiment_independence_background/ellipsoid_cracker.txt"
    # UNIT_SPHERE = "/home/mzins/dev/Learning_Uncertainties/ellipse_from_object/basic_shape/unit_sphere.obj"

    create_if_needed(output_images_folder)
    create_if_needed(output_masks_folder)
    create_if_needed(output_debug_folder)

    # background
    #bg_folder = "experiment_independence_background/scene/driller_background_1"
    # bg_folder = "experiment_independence_background/scene/driller_background_2"
    bg_folder = "experiment_independence_background/scene/cracker_background_1"
    # bg_folder = "experiment_independence_background/scene/cracker_background_2"

    # Main object
    # driller = load_mesh (os.path.join(bg_folder, "driller.obj"))
    cracker = load_mesh (os.path.join(bg_folder, "cracker.obj"))

    # # Other objects and background
    # bg_objects = ["table", "scissors", "background", "can", "cleaner", "cracker"]
    # bg_objects = ["floor", "banana", "soccer", "plate", "orange", "airplane", "pitcher", "Linemod_background"]

    bg_objects = ["table", "scissors", "background", "can", "cleaner", "driller"]
    # bg_objects = ["floor", "banana", "soccer", "plate", "orange", "airplane", "pitcher", "Linemod_background"]


    for obj in bg_objects:
        load_mesh(os.path.join(bg_folder, obj + ".obj"))



    # objects = [driller]
    objects = [cracker]


    WIDTH, HEIGHT = 640, 480


    scene = bpy.context.scene
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.color_mode ='RGBA'
    bpy.context.scene.render.film_transparent = True


    scene = bpy.context.scene
    cam = scene.objects['Camera']
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty


    light = scene.objects["Light"]
    light.location.x = 0.0
    light.location.y = -0.5
    light.location.z = 4
    light.data.energy = 400.0
    light.data.specular_factor = 0.5



    # light2 = bpy.data.lights.new(name="light_2.80", type='POINT')
    # print(dir(light2))
    # light2.distance = 100
    # # light2.location.x = -0.5
    # # light2.location.y = -0.5
    # # light2.location.z = 4
    # # light2.data.energy = 400.0
    # # light2.data.specular_factor = 0.5
    # light2_object = bpy.data.objects.new(name="light_2.80", object_data=light2)
    # bpy.context.collection.objects.link(light2_object)



    AZIMUTH_SAMPLES = 10
    ELEVATION_SAMPLES = 10
    DISTANCES_SAMPLES = 2
    ROTZ_SAMPLES = 1
    ELEV_LIMIT = 10

    azimuths = np.linspace(-np.pi, 0, AZIMUTH_SAMPLES)
    elevations = np.linspace(np.deg2rad(10), np.deg2rad(80), ELEVATION_SAMPLES) 

    N = 100
    if mode =="test":
        az = np.random.uniform(-np.pi, 0, size=(N))
        elev = np.random.uniform(np.deg2rad(10), np.deg2rad(85), size=(N))
        d = np.random.uniform(0.9, 1.2, size=(N))
    else:
        distances = np.linspace(0.9, 1.2, DISTANCES_SAMPLES, endpoint=True)
        rotations = np.linspace(0.0, np.deg2rad(360.0), ROTZ_SAMPLES, endpoint=False)
        az, elev, d = np.meshgrid(azimuths, elevations, distances)

    az = az.flatten()
    elev = elev.flatten()
    d = d.flatten()

    x = d * np.cos(az) * np.cos(elev)
    y = d * np.sin(az) * np.cos(elev)
    z = d * np.sin(elev)

    cam_positions = np.vstack((x, y, z)).T
    #cam_positions = cam_positions[np.where(cam_positions[:, 2] >= 0)[0], :]  # keep only cameras over


    NB_IMAGES = cam_positions.shape[0]

    # Load ellipspoid + create its mesh at a certain scale
    from ellipsoid_tools import decompose_ellipsoid
    Q = np.loadtxt(ELLIPSOID_MATRIX)
    # rescale for 2 stddev
    # axes, R, center = decompose_ellipsoid(Q)
    # scale = 1#2.0
    # axes *= scale
    # euler = Rot.from_matrix(R).as_euler("xyz")
    # phys_params = np.hstack((axes, center, euler))
    # Q = ellipsoid_phys_to_alg_mat(phys_params)

    # ellipsoid_pts, _ = read_obj(UNIT_SPHERE)
    # axes, R, center = decompose_ellipsoid(Q)
    # ellipsoid_pts[:, 0] *= axes[0]
    # ellipsoid_pts[:, 1] *= axes[1]
    # ellipsoid_pts[:, 2] *= axes[2]
    # ellipsoid_pts = R @ ellipsoid_pts.T
    # ellipsoid_pts = ellipsoid_pts.T
    # ellipsoid_pts += center
    # ellipsoid_pts_h = np.hstack((ellipsoid_pts, np.ones((ellipsoid_pts.shape[0], 1))))

    # configure Cycles engine and use GPU
    bpy.context.scene.render.engine = RENDER_ENGINE
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.feature_set = "SUPPORTED"
    bpy.context.preferences.addons['cycles'].preferences.devices[0].use = True
    bpy.context.scene.eevee.motion_blur_samples = 1
    bpy.context.scene.eevee.volumetric_samples = 4
    bpy.context.scene.eevee.taa_render_samples = 8
    bpy.context.scene.eevee.taa_samples = 8


    K = get_calibration_matrix_K_from_blender(cam)
    print("K = \n", K)


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

    LABELS = []
    LABELS_CAM_POSES = []
    LABELS_BOX_INFO = []

    for i in range(NB_IMAGES):
        cam.location.x = cam_positions[i, 0]
        cam.location.y = cam_positions[i, 1]
        cam.location.z = cam_positions[i, 2]

        b_empty.location.x = (np.random.rand() - 0.5) / 5
        b_empty.location.y = (np.random.rand() - 0.5) / 5

        bpy.context.scene.render.filepath = os.path.join(output_images_folder, "img_%08d.png" % i)
        bpy.ops.render.render(write_still=True)

        # if MASK_BACKGROUND:
        #     img = cv2.imread(bpy.context.scene.render.filepath, cv2.IMREAD_UNCHANGED)
        #     threshold = img[:, :, 3]
        #     img = img[:, :, :3]
        #     mask = threshold != 0
        #     img[mask == 0] = [0, 255, 0]
        #     cv2.imwrite(bpy.context.scene.render.filepath, img)

        # transform points to cam and project points on image
        cam_pose = get_object_pose(cam)
        correct = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
        cam_pose =  cam_pose @ correct
        T_to_cam = np.linalg.inv(cam_pose)
        T_to_cam = T_to_cam[:3, :]
        P = K @ T_to_cam

        proj = P @ all_pts_world_h
        proj /= proj[2, :]
        proj[0, :] = np.clip(proj[0, :], 0, WIDTH-1)
        proj[1, :] = np.clip(proj[1, :], 0, HEIGHT-1)
        proj = np.round(proj).astype(int)

        # proj2 = P @ ellipsoid_pts_h.T
        # proj2 /= proj2[2, :]
        # proj2 = np.round(proj2).astype(int)
        # proj2[0, :] = np.clip(proj2[0, :], 0, WIDTH-1)
        # proj2[1, :] = np.clip(proj2[1, :], 0, HEIGHT-1)

        proj_Q = P @ Q @ P.T
        axes, angle, c = decompose_ellipse(proj_Q)
        w, h = axes
        if w >= h:
            p = np.array([w, 0])
            pp = np.array([0, h])
        else:
            p = np.array([0, h])
            pp = np.array([-w, 0])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        p = R @ p
        pp = R @ pp

        if p[0] >= 0 and p[1] >= 0:
            p1 = p
            p2 = pp
            p3 = -p
            p4 = -pp
        elif p[0] < 0 and p[1] >= 0:
            p1 = -pp
            p2 = p
            p3 = pp
            p4 = -p
        elif p[0] < 0 and p[1] < 0:
            p1 = -p
            p2 = -pp
            p3 = p
            p4 = pp
        else:
            p1 = pp
            p2 = -p
            p3 = -pp
            p4 = p


        p0 = c
        p1 += c
        p2 += c
        p3 += c
        p4 += c
        
        ################
        # debug_img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        # debug_img[proj[1, :], proj[0, :], :] = 255
        # debug_img[proj2[1, :], proj2[0, :], 1] = 255
        # debug_img_cpy = debug_img.copy()

        # debug_img = draw_ellipse_from_mat(debug_img, proj_Q, color=(0, 0, 255), thickness=1)
        # debug_img = cv2.circle(debug_img, tuple(map(int, c.tolist())),  4, (255, 0, 0), -1)
        # debug_img = cv2.circle(debug_img, tuple(map(int, p1.tolist())), 4, (0, 0, 255), -1)
        # debug_img = cv2.circle(debug_img, tuple(map(int, p2.tolist())), 4, (0, 255, 0), -1)
        # debug_img = cv2.circle(debug_img, tuple(map(int, p3.tolist())), 4, (255, 0, 255), -1)
        # debug_img = cv2.circle(debug_img, tuple(map(int, p4.tolist())), 4, (255, 255, 0), -1)

        # cv2.imwrite(os.path.join(output_debug_folder, "img_%08d.png" % i), debug_img)

        ellipsoid = Ellipsoid.from_dual(Q)
        ell = ellipsoid.project(P)


        if DO_CROP:
            # crop object and save
            img = cv2.imread(bpy.context.scene.render.filepath, cv2.IMREAD_UNCHANGED)
            

            img = img[:, :, :3]
            # pts = np.where(np.logical_not(np.logical_and(img[:, :, 0] == 0, np.logical_and(img[:, :, 1] == 255, img[:, :, 2] == 0))))
            # if len(pts) == 0 or len(pts[0]) == 0:   # the object was outside the cam fov
            #     print("outside cam fov")
            #     sys.exit(0)
            # pts = np.vstack(pts).T
            pts = proj[:2, :].T

            mask = np.zeros((HEIGHT, WIDTH, 1), dtype=np.uint8)
            mask[pts[:, 1], pts[:, 0]] = 255
            
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            w = max_x - min_x + 1        
            h = max_y - min_y + 1
            dim = max(w, h)+2
            center_x, center_y = int(np.round((min_x+max_x) / 2)), int(np.round((min_y+max_y) / 2))
            xmin, xmax = center_x - int(np.ceil(dim/2)), center_x + int(np.ceil(dim/2))
            ymin, ymax = center_y - int(np.ceil(dim/2)), center_y + int(np.ceil(dim/2))

            if xmin < 0: xmin = 0
            if ymin < 0: ymin = 0
            if xmax >= WIDTH: xmax = WIDTH-1
            if ymax >= HEIGHT: ymax = HEIGHT-1
            if xmin < 0 or ymin < 0 or xmax > WIDTH-1 or ymax > HEIGHT-1:
                print("box is outside image")
                sys.exit(0)
            
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(WIDTH-1, xmax)
            ymax = min(HEIGHT-1, ymax)
            crop = img[ymin:ymax+1, xmin:xmax+1, :]
            crop_mask = mask[ymin:ymax+1, xmin:xmax+1]
            
            # mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.float)
            # obj_pts = np.where(np.logical_not(np.logical_and(crop[:, :, 0] == 0, np.logical_and(crop[:, :, 1] == 255, crop[:, :, 2] == 0))))
            # mask[obj_pts[0], obj_pts[1]] = 1
            # mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
            # obj_pts = np.where(mask_256 < 0.95)
            crop_256 = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)

            mask_crop_256 = cv2.resize(crop_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            # crop_256[obj_pts[0], obj_pts[1], :] = [0, 255, 0]       
            
            mask_crop_256 = cv2.threshold(mask_crop_256.copy(), 1, 255, cv2.THRESH_BINARY)[1]
            # mask_crop_256 = cv2.dilate(mask_crop_256, None, iterations=2)
            # mask_crop_256 = cv2.erode(mask_crop_256, None, iterations=2)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9, 9))

            mask_crop_256 = cv2.morphologyEx(mask_crop_256, cv2.MORPH_CLOSE, kernel)
            # print(mask_crop_256.min(), mask_crop_256.max())
            # # m = mask_crop_256
            # # m[m == 0] = 1
            # # m[m == 255] = 0
            # # m[m == 1] = 255
            # # mask_crop_256 = m
            # print(mask_crop_256)
            

            ctr = cv2.findContours(mask_crop_256, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # print("CTR = ", ctr)

            mask_crop_256 = cv2.drawContours(mask_crop_256, ctr[0], -1, 255, -1)



            

            ps = np.vstack((p0, p1, p2, p3, p4))
            ps -= np.array([xmin, ymin])
            factor = 256.0 / crop.shape[0]
            ps *= factor
            p0 = ps[0, :]
            p1 = ps[1, :]
            p2 = ps[2, :]
            p3 = ps[3, :]
            p4 = ps[4, :]


            # ell = ell.translate([-xmin, -ymin]).full_scale(factor)
            # draw_ellipse(crop_256, ell, (0, 255, 0), 1)
            # cv2.circle(crop_256, tuple(map(int, p0.tolist())),  4, (255, 0, 0), -1)
            # cv2.circle(crop_256, tuple(map(int, p1.tolist())), 4, (0, 0, 255), -1)
            # cv2.circle(crop_256, tuple(map(int, p2.tolist())), 4, (0, 255, 0), -1)
            # cv2.circle(crop_256, tuple(map(int, p3.tolist())), 4, (255, 0, 255), -1)
            # cv2.circle(crop_256, tuple(map(int, p4.tolist())), 4, (255, 255, 0), -1)


            cv2.imwrite(bpy.context.scene.render.filepath, crop_256)

            # ell_mask = ell.translate([-xmin, -ymin]).full_scale(factor)
            # mask = np.zeros_like(crop_256)
            # draw_ellipse(mask, ell_mask, (255, 255, 255), -1)
            cv2.imwrite(os.path.join(output_masks_folder, "mask_%08d.png" % i), np.dstack([mask_crop_256]*3))
            
            LABELS_BOX_INFO.append((xmin, ymin, factor))  
        else:
            LABELS_BOX_INFO.append((0, 0, 1))  
        LABELS.append(p0.tolist() + p1.tolist() + p2.tolist() + p3.tolist() + p4.tolist())
        LABELS_CAM_POSES.append(cam_pose[:3, :])

        # bg = cv2.imread(os.path.join(fp, 'rendering_%03d.png' % i)).astype(float)
        # fg = np.dstack([img.astype(float)]*3)
        # res = 0.5 * bg + 0.5 * fg
        # res = res.astype(np.uint8)
        # cv2.imwrite(os.path.join(fp, 'mixed/rendering_%03d_mixed.png' % i), res)


    b = time.time()
    print("done in ", b-a)

    LABELS = np.vstack(LABELS)
    with h5py.File(os.path.join(output_labels_folder, output_labels_name), 'w') as hf:
        hf.create_dataset('points', data=LABELS, compression="gzip", compression_opts=3)
        
        
    with h5py.File(os.path.join(output_labels_folder, output_labels_cam_name), 'w') as hf:
        hf.create_dataset('poses', data=LABELS_CAM_POSES, compression="gzip", compression_opts=3)
        hf.create_dataset('ellipsoid', data=Q, compression="gzip", compression_opts=3)
        hf.create_dataset('intrinsics', data=K, compression="gzip", compression_opts=3)
        
        
    with h5py.File(os.path.join(output_labels_folder, output_labels_box_name), 'w') as hf:
        hf.create_dataset('boxes', data=LABELS_BOX_INFO, compression="gzip", compression_opts=3)
        