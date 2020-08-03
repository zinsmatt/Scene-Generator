#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import os
import bpy
import glob

objects_folder = "/home/matt/dev/Scene-Generator/data/objects/"

file_list = sorted(glob.glob(os.path.join(objects_folder, "*.obj")))

for item in file_list:
    bpy.ops.import_scene.obj(filepath=item, axis_forward="Y", axis_up="Z")
    # obj = bpy.context.collection.objects[name]
    # obj.rotation_euler = [0,0,0]
    # obj.location.x = 0.1
    # obj.location.y = 0.3
    # obj.location.z = 0.0