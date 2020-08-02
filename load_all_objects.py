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
    name = bpy.ops.import_scene.obj(filepath=item, axis_forward="Y", axis_up="Z")
    # obj = bpy.context.collection.objects[name]
    # obj.rotation_euler = [0,0,0]