#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 22:50:37 2022

@author: john
"""
from proj1 import *

n                   = 225
mesh                = mesh_grid(n)
constants           = constants(50 , 273.16)
G                   = label_grid(mesh)
plotter(G, mesh, title = "Geometry", colortitle =" ", cb = False)
