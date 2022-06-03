#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 08:57:46 2022

@author: john
"""
from proj1 import *

n                   = 225
mesh                = mesh_grid(n)
constants           = constants(50 , 273.16)
G                   = label_grid(mesh)
L, r, u_bound, idx  = generate_laplacian(G, mesh, constants)
u                   = solve_steady(L, r, u_bound)
Gu                  = generate_u_grid(G, u, idx, constants, mesh) 
plotter(Gu, mesh, title = "Steady state solution", colortitle = r"$u(x, y, t \rightarrow \infty)$ (Kelvin)")

