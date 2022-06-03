#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 18:37:20 2022

@author: john
"""
from proj1 import *
import time

n                   = 225
mesh                = mesh_grid(n)
constants           = constants(50 , 273.16)
G                   = label_grid(mesh)
#plotter(G, mesh)
T                   = 30*max(mesh.sizex, mesh.sizey)
L,r, u_bound, idx   = generate_laplacian(G, mesh, constants)
A, rhs              = generate_euler_explicit(L, r, u_bound, 0.0001)
t0 = time.time()
u_t                 = solve_euler_explicit(A, 0.0001, T, rhs, constants)
print(time.time() - t0)
Gu_t                = generate_u_grid(G, u_t, idx, constants, mesh)
plotter(Gu_t, mesh, title = "Unsteady solution (forward Euler)", \
        colortitle = r"$u(x, y, t = {})$".format(T))
