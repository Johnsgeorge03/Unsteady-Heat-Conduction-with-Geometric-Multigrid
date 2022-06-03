#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 21:54:54 2022

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
A, B, rhs           = generate_implicit(L, r, u_bound, 0.01)
t0                  = time.time()
u_t                 = solve_implicit(A, B, 0.01, T, rhs, constants)
print(time.time() - t0)
Gu_t                = generate_u_grid(G, u_t, idx, constants, mesh)
plotter(Gu_t, mesh, title = "Unsteady solution (implicit midpoint)", \
        colortitle = r"$u(x, y, t = {})$".format(T))