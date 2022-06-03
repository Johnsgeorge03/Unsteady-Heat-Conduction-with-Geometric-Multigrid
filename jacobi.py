#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 22:36:35 2022

@author: john
"""
from proj1 import *

alpha                   = 0.05
n                       = 225
mesh                    = mesh_grid(n)
constants               = constants(50 , 273.16)
G                       = label_grid(mesh)
L, r, u_bound, idx      = generate_laplacian(G, mesh, constants)


u_guess                 = 500*np.random.rand(L.shape[0])
Gu_guess                = generate_u_grid(G, u_guess, idx, constants, mesh)
plotter(Gu_guess, mesh, title = "Initial guess for u(x,y)",
        colortitle = r"$u(x,y,t \rightarrow \infty)$ (Kelvin)")

u_djacobi               = damped_jacobi(L, u_guess, u_bound - r, 300, alpha)
Gu_dj                   = generate_u_grid(G, u_djacobi, idx, constants, mesh) 
plotter(Gu_dj, mesh, title = "Damped Jacobi (steady) - 300 iterations",
        colortitle = r"$u(x,y,t \rightarrow \infty)$ (Kelvin)")

u_jacobi                = jacobi(L, u_guess, u_bound - r, 300)
Gu_j                    = generate_u_grid(G, u_jacobi, idx, constants, mesh) 
plotter(Gu_j, mesh, title = "Undamped Jacobi (steady) - 300 iterations", 
        colortitle = r"$u(x,y,t \rightarrow \infty)$ (Kelvin)")

# Dinv                    = sp.sparse.diags(L.diagonal()**(-1), format = 'csr')
# M                       = sp.sparse.eye(L.shape[0]) - Dinv * L
# eig_val_j               = sp.sparse.linalg.eigs(M)[0]
# Md                      = sp.sparse.eye(L.shape[0]) - (1 - alpha)*Dinv*L
# eig_val_dj              = sp.sparse.linalg.eigs(Md)[0]