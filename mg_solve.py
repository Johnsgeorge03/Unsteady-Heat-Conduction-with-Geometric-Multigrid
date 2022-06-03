#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:36:47 2022

@author: john
"""
from proj1 import *

N         = 401
mesh_h    = mesh_grid(N)
mesh_2h   = mesh_grid(int((N+1)/2))
mesh_4h   = mesh_grid(int((N+3)/4))
constants = constants(50 , 273.16)


G_h     = label_grid(mesh_h)
G_2h    = label_grid(mesh_2h)
G_4h    = label_grid(mesh_4h)



L_h, r_h, u_bound_h, index_h     = generate_laplacian(G_h, mesh_h, constants)
L_2h, r_2h, u_bound_2h, index_2h = generate_laplacian(G_2h, mesh_2h, constants)
L_4h, r_4h, u_bound_4h, index_4h = generate_laplacian(G_4h, mesh_4h, constants)



R_h_to_2h   = restrict(mesh_h, mesh_2h, G_h, G_2h, index_h, index_2h)            
R_2h_to_4h  = restrict(mesh_2h, mesh_4h, G_2h, G_4h, index_2h, index_4h)


I_2h_to_h, I_2h_to_h_win   = interpolate(mesh_h, mesh_2h, G_h, G_2h, index_h, index_2h)
I_4h_to_2h, I_4h_to_2h_win = interpolate(mesh_2h, mesh_4h, G_2h, G_4h, index_2h, index_4h)


def vcycle(u, rhs, nu):
    rhs0 = rhs.copy()
    u0   = u.copy()
    
    u0   = damped_jacobi(L_h, u0, rhs0, nu, 0.05)
    rhs1 = R_h_to_2h@(rhs0 - L_h@u0)
    
    u1   = damped_jacobi(L_2h, np.zeros_like(rhs1), rhs1, nu, 0.05)
    rhs2 = R_2h_to_4h@(rhs1 - L_2h@u1)
    
    u2   = solve_steady(L_4h, 0*rhs2, rhs2)
    
    u1   += I_4h_to_2h@u2 
    u1   = damped_jacobi(L_2h, u1, rhs1, nu, 0.05)
    
    u0   += I_2h_to_h@u1 
    u0   = damped_jacobi(L_h, u0, rhs0, nu, 0.05)
    
    return u0

rhs = u_bound_h - r_h

res = []
u   = 295*np.ones(L_h.shape[0])

for i in range(80):
    res.append(np.linalg.norm(rhs - L_h @ u, ord = np.inf))
    if(res[0]/res[-1] >= 1e12):
        break
    u = vcycle(u, rhs, 3)
    

plt.figure(figsize=(8, 8), dpi = 400)
plt.semilogy(res, lw=3, label="residual")
plt.xlabel(r"Iterations of V-cycle", fontsize=15)
plt.ylabel(r"$||f - Au||_{\infty}$", fontsize=15)
plt.title(r"Convergence plot for residual norm (2-level mutligrid)", fontsize=20)
plt.legend()
plt.grid()

Gu = generate_u_grid(G_h, u, index_h, constants, mesh_h)
plotter(Gu, mesh_h, title= "Muligrid steady state solution")

u_h = solve_steady(L_h, r_h, u_bound_h)
Gu_h = generate_u_grid(G_h, u_h, index_h, constants, mesh_h)
plotter(Gu_h, mesh_h, title = "Steady state direct solve")
