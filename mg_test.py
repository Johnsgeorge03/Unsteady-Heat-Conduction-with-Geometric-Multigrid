#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 17:48:33 2022

@author: john
"""
from proj1 import *



constants = constants(0, 373.16)
N = [301, 275, 251, 225, 201]
err = []
delta_x = []
for n in N:
    mesh_h    = test_mesh(n)
    mesh_2h   = test_mesh(int((n+1)/2))
    
    G_h     = label_test_grid(mesh_h)
    G_2h    = label_test_grid(mesh_2h)
    
    L_h, r_h, u_bound_h, index_h     = generate_laplacian(G_h, mesh_h, constants)
    L_2h, r_2h, u_bound_2h, index_2h = generate_laplacian(G_2h, mesh_2h, constants)
    
   
    u_true              = true_solution_steady(mesh_h, G_h,  constants, index_h)
    R_h_to_2h           = restrict(mesh_h, mesh_2h, G_h, G_2h, index_h, index_2h)   
    I_2h_to_h, I_2h_to_h_win   = interpolate(mesh_h, mesh_2h, G_h, G_2h, index_h, index_2h)
    
    u_2h = true_solution_steady(mesh_2h, G_2h, constants, index_2h)
    
    u                   = I_2h_to_h@(R_h_to_2h@u_true)\
                        + I_2h_to_h_win@(constants.u_window*np.ones(L_2h.shape[0]))
    
    delta_x.append(max(mesh_h.dx, mesh_h.dy))
    err.append(npla.norm(u - u_true, ord = np.inf))

plt.figure(figsize=(8, 6), dpi = 400)
plt.loglog(delta_x, err, "-o", color="black", label="error")
plt.loglog(delta_x, delta_x, "--", label=r"$O(\Delta x^1)$")
plt.loglog(delta_x, np.array(delta_x) ** 2, "--", label=r"$O(\Delta x^2)$")
plt.loglog(delta_x, np.array(delta_x) ** 3, "--", label=r"$O(\Delta x^3)$")
plt.loglog(delta_x, np.array(delta_x) ** 4, "--", label=r"$O(\Delta x^4)$")
plt.xlabel(r"max($\Delta x, \Delta y$)", fontsize=15)
plt.ylabel(r"$||\bar{u} - u||_{\infty}$", fontsize=15)
plt.title(r"Restriction, Interpolation error vs max($\Delta x, \Delta y)$", fontsize=20)
plt.legend()
plt.grid()

# Gu_h     = generate_u_grid(G_h, u, index_h, constants, mesh_h)
# plotter(Gu_h, mesh_h)

# Gu_2h    = generate_u_grid(G_2h, u_2h, index_2h, constants, mesh_2h)
# plotter(Gu_2h, mesh_2h)

# Gu_true    = generate_u_grid(G_h, u_true, index_h, constants, mesh_h)
# plotter(Gu_true, mesh_h)



# X_h, Y_h    = np.meshgrid(mesh_h.x, mesh_h.y)

# X_2h, Y_2h  = np.meshgrid(mesh_2h.x, mesh_2h.y)

# # for plotting
# idx_h  = np.where((G_h == 1) | (G_h == 2))
# idx_2h = np.where((G_2h == 1) | (G_2h == 2))

# plt.figure(figsize =(8,8))
# plt.scatter(X_h[idx_h], Y_h[idx_h], color ='red')
# plt.scatter(X_2h[idx_2h], Y_2h[idx_2h], color ='blue')
