#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Wed Mar 30 11:05:10 2022

@author: john
"""

from proj1 import *

N       = 225
mesh_h  = mesh_grid(N)
mesh_2h = mesh_grid(int((N+1)/2))
mesh_4h = mesh_grid(int((N+3)/4))
constants = constants(50 , 273.16)


G_h     = label_grid(mesh_h)
G_2h    = label_grid(mesh_2h)
G_4h    = label_grid(mesh_4h)





# for plotting
idx_h  = np.where((G_h == 1) | (G_h == 2))
idx_2h = np.where((G_2h == 1) | (G_2h == 2))
idx_4h = np.where((G_4h == 1) | (G_4h == 2))


L_h, r_h, u_bound_h, index_h     = generate_laplacian(G_h, mesh_h, constants)
L_2h, r_2h, u_bound_2h, index_2h = generate_laplacian(G_2h, mesh_2h, constants)
L_4h, r_4h, u_bound_4h, index_4h = generate_laplacian(G_4h, mesh_4h, constants)



R_h_to_2h   = restrict(mesh_h, mesh_2h, G_h, G_2h, index_h, index_2h)            
R_2h_to_4h  = restrict(mesh_2h, mesh_4h, G_2h, G_4h, index_2h, index_4h)

I_2h_to_h, I_2h_to_h_win   = interpolate(mesh_h, mesh_2h, G_h, G_2h, index_h, index_2h)

I_4h_to_2h, I_4h_to_2h_win = interpolate(mesh_2h, mesh_4h, G_2h, G_4h, index_2h, index_4h)



u_h     = solve_steady(L_h, r_h, u_bound_h)
Gu_h    = generate_u_grid(G_h, u_h, index_h, constants, mesh_h) 
plotter(Gu_h, mesh_h)



u_h2    = solve_steady(L_2h, r_2h, u_bound_2h)
Gu_h2   = generate_u_grid(G_2h, u_h2, index_2h, constants, mesh_2h) 
plotter(Gu_h2, mesh_2h)

u_h4    = solve_steady(L_4h, r_4h, u_bound_4h)
Gu_h4   = generate_u_grid(G_4h, u_h4, index_4h, constants, mesh_4h) 
plotter(Gu_h4, mesh_4h)

u_hi    = I_2h_to_h@u_h2 + I_2h_to_h_win@(constants.u_window*np.ones_like(u_h2))
Gu_hi   = generate_u_grid(G_h, u_hi, index_h, constants, mesh_h) 
plotter(Gu_hi, mesh_h)

u_2hi   = I_4h_to_2h@u_h4 + I_4h_to_2h_win@(constants.u_window*np.ones_like(u_h4))
Gu_2hi  = generate_u_grid(G_2h, u_2hi, index_2h, constants, mesh_2h) 
plotter(Gu_2hi, mesh_2h)

# u_2h    = R_h_to_2h@u_h
# Gu_2h   = generate_u_grid(G_2h, u_2h, index_2h, constants, mesh_2h) 
# plotter(Gu_2h, mesh_2h)

# u_4h    = R_2h_to_4h@u_2h
# Gu_4h   = generate_u_grid(G_4h, u_4h, index_4h, constants, mesh_4h) 
# plotter(Gu_4h, mesh_4h)


X_h, Y_h    = np.meshgrid(mesh_h.x, mesh_h.y)

X_2h, Y_2h  = np.meshgrid(mesh_2h.x, mesh_2h.y)

X_4h, Y_4h  = np.meshgrid(mesh_4h.x, mesh_4h.y)

plt.figure(figsize =(50,50))
plt.scatter(X_h[idx_h], Y_h[idx_h], color ='red')
plt.scatter(X_2h[idx_2h], Y_2h[idx_2h], color ='blue')
plt.scatter(X_4h[idx_4h], Y_4h[idx_4h], color ='black', marker = "+", s = 100)