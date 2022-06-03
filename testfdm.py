#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 17:48:33 2022

@author: john
"""
from proj1 import *

def rhs_unsteady(mesh, u_bound, time_step,  G, idx, t):
    dof = np.count_nonzero(G == 1)
    rr = np.zeros(dof)
    L = mesh.sizex
    H = mesh.sizey - mesh.dy
    for i in range(dof):
        gj, gi = np.where(idx == i)
        rr[i] = (np.pi**2/L**2 + np.pi**2/(4*H**2))*np.sin(np.pi*mesh.x[gi]/L)\
        *np.sin(np.pi*mesh.y[gj]/(2*H))*(373.16+ np.cos(t)) \
        - np.sin(np.pi*mesh.x[gi]/L)*np.sin(np.pi*mesh.y[gj]/(2*H))*np.sin(t)
    return time_step*(rr - u_bound)


def rhs_steady(mesh, G, idx):
    dof = np.count_nonzero(G == 1)
    rr = np.zeros(dof, dtype = np.float32)
    L = mesh.sizex
    H = mesh.sizey - mesh.dy
    for i in range(dof):
        gj, gi = np.where(idx == i)
        rr[i] = (np.pi**2/L**2 + np.pi**2/(4*H**2))*np.sin(np.pi*mesh.x[gi]/L)\
        *np.sin(np.pi*mesh.y[gj]/(2*H))
    return rr


def true_solution_unsteady(mesh, G, constants, idx, t):
    dof = np.count_nonzero(G == 1)
    u = np.zeros(dof)
    L = mesh.sizex
    H = mesh.sizey - mesh.dy
    for i in range(dof):
        gj, gi = np.where(idx == i)
        u[i]   = np.sin(np.pi*mesh.x[gi]/L)\
        *np.sin(np.pi*mesh.y[gj]/(2*H))*(373.16 + np.cos(t)) + 273.16
    return u




def generate_test_euler_explicit(L, time_step):
    print("Generating test explicit euler system")
    A   = sp.sparse.csc_matrix(time_step*L + sp.sparse.eye(L.shape[0]))
    return A



def solve_test_euler_explicit(A, time_step, final_time, rhs, mesh, constants, u_bound, idx, G):
    print("Solving test euler explicit")
    t   = 0
    u_t = true_solution_unsteady(mesh, G, constants, idx, t)
    while (t <= final_time):
        u_t = A@u_t + rhs(mesh, u_bound, time_step, G, idx, t)
        t   += time_step
        print(t)
    return u_t 



def generate_test_implicit(L, time_step):
    print("Generating test implict RK system")
    A   = sp.sparse.csc_matrix(sp.sparse.eye(L.shape[0]) - time_step*L/2)
    B   = sp.sparse.csc_matrix(sp.sparse.eye(L.shape[0]) + time_step*L/2)
    return A, B



def solve_test_implicit(A, B, time_step, final_time, rhs, constants, mesh, u_bound, G, idx):
    print("Solving test implicit")
    t   = 0
    u_t = true_solution_unsteady(mesh, G, constants, idx, t)
    b   = B@u_t + rhs(mesh, u_bound, time_step,  G, idx, t)
    solve = spla.factorized(A)
    while (t <= final_time):
        u_t = solve(b)
        b   = B@u_t + rhs(mesh, u_bound, time_step,  G, idx, t)
        t   += time_step
        print(t)
    return u_t 


constants_unsteady = constants(0, 273.16)
constants_steady   = constants(0, 373.16)
N = [300, 275, 250, 225, 200] # high to low
err = []
delta_x = []
T = 0.001

 ### FOR TESTING EULER EXPLICIT ####
# for n in N:
#     mesh               = test_mesh(n)
#     G                  = label_test_grid(mesh)
#     L, r, u_bound, idx = generate_laplacian(G, mesh, constants_unsteady)
#     A                  = generate_test_euler_explicit(L, 0.00001)
#     u_t                = solve_test_euler_explicit(A, 0.00001, T, rhs_unsteady, \
#                                     mesh, constants_unsteady, u_bound, idx, G)
    
#     u_true = true_solution_unsteady(mesh, G, constants, idx, T)
#     delta_x.append(max(mesh.dx, mesh.dy))
#     err.append(npla.norm(u_t - u_true, ord = np.inf))


###### FOR TESTING IMPLICIT SCHEME ########
# for n in N:
#     mesh               = test_mesh(n)
#     G                  = label_test_grid(mesh)
#     L, r, u_bound, idx = generate_laplacian(G, mesh, constants_unsteady)
#     A, B               = generate_test_implicit(L, 0.0001)
#     u_t                = solve_test_implicit(A, B, 0.0001, T, rhs_unsteady, constants_unsteady\
#                                               , mesh, u_bound, G, idx)
    
#     u_true             = true_solution_unsteady(mesh, G, constants_unsteady, idx, T)
#     delta_x.append(max(mesh.dx, mesh.dy))
#     err.append(npla.norm(u_t - u_true, ord = np.inf))
    
#### FOR TESTING STEADY STATE SOLVE######

for n in N:
    mesh               = test_mesh(n)
    G                  = label_test_grid(mesh)
    L, r, u_bound, idx = generate_laplacian(G, mesh, constants_steady)
    r                  = rhs_steady(mesh, G, idx)
    u                  = solve_steady(L, r, u_bound)
    u_true = true_solution_steady(mesh, G, constants_steady, idx)
    delta_x.append(max(mesh.dx, mesh.dy))
    err.append(npla.norm(u - u_true, ord = np.inf))

plt.figure(figsize=(8, 6), dpi = 400)
plt.loglog(delta_x, err, "-o", color="black", label="FD_error")
plt.loglog(delta_x, delta_x, "--", label=r"$O(\Delta x^1)$")
plt.loglog(delta_x, np.array(delta_x) ** 2, "--", label=r"$O(\Delta x^2)$")
plt.loglog(delta_x, np.array(delta_x) ** 3, "--", label=r"$O(\Delta x^3)$")
plt.loglog(delta_x, np.array(delta_x) ** 4, "--", label=r"$O(\Delta x^4)$")
plt.xlabel(r"max($\Delta x, \Delta y$)", fontsize=15)
plt.ylabel(r"$||\bar{u} - u||_{\infty}$", fontsize=15)
plt.title(r"Error vs max($\Delta x, \Delta y)$", fontsize=20)
plt.legend()
plt.grid()



