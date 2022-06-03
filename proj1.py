#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 10:21:31 2022

@author: john
"""

import numpy as np
import numpy.linalg as npla
import scipy as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# #0 - wall
# #1 - in_domain
# #2 - heater
# #3 - window

class mesh_grid:
    def __init__(self, n):
        self.n = n    
        self.sizex = 5
        self.sizey = 5
    
        #one-d grid 
        self.x = np.linspace(0, self.sizex, self.n)
        self.y = np.linspace(0, self.sizey, self.n)
        
        #grid cell size 
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        
        
        #heater dimensions
        self.hwidth = 0.15
        self.hlen   = 1
        
        #inner room dimensions
        self.inroomlen = 1.8
        self.inroomwidth = 1.8
        
        #inner door length
        self.indoorlen = 0.7
        
        #wall thickness 
        self.wthick = 0.1
        
        #window length (1,2)
        self.w1len = 1.2
        self.w2len = 0.5
        
        #window width (1,2)
        #self.wwidth = self.wthick + self.dy

class test_mesh:
    def __init__(self, n):
        self.n = n
        self.sizex = 5
        self.sizey = 5
        
        #one-d grid 
        self.x = np.linspace(0, self.sizex, self.n)
        self.y = np.linspace(0, self.sizey, self.n)
        
        #grid cell size 
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        
        #wall thickness 
        self.wthick = 0


def label_testmg_grid(mesh):
    print("Labeling test grid")
    G = np.zeros((mesh.n, mesh.n), dtype = np.int32)
    for j in range(len(mesh.y)):
        for i in range(len(mesh.x)):
            ### Window
            if(mesh.y[j] == mesh.sizey):  
                G[j][i] = 3
            
            ### Wall
            elif (mesh.x[i] == 0 or mesh.x[i] == mesh.sizex or mesh.y[j] == 0):
                G[j][i] = 0
            
            ### IN_DOMAIN
            else:
                G[j][i] = 1
                
    return G
     
        
def label_test_grid(mesh):
    print("Labeling test grid")
    G = np.zeros((mesh.n, mesh.n), dtype = np.int32)
    for j in range(len(mesh.y)):
        for i in range(len(mesh.x)):
            ### WALL
            if(mesh.y[j] == mesh.sizey):  
                G[j][i] = 0
            
            ### WINDOW
            elif (mesh.x[i] == 0 or mesh.x[i] == mesh.sizex or mesh.y[j] == 0):
                G[j][i] = 3 
            
            ### IN_DOMAIN
            else:
                G[j][i] = 1
                
    return G



class constants():
    def __init__(self, source, u_window):
        self.source   = source #source term
        self.u_window = u_window #window temp
    
        
     



 
def plotter(Gu, mesh, title = "", colortitle = "", cb = True):
    plt.figure(figsize = (8,8), dpi = 500)
    plt.imshow(Gu, origin ="lower", extent=[mesh.x[0], mesh.x[-1], mesh.y[0], mesh.y[-1]])
    if cb:
        plt.colorbar(label = colortitle)
    plt.title(title, fontsize = 18)
    plt.xlabel("x (m)", fontsize = 15)
    plt.ylabel("y (m)", fontsize = 15)
    plt.gca().set_aspect("equal")    




def label_grid(mesh):
    print("Labeling grid")
    G = np.zeros((mesh.n, mesh.n), dtype = np.int32)
    for j in range(len(mesh.y)):
        for i in range(len(mesh.x)):
            ### WINDOW
            if(((mesh.y[j] <= mesh.sizey) and (mesh.y[j] >= (mesh.sizey - mesh.wthick)) #window
                 and (mesh.x[i] >= mesh.wthick)
                 and (mesh.x[i] <= mesh.wthick + mesh.w1len))
                 or ((mesh.y[j] <= mesh.sizey) and (mesh.y[j] >= (mesh.sizey - mesh.wthick))
                 and (mesh.x[i] >= mesh.sizex - mesh.inroomwidth + 0.5*(mesh.inroomwidth - mesh.w2len))
                 and (mesh.x[i] <= mesh.sizex - mesh.inroomwidth + 0.5*(mesh.inroomwidth + mesh.w2len)))):
                 
                G[j][i] = 3
            
            ### WALL
            elif ((mesh.x[i] <= mesh.wthick) or (mesh.x[i] >= (mesh.sizex - mesh.wthick)) or (mesh.y[j] <= mesh.wthick )
                or (mesh.y[j] >= (mesh.sizey - mesh.wthick)) 
                or ((mesh.x[i] >= 0.5*(mesh.sizex - mesh.wthick)) and (mesh.x[i] <= 0.5*(mesh.sizex + mesh.wthick))
                and (mesh.y[j] >= mesh.wthick) and (mesh.y[j] <= 0.5*(mesh.sizey - mesh.hlen)))
                
             
                or ((mesh.x[i] >= (mesh.sizex - mesh.inroomwidth)) and ( mesh.x[i] <= (mesh.sizex - mesh.wthick))
                and (mesh.y[j] >= 0.5*(mesh.sizey + mesh.hlen)) and (mesh.y[j] <= 0.5*(mesh.sizey + mesh.hlen) + mesh.wthick))
                
                or ((mesh.x[i] >= (mesh.sizex - mesh.inroomwidth)) and ( mesh.x[i] <= (mesh.sizex - mesh.inroomwidth + mesh.wthick))
                and (mesh.y[j] >= 0.5*(mesh.sizey + mesh.hlen) + mesh.wthick)  and (mesh.y[j] <= (mesh.sizey - mesh.indoorlen - mesh.wthick)))):
                
                G[j][i] = 0
            
            ### HEATER
            elif((( mesh.x[i] >= mesh.wthick + 3*mesh.dx ) and ( mesh.x[i] <= (mesh.wthick + 3*mesh.dx + mesh.hwidth)) 

                     and ( mesh.y[j] >= mesh.wthick + 3*mesh.dy) 
                     and ( mesh.y[j] <= (mesh.hlen + mesh.wthick + 3*mesh.dy)))
                     or(( mesh.x[i] >= (mesh.sizex - mesh.wthick - mesh.hwidth - 3*mesh.dx))
                        and ( mesh.x[i] <= (mesh.sizex - mesh.wthick - 3*mesh.dx))
                        and ( mesh.y[j] >= (0.5*(mesh.sizey - mesh.hlen) - 3*mesh.dy)) 
                        and ( mesh.y[j] <= 0.5*(mesh.sizey + mesh.hlen) - 3*mesh.dy))
                     or(( mesh.x[i] >= (mesh.sizex - mesh.inroomwidth - mesh.hlen))
                        and ( mesh.x[i] <= (mesh.sizex - mesh.inroomwidth - 0.2*mesh.indoorlen))
                        and ( mesh.y[j] >= mesh.sizey - mesh.wthick - mesh.hwidth - 3*mesh.dy) 
                        and ( mesh.y[j] <= mesh.sizey - mesh.wthick - 3*mesh.dy))):
                G[j][i] = 2
            
            ### IN_DOMAIN
            else:
                G[j][i] = 1
                
    return G





def generate_laplacian(G, mesh, constants):
    print("Generating laplacian system")
    dof = np.count_nonzero(G == 1) + np.count_nonzero(G == 2)
    L   = sp.sparse.lil_matrix((dof, dof))
    r   = np.zeros(dof)
    u_bound = np.zeros(dof)
    I   = -1
    idx = -1*np.ones_like(G,dtype = np.int32)
    for j in range(len(mesh.y)):
        for i in range(len(mesh.x)):
            if ((G[j][i] == 1 or G[j][i] == 2) and I <= dof): #if the point is in domain do the following
                I += 1
                idx[j][i] = I
                idx_ijp1 = np.count_nonzero(G[j][i+1:] == 1) + np.count_nonzero(G[j][i+1:] == 2) \
                    + np.count_nonzero(G[j+1][:i+1] == 1)  + np.count_nonzero(G[j+1][:i+1] == 2)
                    
                idx_ijm1 = np.count_nonzero(G[j-1][i:] == 1) + np.count_nonzero(G[j-1][i:] == 2) \
                    + np.count_nonzero(G[j][:i] == 1)  + np.count_nonzero(G[j][:i] == 2)
                
               
                if(G[j][i] == 2):
                    r[I] = constants.source
                    
                    
                # BOTTOM
                
                if (G[j-1][i] == 0): #wall
                    L[I,I + idx_ijp1] = 2/mesh.dy**2
                    
                    
                elif(G[j-1][i] == 3): #window
                    u_bound[I] += -constants.u_window/mesh.dy**2 
                    
                
                elif((G[j-1][i] == 1 or G[j-1][i] == 2) and G[j+1][i] != 0): #in_domain
                    L[I, I - idx_ijm1] = 1/mesh.dy**2
                    
                    
                # TOP
                
                if (G[j+1][i] == 0): #wall
                    L[I, I - idx_ijm1] = 2/mesh.dy**2
                 
                
                elif(G[j+1][i] == 3): #window
                    u_bound[I] += -constants.u_window/mesh.dy**2 
                    
                    
                elif((G[j+1][i] == 1 or G[j+1][i] == 2) and G[j-1][i]!=0): #in_domain
                    L[I, I + idx_ijp1] = 1/mesh.dy**2
                    
                
                # RIGHT
                    
                if (G[j][i+1] == 0): #wall
                    L[I, I-1] = 2/mesh.dx**2

                        
                elif(G[j][i+1] == 3): #window
                    u_bound[I] += -constants.u_window/mesh.dx**2
                
                elif((G[j][i+1] == 1 or G[j][i+1] == 2) and G[j][i-1]!=0): #in_domain
                    L[I, I+1] = 1/mesh.dx**2
                    
                    
                # LEFT
                
                if (G[j][i-1] == 0): #wall
                    L[I, I+1] = 2/mesh.dx**2
                
                     
                elif(G[j][i-1] == 3): #window
                    u_bound[I] += -constants.u_window/mesh.dx**2
                
                elif((G[j][i-1] == 1 or G[j][i-1] == 2) and G[j][i+1] != 0): #in_domain
                    L[I, I-1] = 1/mesh.dx**2
                    
                L[I, I] = (-2/mesh.dx**2 - 2/mesh.dy**2)
                
    return sp.sparse.csc_matrix(L), r, u_bound, idx
            
            
    
            
            
def solve_steady(L, r, u_bound):
    print("Solving for steady-state system")
    solve = spla.factorized(L)
    u     = solve(u_bound - r)
    return u





def generate_u_grid(G, u, index, constants, mesh):
    print("Generating u grid")
    Gu = np.copy(G)
    for j in range(len(mesh.y)):
        for i in range(len(mesh.x)):
            if(G[j][i] == 1 or G[j][i] == 2):
                Gu[j][i] = u[int(index[j][i])]
            elif(G[j][i] == 3 or G[j][i] == 0):
                Gu[j][i] = constants.u_window
    return Gu





def generate_euler_explicit(L, r, u_bound, time_step):
    print("Generating explicit euler system")
    A   = sp.sparse.csc_matrix(time_step*L + sp.sparse.eye(L.shape[0]))
    rhs = time_step*(r - u_bound)
    return A, rhs

 
   

def solve_euler_explicit(A, time_step, final_time, rhs, constants):
    print("Solving euler explicit")
    u_t = constants.u_window*np.ones(A.shape[0])
    t   = 0
    while (t <= final_time):
        u_t = A@u_t + rhs
        t   += time_step
    return u_t 



def generate_implicit(L, r, u_bound, time_step):
    print("Generating implict RK system")
    A   = sp.sparse.csc_matrix(sp.sparse.eye(L.shape[0]) - time_step*L/2)
    B   = sp.sparse.csc_matrix(sp.sparse.eye(L.shape[0]) + time_step*L/2)
    rhs = time_step*(r - u_bound)
    return A, B, rhs



def solve_implicit(A, B, time_step, final_time, rhs, constants):
    print("Solving implicit")
    u_t = constants.u_window*np.ones(A.shape[0])
    t   = 0
    b   = B@u_t + rhs
    solve = spla.factorized(A)
    while (t <= final_time):
        u_t = solve(b)
        b   = B@u_t + rhs
        t   += time_step
        #print(t)
    return u_t  



def jacobi(A, x, b, nu):
    '''
    Jacobi 
    nu    - no.of runs
    x     - initial guess
    b     - right hand side
    '''
    print("Jacobi iteration")
    Dinv = sp.sparse.diags(A.diagonal()**(-1), format='csr')
    for i in range(nu):
      x  = x +  Dinv * (b - A * x)
    return x



def damped_jacobi(A, x, b, nu, alpha):
    '''
    Jacobi for smoothing 
    nu    - no.of runs
    x     - initial guess
    b     - right hand side
    '''
    print("Damped Jacobi iteration")
    Dinv = sp.sparse.diags(A.diagonal()**(-1), format='csr')
    for i in range(nu):
      x  = alpha*x + (1 - alpha)*(x +  Dinv * (b - A * x))
    return x



def restrict(mesh_h, mesh_2h, G_h, G_2h, index_h, index_2h):
    print("Constructing restriction matrix")
    
    dof_h      = np.count_nonzero(G_h == 1) + np.count_nonzero(G_h == 2)
    dof_2h     = np.count_nonzero(G_2h == 1) + np.count_nonzero(G_2h == 2)
    
    R_h_to_2h  = sp.sparse.lil_matrix((dof_2h, dof_h), dtype = np.int8)

    for i in range(dof_2h):
        gj, gi = np.where(index_2h == i)
        xi = np.where(np.abs(mesh_h.x - mesh_2h.x[gi]) <= 1e-7)
        yj = np.where(np.abs(mesh_h.y - mesh_2h.y[gj]) <= 1e-7)
        if (xi[0].any() and yj[0].any()):
            J = index_h[yj,xi]
            R_h_to_2h[i, J] = 1
    
    return sp.sparse.csc_matrix(R_h_to_2h)




def interpolate(mesh_h, mesh_2h, G_h, G_2h, index_h, index_2h):
    print("Constructing interpolation matrix")
    
    dof_h           = np.count_nonzero(G_h == 1) + np.count_nonzero(G_h == 2)
    dof_2h          = np.count_nonzero(G_2h == 1) + np.count_nonzero(G_2h == 2)
    
    I_2h_to_h       = sp.sparse.lil_matrix((dof_h, dof_2h))
    I_2h_to_h_win   = sp.sparse.lil_matrix((dof_h, dof_2h))
    
    
    
    for  i in range(dof_h):
            gj_h, gi_h = np.where(index_h == i)
            xi_2h = np.where(np.abs(mesh_2h.x - mesh_h.x[gi_h]) <= 1e-7)
            yj_2h = np.where(np.abs(mesh_2h.y - mesh_h.y[gj_h]) <= 1e-7)
            if (xi_2h[0].any() and yj_2h[0].any()): # if coarse and fine overlap
                if (i == 56437):
                    print(i)
                J = index_2h[yj_2h, xi_2h]
                I_2h_to_h[i, J] = 1
            else:
                xip1_2h = np.where(np.abs(mesh_2h.x - mesh_h.x[gi_h + 1]) <= 1e-7) # wall or indomain
                xim1_2h = np.where(np.abs(mesh_2h.x - mesh_h.x[gi_h - 1]) <= 1e-7)
                yjp1_2h = np.where(np.abs(mesh_2h.y - mesh_h.y[gj_h + 1]) <= 1e-7)
                yjm1_2h = np.where(np.abs(mesh_2h.y - mesh_h.y[gj_h - 1]) <= 1e-7)
                
                # index could be negative if wall 
                #there is coarse point to right and if not a wall get index
                if(yj_2h[0].any() and xip1_2h[0].any()  \
                   and index_2h[yj_2h, xip1_2h[0]] != -1):
                    Jxp1 = index_2h[yj_2h, xip1_2h]
                 
                # there is coarse point to left and that is not a wall, get index
                if(yj_2h[0].any() and xim1_2h[0].any() \
                    and index_2h[yj_2h, xim1_2h[0]] != -1):
                    Jxm1 = index_2h[yj_2h, xim1_2h]
                    
                # there is coarse point to top and if not a wall, get index
                if(xi_2h[0].any() and yjp1_2h[0].any() \
                   and index_2h[yjp1_2h[0], xi_2h] != -1):
                    Jyp1 = index_2h[yjp1_2h, xi_2h]
                    
                if(xi_2h[0].any() and yjm1_2h[0].any() \
                   and index_2h[yjm1_2h[0], xi_2h] != -1):
                    Jym1 = index_2h[yjm1_2h, xi_2h]
                    #print(i, Jym1)
                Jxm1ym1, Jxm1yp1, Jxp1ym1, Jxp1yp1 = -1, -1, -1, -1
                
                
                if(~xi_2h[0].any() and ~yj_2h[0].any()):
                    if index_2h[yjp1_2h, xip1_2h] != -1:
                        Jxp1yp1 = index_2h[yjp1_2h, xip1_2h]
                
                    if index_2h[yjp1_2h, xim1_2h] != -1:
                        Jxm1yp1 = index_2h[yjp1_2h, xim1_2h]
                    
                    if index_2h[yjm1_2h, xip1_2h] != -1:
                        Jxp1ym1 = index_2h[yjm1_2h, xip1_2h]
                    if index_2h[yjm1_2h, xim1_2h] != -1:
                        Jxm1ym1 = index_2h[yjm1_2h, xim1_2h]
                
                
                if (~xi_2h[0].any() and yj_2h[0].any()): #in horizontal not vertical
                    if(G_h[gj_h, gi_h - 1] == 0):#left wall
                        I_2h_to_h[i, Jxp1] = 1
                    elif(G_h[gj_h, gi_h + 1] == 0): #right wall
                        I_2h_to_h[i, Jxm1] = 1
                    elif(G_h[gj_h, gi_h - 1] == 3): #left window
                        I_2h_to_h[i, Jxp1] = 0.5
                        I_2h_to_h_win[i, 0] = 0.5
                    elif(G_h[gj_h, gi_h + 1] == 3): # right window
                        I_2h_to_h[i, Jxm1] = 0.5
                        I_2h_to_h_win[i, 0] = 0.5  
                    else:
                        I_2h_to_h[i, Jxp1] = 0.5
                        I_2h_to_h[i, Jxm1] = 0.5
                
                
                if (~yj_2h[0].any() and xi_2h[0].any()): #in vertical not horizontal
                    if(G_h[gj_h - 1, gi_h] == 0): #bottom wall
                        I_2h_to_h[i, Jyp1] = 1
                    elif(G_h[gj_h + 1, gi_h] == 0): #top wall
                        I_2h_to_h[i, Jym1] = 1
                    elif(G_h[gj_h + 1, gi_h] == 3): # top window
                        I_2h_to_h[i, Jym1] = 0.5
                        I_2h_to_h_win[i, 0] = 0.5
                    elif(G_h[gj_h - 1, gi_h] == 3): # bottom window
                        I_2h_to_h[i, Jyp1] = 0.5
                        I_2h_to_h_win[i, 0] = 0.5
                    else:
                        I_2h_to_h[i, Jyp1] = 0.5
                        I_2h_to_h[i, Jym1] = 0.5
                        
                if(~xi_2h[0].any() and ~yj_2h[0].any()):
                    
                    if ( Jxp1yp1 != -1 and Jxm1yp1 != -1 and \
                        Jxp1ym1 != -1 and Jxm1ym1 != -1):
                        I_2h_to_h[i, Jxm1yp1] = 0.25
                        I_2h_to_h[i, Jxm1ym1] = 0.25
                        I_2h_to_h[i, Jxp1ym1] = 0.25
                        I_2h_to_h[i, Jxp1yp1] = 0.25
                        
                        
                    # sharp edge wall
                    elif(Jxp1yp1 == -1 and Jxp1ym1 != -1 and \
                         Jxm1ym1 != -1 and Jxm1yp1 != -1):#top-right wall
                        if (G_h[gj_h + 1, gi_h] == 0): 
                            I_2h_to_h[i, Jxm1ym1] = 0.5
                            I_2h_to_h[i, Jxp1ym1] = 0.5
                        
                        elif(G_h[gj_h, gi_h + 1] == 0):
                            I_2h_to_h[i, Jxm1yp1] = 0.5
                            I_2h_to_h[i, Jxm1ym1] = 0.5
                            
                        else:
                            I_2h_to_h[i, Jxm1yp1] = 3/8
                            I_2h_to_h[i, Jxm1ym1] = 1/4
                            I_2h_to_h[i, Jxp1ym1] = 3/8
                    
                    elif(Jxp1ym1 == -1 and Jxp1yp1 != -1 and \
                         Jxm1ym1 != -1 and Jxm1yp1 != -1): #bottom-right wall
                        if (G_h[gj_h - 1, gi_h] == 0):
                            I_2h_to_h[i, Jxp1yp1] = 0.5
                            I_2h_to_h[i, Jxm1yp1] = 0.5
                            
                        elif (G_h[gj_h, gi_h + 1] == 0):
                            I_2h_to_h[i, Jxm1yp1] = 0.5
                            I_2h_to_h[i, Jxm1ym1] = 0.5
                            
                        else:
                            I_2h_to_h[i, Jxm1yp1] = 1/4
                            I_2h_to_h[i, Jxm1ym1] = 3/8
                            I_2h_to_h[i, Jxp1yp1] = 3/8
                            
                    elif(Jxm1yp1 == -1 and Jxp1ym1 != -1 and \
                         Jxm1ym1 != -1 and Jxp1yp1 != -1): #top-left wall
                         if (G_h[gj_h + 1, gi_h] == 0): 
                             I_2h_to_h[i, Jxm1ym1] = 0.5
                             I_2h_to_h[i, Jxp1ym1] = 0.5
                             
                         elif (G_h[gj_h, gi_h - 1] == 0):
                             I_2h_to_h[i, Jxp1ym1] = 0.5
                             I_2h_to_h[i, Jxp1yp1] = 0.5
                            
                        
                         else:
                             I_2h_to_h[i, Jxp1yp1] = 3/8
                             I_2h_to_h[i, Jxm1ym1] = 3/8
                             I_2h_to_h[i, Jxp1ym1] = 1/4
                        
                        
                    elif(Jxm1ym1 == -1 and Jxp1ym1 != -1 and \
                         Jxp1yp1 != -1 and Jxm1yp1 != -1): #bottom-left wall
                        
                         if (G_h[gj_h - 1, gi_h] == 0):
                             I_2h_to_h[i, Jxp1yp1] = 0.5
                             I_2h_to_h[i, Jxm1yp1] = 0.5
                         elif (G_h[gj_h, gi_h - 1] == 0):
                             I_2h_to_h[i, Jxp1ym1] = 0.5
                             I_2h_to_h[i, Jxp1yp1] = 0.5
                        
                         else:
                             I_2h_to_h[i, Jxm1yp1] = 3/8
                             I_2h_to_h[i, Jxp1yp1] = 1/4
                             I_2h_to_h[i, Jxp1ym1] = 3/8
                    
                    
                    
                    
                    
                    
                    elif(Jxm1ym1 == -1 and Jxp1ym1 == -1 and \
                         Jxp1yp1 != -1 and Jxm1yp1 == -1): #bottom left corner
                        if(G_h[gj_h, gi_h - 1] == 3):
                            I_2h_to_h[i, Jxp1yp1] = 0.25
                            I_2h_to_h_win[i, 0]   = 0.75
                        else:
                            I_2h_to_h[i, Jxp1yp1] = 1
                    
                        
                    elif(Jxm1ym1 == -1 and Jxp1ym1 == -1 and \
                         Jxp1yp1 == -1 and Jxm1yp1 != -1): #bottom right corner
                        if(G_h[gj_h, gi_h + 1] == 3):
                            I_2h_to_h[i, Jxm1yp1] = 0.25
                            I_2h_to_h_win[i, 0]   = 0.75
                        else:
                            I_2h_to_h[i, Jxm1yp1] = 1
                        
                        
                    elif(Jxm1ym1 == -1 and Jxp1ym1 != -1 and \
                         Jxp1yp1 == -1 and Jxm1yp1 == -1): #top left corner
                        if(G_h[gj_h, gi_h - 1] == 3):
                            I_2h_to_h[i, Jxp1ym1] = 0.5
                            I_2h_to_h_win[i, 0]   = 0.5
                        else:
                            I_2h_to_h[i, Jxp1ym1] = 1
                        
                    elif(Jxm1ym1 != -1 and Jxp1ym1 == -1 and \
                         Jxp1yp1 == -1 and Jxm1yp1 == -1): #top right corner
                        if(G_h[gj_h, gi_h + 1] == 3):
                            I_2h_to_h[i, Jxm1ym1] = 0.5
                            I_2h_to_h_win[i, 0]   = 0.5
                        else:
                            I_2h_to_h[i, Jxm1ym1] = 1
                        
                    
                    
                    
                    
                    elif(Jxm1ym1 == -1 and Jxp1ym1 == -1 and \
                         Jxp1yp1 != -1 and Jxm1yp1 != -1 and \
                         G_h[gj_h - 1, gi_h] != 3): #bottom wall and not window
                        I_2h_to_h[i, Jxp1yp1] = 0.5
                        I_2h_to_h[i, Jxm1yp1] = 0.5
                    
                    elif(Jxm1ym1 != -1 and Jxp1ym1 != -1 and \
                         Jxp1yp1 == -1 and Jxm1yp1 == -1 and \
                            G_h[gj_h + 1, gi_h] != 3 ): #top wall and not window
                        I_2h_to_h[i, Jxm1ym1] = 0.5
                        I_2h_to_h[i, Jxp1ym1] = 0.5
                    
                    elif(Jxm1ym1 == -1 and Jxp1ym1 != -1 and \
                         Jxp1yp1 != -1 and Jxm1yp1 == -1 and \
                         G_h[gj_h, gi_h - 1] != 3): #left wall and not window
                        I_2h_to_h[i, Jxp1ym1] = 0.5
                        I_2h_to_h[i, Jxp1yp1] = 0.5
                        
                    elif(Jxm1ym1 != -1 and Jxp1ym1 == -1 and \
                         Jxp1yp1 == -1 and Jxm1yp1 != -1 and \
                         G_h[gj_h, gi_h + 1] != 3): #right wall and not window
                        I_2h_to_h[i, Jxm1yp1] = 0.5
                        I_2h_to_h[i, Jxm1ym1] = 0.5
                        
                        
                        
                    
                    elif(G_h[gj_h + 1, gi_h] == 3): # top window
                            I_2h_to_h[i, Jxm1ym1]  = 0.25
                            I_2h_to_h[i, Jxp1ym1]  = 0.25
                            I_2h_to_h_win[i, 0]    = 0.5
                    
                    elif(G_h[gj_h - 1, gi_h] == 3): # bottom window
                            I_2h_to_h[i, Jxm1yp1]  = 0.25
                            I_2h_to_h[i, Jxp1yp1]  = 0.25
                            I_2h_to_h_win[i, 0]    = 0.5
                            
                    elif(G_h[gj_h, gi_h - 1] == 3): # left window
                            I_2h_to_h[i, Jxp1yp1]  = 0.25
                            I_2h_to_h[i, Jxp1ym1]  = 0.25
                            I_2h_to_h_win[i, 0]    = 0.5
                    
                    elif(G_h[gj_h, gi_h + 1] == 3): # right window
                            I_2h_to_h[i, Jxm1yp1]  = 0.25
                            I_2h_to_h[i, Jxm1ym1]  = 0.25
                            I_2h_to_h_win[i, 0]    = 0.5
                    
                    
                    
                    
    return sp.sparse.csc_matrix(I_2h_to_h), sp.sparse.csc_matrix(I_2h_to_h_win)


def true_solution_steady(mesh, G, constants, idx):
    dof = np.count_nonzero(G == 1)
    u = np.zeros(dof)
    L = mesh.sizex
    H = mesh.sizey - mesh.dy
    for i in range(dof):
        gj, gi = np.where(idx == i)
        u[i]   = np.sin(np.pi*mesh.x[gi]/L)\
        *np.sin(np.pi*mesh.y[gj]/(2*H)) + 373.16
    return u






