"""
FEniCS program: Heat equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.
  -nabla_div(J) = f = 0  in the unit square
  u = g             on the boundary
  dot(J,n) = h = 0     
 
  J = -kappa*nabla(u)
  By: Abdul Altiti
  Obtained from the FeniCS demo programs and modified
"""

from __future__ import print_function
from fenics import *
import numpy as np

L = 0.03;
W = 0.08;
kappa = 385;

""" Create mesh and define function space """
mesh = RectangleMesh(Point(0,0), Point(L,W),15,40)  #2D Mesh
#mesh = BoxMesh(Point(0,0,0), Point(L,L,L),5,5,5) #3D Mesh

"""Define function space."""
#This creates a H1 function space from which we can construct the solution (trail) and test function spaces.
V = FunctionSpace(mesh, 'P', 1)

"""Define boundary conditions"""
#g_bot = 300;                                                #Bottom BC for Problems a) and c)
g_bot = Expression('300*(1+(1/3)*x[0])', degree = 2)    #Bottom BC for Problem b)

#Function that defines that the bottom boundary is to be fixed for given g
def boundary_Bot(x, on_boundary):
    tol = 1e-14;
    return on_boundary and near(x[1],0,tol)

bc_bot = DirichletBC(V,g_bot,boundary_Bot)


#g_top = 310;                                                #Top BC for Problems a) and c)
g_top = Expression('310*(1+8*x[0]*x[0])', degree = 2)      #Top BC for Problem b)

#Function that defines that the top boundary is to be fixed for given g
def boundary_Top(x, on_boundary):
    tol = 1e-14;
    return on_boundary and near(x[1],0.08,tol)
    
bc_top = DirichletBC(V,g_top,boundary_Top)
 
"""Define variational problem"""
u = TrialFunction(V)
w = TestFunction(V)
f = Constant((0))
h = Constant((0))

a = kappa*dot(grad(u),grad(w))*dx   #LHS of Weak Form
L = f*w*dx +h*w*ds                  #RHS of Weak Form

""" Solving of the Weak Form"""
u = Function(V)
bcs = [bc_top, bc_bot]
solve(a == L, u, bcs)


#Export Solution as VTK files 
File('temperature.pvd') << u 
