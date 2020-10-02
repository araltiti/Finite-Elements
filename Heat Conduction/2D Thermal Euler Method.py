"""
FEniCS program: Heat equation with Dirichlet conditions.

  u'= Laplace(u) + f  in the unit square
  u = 300             on the left boundary (x=0)
  u = 310             on the right boundary (x=1)
  
  u = 300             at t = 0 and x < 0.5
  u = 300+20*(x-0.5)  at t =0 and x => 0.5
  f = 0
  
  @author: Abdul Altiti
"""

from __future__ import print_function
from fenics import *
import numpy as np
from dolfin import *

T = 1.0       # final time
num_steps = 100  # number of time steps
dt = T / num_steps # time step size

""" Mesh and Function Space"""

mesh = UnitSquareMesh(10,10)    #Creates 20x20 triangle mesh of a unit square
V = FunctionSpace(mesh, 'P', 1) #Creates a Langrange Function Space of order = 1

"""Define boundary condition"""
# Left (x=0) Boundary Condition set to 310
g_left = Constant(300)

def boundary_left(x, on_boundary):
    tol = 1e-14
    return on_boundary and near(x[0], 0, tol)
   
bc_left = DirichletBC(V, g_left, boundary_left)

# Right (x=1) Boundary Condition set to 310
g_right = Constant(310)
def boundary_right(x, on_boundary):
    tol = 1e-14
    return on_boundary and near(x[0], 1, tol)
    
bc_right = DirichletBC(V,g_right,boundary_right)

bcs = [bc_left, bc_right]   #Both Boundary Conditions

"""Define initial value"""

u_0 = interpolate(Expression("(x[0] < 0.5) ? 300 : 300+20*x[0] - 10", degree = 2), V)   #Initial temperature at time = 0
u_n = interpolate(u_0, V)   #Interpolating u_0


"""Define variational problem"""
u = TrialFunction(V)
v = TestFunction(V)


"""Backward Euler Weak Form"""
a = u*v*dx + dt*dot(grad(u), grad(v))*dx
L = u_n*v*dx

"""Forward Euler Weak Form"""
#a = u*v*dx
#L = u_n*v*dx - dt*dot(grad(u_n),grad(v))*dx

"""Mid Point Euler Weak Form""" 
#a = u*v*dx +dt*dot(grad(v),grad(0.5*u))*dx
#L = u_n*v*dx - dt*dot(grad(v),grad(0.5*u_n))*dx  

vtkfile = File('Backward10Steps.pvd')

"""Time-stepping Loop"""
u = Function(V)
t = 0   #Initial Time
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    solve(a == L, u, bcs)

    #Export Solution as VTK files 
    vtkfile << (u,t) 
    
    # Update previous solution
    u_n.assign(u)
    