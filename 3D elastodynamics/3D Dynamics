from __future__ import print_function
from fenics import *
import numpy as np
from dolfin import *

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

def update(u, u0, v0, a0, beta, gamma, dt):
    """Update fields at the end of each time step."""

    # Get vectors (references)
    u_vec, u0_vec  = u.vector(), u0.vector()
    v0_vec, a0_vec = v0.vector(), a0.vector()

    # Update acceleration and velocity

    # a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
    a_vec = (1.0/(2.0*beta))*( (u_vec - u0_vec - v0_vec*dt)/(0.5*dt*dt) - (1.0-2.0*beta)*a0_vec )

    # v = dt * ((1-gamma)*a0 + gamma*a) + v0
    v_vec = dt*((1.0-gamma)*a0_vec + gamma*a_vec) + v0_vec

    # Update (u0 <- u0)
    v0.vector()[:], a0.vector()[:] = v_vec, a_vec
    u0.vector()[:] = u.vector()

"""Define mesh and function space"""
mesh = BoxMesh(Point(0, 0, 0), Point(0.1, 0.1, 1.0), 4, 4, 40)
V = VectorFunctionSpace(mesh, 'P', 1)

"""Boundary Conditions"""
# Clamped Boundary Conidition
def clamped_boundary(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[2], 0, tol)

bc_clamped = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)

# Y-translation boundary condition
def boundary_Top(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[2], 1, tol)

bc_Top = DirichletBC(V, Constant((0 , 1.0, 0)), boundary_Top) 

bcs = [bc_clamped, bc_Top] #Combined BC

"""Define Weak Formulation"""
u = TrialFunction(V)
r = TestFunction(V)

E  = 1000.0 #Elastic Modulus
nu = 0.3    #Poisson Ratio
lmbda    = E / (2.0*(1.0 + nu))  #Lambda Factor for consitituve relation
mu = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))  #mu Factor for consitituve relation

# Mass density andviscous damping coefficient
rho = 1.0
eta = 0.1

# Time stepping parameters Newmark method

beta    = 0.25
gamma   = 0.5
num_steps = 100
T       = 10
dt      = T/num_steps
t       = 0.0

# Some useful factors
factor_m1  = rho/(beta*dt*dt)
factor_m2  = rho/(beta*dt)
factor_m3  = rho*(1.0-2.0*beta)/(2.0*beta)

factor_d1  = eta*gamma/(beta*dt)
factor_d2  = eta*(gamma-beta)/beta
factor_d3  = eta*(gamma-2.0*beta)*dt/(2.0*beta)

# Fields from previous time step (displacement, velocity, acceleration)
u0 = Function(V)
v0 = Function(V)
a0 = Function(V)

# External forces (body and applied tractions)
f  = Constant((0.0, 0.0, 0.0))  #Body Force
p = Constant ((0.0, 0.0, 0.0))  #Tration


# Stress tensor
def sigma(r):
    return 2.0*mu*sym(grad(r)) + lmbda*tr(sym(grad(r)))*Identity(len(r))

# Forms
a = factor_m1*inner(u, r)*dx + factor_d1*inner(u, r)*dx \
   +inner(sigma(u), grad(r))*dx

L = dot(f,r)*dx + dot(p,r)*ds

# Time-stepping
u = Function(V)
vtk_file = File("Dynamics.pvd")
while t <= T:

    t += dt

    solve(a == L, u, bcs)
    update(u, u0, v0, a0, beta, gamma, dt)

    # Save solution to VTK format
    vtk_file << (u,t)
