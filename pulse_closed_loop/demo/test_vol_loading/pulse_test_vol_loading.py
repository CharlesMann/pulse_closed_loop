# @Author: charlesmann
# @Date:   2022-02-17T15:39:45+01:00
# @Last modified by:   charlesmann
# @Last modified time: 2022-03-01T11:56:26+01:00



import dolfin
from fenics_plotly import plot

#import pulse
import sys
sys.path.append('/home/fenics/shared/repositories/pulse_closed_loop/pulse_closed_loop/')
import pulse_altered as pulse

import numpy as np

try:
    from dolfin_adjoint import Constant, DirichletBC, Function, Mesh, interpolate
except ImportError:
    from dolfin import Function, Constant, DirichletBC, Mesh, interpolate

gamma_space = "R_0"

geometry = pulse.HeartGeometry.from_file(pulse.mesh_paths["simple_ellipsoid"])

if gamma_space == "regional":
    activation = pulse.RegionalParameter(geometry.cfun)
    target_activation = pulse.dolfin_utils.get_constant(0.2, len(activation))
else:
    activation = Function(dolfin.FunctionSpace(geometry.mesh, "R", 0))
    target_activation = Constant(0.2)

active_model = pulse.ActiveModels.active_stress

matparams = pulse.HolzapfelOgden.default_parameters()
material = pulse.HolzapfelOgden(
    activation=activation,
    active_model=active_model,
    parameters=matparams,
    f0=geometry.f0,
    s0=geometry.s0,
    n0=geometry.n0,
)

# Trying to switch to active stress
material.active.model = 'active_stress'

lvp = Constant(0.0)
lv_marker = geometry.markers["ENDO"][0]
lv_pressure = pulse.NeumannBC(traction=lvp, marker=lv_marker, name="lv")
neumann_bc = [lv_pressure]

base_spring = 0.1
robin_bc = [
    pulse.RobinBC(value=Constant(base_spring), marker=geometry.markers["EPI"][0]),
]

lvv = Constant(0.0)
lv_bc = pulse.Lagrange_LVV(lvv=lvv, marker=lv_marker, name="lvv")

def fix_basal_plane(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    bc = DirichletBC(
        V.sub(0),
        Constant(0.0),
        geometry.ffun,
        geometry.markers["BASE"][0],
    )
    return bc

dirichlet_bc = (fix_basal_plane,)

bcs = pulse.BoundaryConditions(
    dirichlet=dirichlet_bc,
    #neumann=neumann_bc,
    robin=robin_bc,
    lagrange_lv = lv_bc,
)

problem = pulse.MechanicsProblem(geometry, material, bcs)

disp_file = dolfin.File('displacement_vol_ctrl.pvd')
#u, p = problem.state.split(deepcopy=True)
#disp_file << u
#target_cbf = np.load('/home/fenics/shared/pulse_demos/full_cycle/prescribed_cb_force.npy')
#target_cbf/=2
#target_lvp = np.load('/home/fenics/shared/pulse_demos/full_cycle/time_normalized_pressure.npy')
u, p, pendo = problem.state.split(deepcopy=True)
target_lvv = np.linspace(geometry.cavity_volume(u=u),geometry.cavity_volume(u=u)*2,10)
#cbf = np.linspace(0,10,10)
# Keep track of volume
lv_vol = np.nan*np.ones(np.shape(target_lvv)[0])
lv_pressures = np.nan*np.ones(np.shape(target_lvv)[0])


for i  in np.arange(10):
    print("Iteration:",i)

    pulse.iterate.iterate(problem, (lvv), (target_lvv[i]),initial_number_of_steps=5)

    u, p, pendo = problem.state.split(deepcopy=True)

    #u_int = interpolate(u, dolfin.VectorFunctionSpace(geometry.mesh, "CG", 1))
    #mesh = Mesh(geometry.mesh)
    #dolfin.ALE.move(mesh, u_int)

    disp_file << u
    lv_vol[i] = geometry.cavity_volume(u=u)
    lv_pressures[i] = pendo.vector().get_local()[0]

"""for j in np.arange(10):

    pulse.iterate.iterate(problem, (activation),(cbf[j]),initial_number_of_steps=5)

    u, p = problem.state.split(deepcopy=True)
    disp_file << u
    lv_vol[i+j] = geometry.cavity_volume(u=u)"""

np.save('lv_vol.npy',lv_vol)
np.save('lv_pressures.npy',lv_pressures)
