# @Author: charlesmann
# @Date:   2022-02-17T15:39:45+01:00
# @Last modified by:   charlesmann
# @Last modified time: 2022-03-03T13:44:52+01:00



import dolfin
from fenics_plotly import plot

#import pulse
import sys
sys.path.append('/home/fenics/shared/repositories/pulse_closed_loop/pulse_closed_loop/')
import pulse_altered as pulse
import CirculatorySystem

import numpy as np

try:
    from dolfin_adjoint import Constant, DirichletBC, Function, Mesh, interpolate
except ImportError:
    from dolfin import Function, Constant, DirichletBC, Mesh, interpolate

gamma_space = "R_0"

geometry = pulse.HeartGeometry.from_file(pulse.mesh_paths["ellipsoid"])

if gamma_space == "regional":
    activation = pulse.RegionalParameter(geometry.cfun)
    target_activation = pulse.dolfin_utils.get_constant(0.2, len(activation))
else:
    activation = Function(dolfin.FunctionSpace(geometry.mesh, "R", 0))
    target_activation = Constant(0.2)

active_model = pulse.ActiveModels.active_stress

matparams = pulse.HolzapfelOgden.default_parameters()

# Kurtis changing stiffness if fiber direction
matparams["a_f"] = 3.0

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

base_spring = 1.0
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

disp_file = dolfin.File('displacement_wk.pvd')

u, p, pendo = problem.state.split(deepcopy=True)
target_lvv = np.linspace(geometry.cavity_volume(u=u),geometry.cavity_volume(u=u)*2,10)

# Keep track of volume and pressure
lv_vol = np.nan*np.ones(100)
lv_pressures = np.nan*np.ones(100)

aortic_vol = np.nan*np.ones(100)
aortic_pressure = np.nan*np.ones(100)

venous_vol = np.nan*np.ones(100)
venous_pressure = np.nan*np.ones(100)

# Set up windkessel model
circ_params = {
    "model": ["three_compartment_wk"],
    "Cao": [8.467e-2],
    "Cven": [0.83e0],
    "Vart0": [0.016625],
    "Vven0": [0.025],
    "Rao": [2.5e4],
    "Rven": [125e-2],
    "Rper": [1500e3],
    "V_ven": [2.2],
    "V_art": [1.25]
}

circ_system = CirculatorySystem.CirculatorySystem(circ_params)
#target_vol = np.linspace(0.75,1.54,10)

# For now, just assigning an active stress
active_stress = np.load('active_stress_assigned.npy')

try:
    for i  in np.arange(100):
        print("Iteration:",i)

        u, p, pendo = problem.state.split(deepcopy=True)

        pressures_volumes = circ_system.update_compartments(pendo.vector().get_local()[0], geometry.cavity_volume(u=u),0.1)

        aortic_vol[i] = pressures_volumes["V_art"]
        aortic_pressure[i] = pressures_volumes["Part"]

        venous_vol[i] = pressures_volumes["V_ven"]
        venous_pressure[i] = pressures_volumes["Pven"]

        pulse.iterate.iterate(problem, (lvv, activation), (pressures_volumes["V_cav"],active_stress[i]),initial_number_of_steps=5)

        u, p, pendo = problem.state.split(deepcopy=True)

        #u_int = interpolate(u, dolfin.VectorFunctionSpace(geometry.mesh, "CG", 1))
        #mesh = Mesh(geometry.mesh)
        #dolfin.ALE.move(mesh, u_int)

        disp_file << u
        lv_vol[i] = geometry.cavity_volume(u=u)
        lv_pressures[i] = pendo.vector().get_local()[0]

        print("Current LV Vol:", lv_vol[i])
        print("Current LV Pressure:", lv_pressures[i])
except:
    np.save('lv_vol.npy',lv_vol)
    np.save('lv_pressures.npy',lv_pressures)
    np.save('aortic_vol.npy',aortic_vol)
    np.save('aortic_pressure.npy',aortic_pressure)
    np.save('venous_vol.npy',venous_vol)
    np.save('venous_pressure.npy',venous_pressure)

np.save('lv_vol.npy',lv_vol)
np.save('lv_pressures.npy',lv_pressures)
np.save('aortic_vol.npy',aortic_vol)
np.save('aortic_pressure.npy',aortic_pressure)
np.save('venous_vol.npy',venous_vol)
np.save('venous_pressure.npy',venous_pressure)
