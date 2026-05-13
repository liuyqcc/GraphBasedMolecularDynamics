import numpy as np
from potential.Constants import Constants
import logging


def scale_displacement(old_p, new_p, dt, mass, atom_num, target_T):
    Const = Constants()
    dx = new_p - old_p
    v = dx / dt
    DOF = atom_num - 6
    K = np.dot(mass, v**2) / 2              # kinetic energy
    T = 2 * K / DOF / Const.kb              # temperature (unit: K)
    if T > target_T*100:
        lamb = np.sqrt(target_T*2/ T)
        v = lamb * v
        new_p = old_p + v * dt
    return new_p


def scale_velocity(p, v, mass, atom_num, target_T, remove_RT=True, freeze_index=[], DOF=None):
    Const = Constants()
    if DOF is None:
        DOF = atom_num * 3 - 6
    if remove_RT:
        v = remove_translation(v, mass)
        v = remove_rotation(p, v, mass)
    v = v.reshape(-1, 3)
    v[freeze_index] = 0
    v = v.flatten()
    K = np.dot(mass, v**2) / 2              # kinetic energy
    T = 2 * K / DOF / Const.kb              # temperature (unit: K)
    lamb = np.sqrt(target_T/ T)
    return lamb * v


def remove_translation(v, mass):
    P = np.sum((mass*v).reshape(-1, 3), axis=0)
    M = np.sum(mass) / 3
    v = v.reshape(-1, 3) - P / M
    return v.flatten()


def remove_rotation(p, v, mass):
    coord = p.reshape(-1, 3)
    cent_of_mass = np.sum((p * mass).reshape(-1, 3) / (np.sum(mass) / 3), axis=0)
    r = coord - cent_of_mass
    mass_weight_coord = r.flatten() * np.sqrt(mass)
    mass_weight_coord = mass_weight_coord.reshape(-1, 3)
    x = mass_weight_coord[:, 0]
    y = mass_weight_coord[:, 1]
    z = mass_weight_coord[:, 2]
    I = np.array([[np.dot(y, y)+np.dot(z, z), -np.dot(x, y), -np.dot(x, z)],
                  [-np.dot(y, x), np.dot(x, x)+np.dot(z, z), -np.dot(y, z)],
                  [-np.dot(z, x), -np.dot(z, y), np.dot(x, x)+np.dot(y, y)]], np.float64)
    L = np.zeros(3, np.float64)
    v_reshpae = v.reshape(-1, 3)
    for i in range(len(r)):
        L[:] += mass[i*3] * np.cross(r[i], v_reshpae[i])
    w = np.linalg.solve(I, L)
    for i in range(len(r)):
        v_reshpae[i] -= np.cross(w, r[i])
    return v_reshpae.flatten()


def init_velocity(p, atom_num, mass, target_T):
    DOF = atom_num*3 - 6
    Const = Constants()
    # v = np.random.normal(0, 1, (atom_num, 3))
    v = np.random.uniform(-1, 1, (atom_num, 3))
    v = v.flatten()
    v = remove_translation(v, mass)
    v = remove_rotation(p, v, mass)
    K = np.dot(mass, v**2) / 2            # kinetic energy
    T = 2 * K / DOF / Const.kb            # temperature
    lamb = np.sqrt(target_T/T)
    v = lamb * v
    return v

