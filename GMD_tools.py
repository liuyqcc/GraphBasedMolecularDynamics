import numpy as np
__k_au2kcalmol = 627.5095


def get_bond_length_vec(coord, index_list):
    p = coord.reshape(-1, 3)
    r_list = []
    v_list = []
    for index in index_list:
        r_list.append(np.linalg.norm(p[index[0]]-p[index[1]]))
        vec = np.zeros_like(p)
        vec[index[0]] = p[index[1]] - p[index[0]]
        vec[index[1]] = -(p[index[1]] - p[index[0]])
        vec = vec / np.linalg.norm(vec)
        v_list.append(vec.flatten())
    return (r_list, v_list)


def check_energy(reactant_energy, energy_list, bias_energy_list, base_energy=None, E_thres=50, E_bias_thres=100):
    """
    Return:
        1: dE is too large. (energy barrier is too large or E-E_base is too large)
        2: E_bias is too large.
    """
    E_thres = E_thres / __k_au2kcalmol
    E_bias_thres = E_bias_thres / __k_au2kcalmol
    if base_energy is not None:
        if min(energy_list) - reactant_energy > E_thres or min(energy_list) - base_energy > E_thres:
                return 1
        if min(bias_energy_list) > E_bias_thres:
            return 2
    else:
        if min(energy_list) - reactant_energy > E_thres:
            return 1
    return 0


def settle_position(p_org, p_new, mass, ra, rb, rc):
    """
    Settle algorithm for water constraints.
    """
    COM = np.sum((p_org.flatten() * mass).reshape(-1, 3) / sum(mass) * 3, axis=0)
    pO, pH1, pH2 = p_org.reshape(-1, 3)
    # new coordinate system
    vec_Y = (pO - COM) / np.linalg.norm(pO - COM)
    vec_X = (pH2 - pH1) / np.linalg.norm(pH2 - pH1)
    vec_X = vec_X - np.dot(vec_X, vec_Y) * vec_Y
    vec_X = vec_X / np.linalg.norm(vec_X)
    vec_Z = np.cross(vec_X, vec_Y)

    COM_ = np.sum((p_new.flatten() * mass).reshape(-1, 3) / sum(mass) * 3, axis=0)
    M = np.array([vec_X, vec_Y, vec_Z]).T
    p_new = np.dot(p_new.reshape(-1, 3) - COM_, M)
    p_standard = np.array([[0, ra, 0],
                           [-rc, -rb, 0],
                           [rc, -rb, 0]])
    # p_org = np.dot(p_org - COM, M)
    pO, pH1, pH2 = p_standard

    pO_ = p_new[0]
    pH1_ = p_new[1]
    pH2_ = p_new[2]
    # pO_, pH1_, pH2_ = p_new
    sin_phi = pO_[2] / ra
    cos_phi = np.sqrt(1 - sin_phi**2)
    sin_psi = (pH1_[2] - pH2_[2]) / (2 * rc * cos_phi)
    cos_psi = np.sqrt(1 - sin_psi**2)
    Xb2 = -rc * cos_psi
    Yb2 = -rb * cos_phi - rc * sin_psi * sin_phi
    Yc2 = -rb * cos_phi + rc * sin_psi * sin_phi
    AB = pO - pH1
    BC = pH1 - pH2
    AC = pO - pH2
    XB1 = pH1_[0]
    YB1 = pH1_[1]
    XC1 = pH2_[0]
    YC1 = pH2_[1]
    a = Xb2 * BC[0] + Yb2 * (-AB[1]) + Yc2 * (-AC[1])
    b = Xb2 * (-BC[1]) + Yb2 * (-AB[0]) + Yc2 * (-AC[0])
    y = YB1 * (-AB[0]) - XB1 * (-AB[1]) + YC1 * (-AC[0]) - XC1 * (-AC[1])
    sin_theta = (a*y - b*np.sqrt(a**2 + b**2 - y**2)) / (a**2 + b**2)
    cos_theta = np.sqrt(1 - sin_theta**2)
    Rx = np.array([[1, 0, 0],
                   [0, cos_phi, -sin_phi],
                   [0, sin_phi, cos_phi]])
    Ry = np.array([[cos_psi, 0, sin_psi],
                   [0, 1, 0],
                   [-sin_psi, 0, cos_psi]])
    Rz = np.array([[cos_theta, -sin_theta, 0],
                   [sin_theta, cos_theta, 0],
                   [0, 0, 1]])
    # Y first, then X, then Z
    R_rot = Rz.dot(Rx).dot(Ry)
    new_p = np.dot(p_standard, R_rot.T)
    # back into old coordinate system
    new_p = np.dot(new_p, M.T) + COM_
    return new_p


def get_ra_rb_rc(mass_list):
    OH = 0.95720  # read from prmtop file, in Angs
    HH = 1.51360  # read from prmtop file, in Angs
    XH = HH / 2
    YO = np.sqrt(OH**2 - XH**2)
    p = np.array([[0, YO, 0],
                  [-XH, 0, 0],
                  [XH, 0, 0]])
    COM = p.flatten() * mass_list.flatten()
    COM = COM.reshape(-1, 3)
    COM = np.sum(COM, axis=0) / np.sum(mass_list) * 3
    new_p = p - COM
    ra = new_p[0][1]
    rb = new_p[2][1]
    rc = new_p[2][0]
    return (ra, rb, rc, new_p)


def settle_generate_C(p_water, dt, mass):
    mass = mass.flatten()
    vec_AB = p_water[1] - p_water[0]
    vec_BC = p_water[2] - p_water[1]
    vec_CA = p_water[0] - p_water[2]
    vec_AB = vec_AB / np.linalg.norm(vec_AB)
    vec_BC = vec_BC / np.linalg.norm(vec_BC)
    vec_CA = vec_CA / np.linalg.norm(vec_CA)
    cosA = vec_AB.dot(-vec_CA)
    cosB = -vec_AB.dot(vec_BC)
    cosC = vec_CA.dot(vec_BC)
    mA = mass[3*0]
    mB = mass[3*1]
    mC = mass[3*2]
    A = np.array([[mA+mB, mA*cosB, mB*cosA],
                  [mC*cosB, mB+mC, mB*cosC],
                  [mC*cosA, mA*cosC, mC+mA]])
    A_inv = np.linalg.inv(A)
    M = 2 * np.array([mA*mB, mB*mC, mC*mA])
    C = A_inv.dot(M) / dt
    return C


def settle_velocity(p_new, v, C, mass, dt):
    """
    Settle algorithm for water constraints.
    """
    v_reshape = v.reshape(-1, 3)
    a, b, c = p_new.reshape(-1, 3)
    va, vb, vc = v_reshape
    e_ab = (b - a) / np.linalg.norm(b-a)
    e_bc = (c - b) / np.linalg.norm(c-b)
    e_ca = (a - c) / np.linalg.norm(a-c)
    v_ab = vb - va
    v_bc = vc - vb
    v_ca = va - vc
    v_p = [e_ab.dot(v_ab), e_bc.dot(v_bc), e_ca.dot(v_ca)]
    tau = C*v_p
    t_ab, t_bc, t_ca = tau
    v_tmp = np.copy(v).reshape(-1, 3)
    v_tmp[0] = v_tmp[0] + dt/(2*mass[0*3]) * (t_ab*e_ab - t_ca*e_ca)
    v_tmp[1] = v_tmp[1] + dt/(2*mass[1*3]) * (t_bc*e_bc - t_ab*e_ab)
    v_tmp[2] = v_tmp[2] + dt/(2*mass[2*3]) * (t_ca*e_ca - t_bc*e_bc)
    return v_tmp.flatten()


def get_freeze_water_index(topol, p_total, qm_index, radius):
    p = p_total.reshape(-1, 3)
    topology = topol.topology
    water_res_name = ('HOH', 'WAT', 'TIP3', 'SOL')
    qm_coord = p[qm_index]
    qm_center = np.mean(qm_coord, axis=0)
    freeze_list = []
    for res in topology.residues():
        if res.name not in water_res_name:
            continue
        water_atom_index= [atom.index for atom in res.atoms()]
        water_center = p[water_atom_index].mean(axis=0)
        if np.linalg.norm(water_center - qm_center) > radius:
            freeze_list.extend(water_atom_index)
    return freeze_list


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ra = 1
    rb = 0.5
    rc = 1.5
    p_org = np.array([[0, ra, 0],
                      [-rc, -rb, 0],
                      [rc, -rb, 0]])
   
    a, b, c = (0, 15, 10)
    cos_phi = np.cos(np.pi / 180 * a)
    sin_phi = np.sin(np.pi / 180 * a)
    cos_psi = np.cos(np.pi / 180 * b)
    sin_psi = np.sin(np.pi / 180 * b)
    cos_theta = np.cos(np.pi / 180 * c)
    sin_theta = np.sin(np.pi / 180 * c)

    Rx = np.array([[1, 0, 0],
                   [0, cos_phi, -sin_phi],
                   [0, sin_phi, cos_phi]])
    Ry = np.array([[cos_psi, 0, sin_psi],
                   [0, 1, 0],
                   [-sin_psi, 0, cos_psi]])
    Rz = np.array([[cos_theta, -sin_theta, 0],
                   [sin_theta, cos_theta, 0],
                   [0, 0, 1]])
    R_rot = Rz.dot(Ry).dot(Rx)
    p_new = p_org.dot(R_rot.T)
    # p_org = p_org.dot(R_rot.T)

    p_new[0] = p_new[0] + np.array([0.1, -0.03, 1.1])
    p_new[1] = p_new[1] + np.array([-0.1, 0.03, -0.1])
    p_new = p_new + np.array([1, 1, 1])
    mass = np.array([1, 1, 1], 'f')
    mass = mass.repeat(3).flatten()
    p_new_mod = settle_position(p_org, p_new, mass, ra, rb, rc)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for item in p_org:
    #     x, y, z = item
    #     ax.scatter(x, y, z, c='black')
    # for item in p_new:
    #     x, y, z = item
    #     ax.scatter(x, y, z, c='red')
    # for item in p_new_mod:
    #     x, y, z = item
    #     ax.scatter(x, y, z, c='grey')
    # diff = (p_new_mod - p_new).dot(R_rot)
    # print(diff)
    # plt.show()

    v = p_org.dot(R_rot.T) - p_org
    # v[0][0] += 1
    # solve velocity
    dt = 0.1
    C = settle_generate_C(p_org, dt, mass)
    v_new = settle_velocity(p_org, v, C, mass, dt)
    v = v.reshape(-1, 3)
    v_new = v_new.reshape(-1, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for item in p_org:
        x, y, z = item
        ax.scatter(x, y, z, c='black')
    for item in p_org+v:
        x, y, z = item
        ax.scatter(x, y, z, c='red')
    for item in p_org+v_new:
        x, y, z = item
        ax.scatter(x, y, z, c='grey')
    plt.show()
    MOM_1 = np.sum((v.flatten() * mass.flatten()).reshape(-1, 3), axis=0)
    MOM_2 = np.sum((v_new.flatten() * mass.flatten()).reshape(-1, 3), axis=0)
    print(v_new-v)
    print(MOM_1-MOM_2)