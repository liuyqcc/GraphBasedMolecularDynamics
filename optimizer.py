import numpy as np
import os
import logging
from potential.IOfunction import read_xyz, write_xyz
from potential.Constants import Constants
from potential.interfaceMNDO import InterfaceMNDO
from potential.interfaceXTB import InterfaceXTB
from potential.atomic_mass import get_mass
from PropagationClassical import PropagationClassical


class Optimizer():
    def __init__(self, state=0):
        self.minimizer = BFGS()
        self.Const = Constants()
        self.interface = None
        self.max_opt_step = 1000
        self.max_MECI_step = 1000
        self.max_MECI_update_step = 10000
        # converge threshold
        self.converge_max_force = 4.5e-4
        self.converge_rmsd_force = 3.0e-4
        self.converge_max_dx = 1.8e-3
        self.converge_rmsd_dx = 1.2e-3
        self.state = state
        # check if minimization oscillation
        self.max_osc_loop = 30
        self.min_ediff = None
        # if ediff <= ediff_threshold, turn to find MECI
        self.ediff_threshold = 5 / 627.5095
        # branching plane vecs
        self.branch_vec_list = []
        # MD relax step
        self.max_MD_relax_step = 100
        self.opt_type = 'lbfgs'
        self.opt_type = 'bfgs'
        # self.opt_type = 'lbfgs'
        self.mod_eig_val = -1

    def set_min_ediff(self, min_ediff=10):
        self.ediff_threshold = min_ediff / 627.5095

    def set_interface(self, interface: InterfaceMNDO):
        self.interface = interface

    def get_force(self, p):
        """
        Parameters
        ----------
        p: numpy.array (1-D)
            Unit: Bohr
        """
        Ei, Fi, NAC_list = self.interface.run(p * self.Const.bohr2angs, save_err=True, save_nor=False)
        return (Ei, Fi)

    def check_converge(self, force, dx, de=None):
        max_force = np.abs(force).max()
        rmsd_force = np.sqrt(np.mean(force**2))
        max_dx = np.abs(dx).max()
        rmsd_dx = np.sqrt(np.mean(dx**2))
        if de is None:
            de = 0

        if (max_force <= self.converge_max_force
                and rmsd_force <= self.converge_rmsd_force
                and max_dx <= self.converge_max_dx
                and rmsd_dx <= self.converge_rmsd_dx
                and abs(de) <= 1e-6):
            return True
        else:
            return False

    def check_converge_MECI(self, g_sum_proj, g_diff_proj, dx, de=None, factor=1):
        if de is None:
            de = 0.0
        g_sum = g_sum_proj + g_diff_proj
        max_f = np.abs(g_sum).max()
        rmsd_f = np.sqrt(np.mean(g_sum**2))
        max_dx = np.abs(dx).max()
        rmsd_dx = np.sqrt(np.mean(dx**2))
        if (max_f <= self.converge_max_force * factor and
                rmsd_f <= self.converge_rmsd_force * factor
                and max_dx <= self.converge_max_dx
                and rmsd_dx <= self.converge_rmsd_dx
                and abs(de) <= 1e-6):
            return True
        else:
            return False

    def opt_min(self, init_p, state=None, ele=None):
        """
        Optimize local minimum, if energy degereate, switch to find MECI.

        Parameters
        ----------
        init_p: numpy.array (1-D)
            Unit: Angs
        state: int
            Current state, 0 for ground state, 1 for the first excited state

        Returns
        -------
        p: numpy.array
            Final coordinates
        result flag: int
            Result type. 0: local minimum, 1: optimization failed, 2: energy degenerate
        energy_list: list
            Energies of the final result.
        """
        self.minimizer.reset()
        max_save = 10
        if state is None:
            state = self.state
        if state > 0:
            current_state = 1
        else:
            current_state = 0
        logging.info(f'State: {state}')
        logging.info(f'Current state:{current_state}')
        p = np.copy(init_p).flatten() / self.Const.bohr2angs
        osc_loop = 0
        min_energy = None
        min_ediff = None
        # 0: normal finished, 1: abort, 2: need MECI optimization
        finished_flag = 1
        g_list = []
        x_list = []
        de = None
        self.interface.remove_fort()
        for i in range(self.max_opt_step):
            Ei, Fi = self.get_force(p)
            e = Ei[current_state]
            f = Fi[current_state]
            if state >= 1:
                ediff = Ei[current_state] - Ei[current_state-1]
            else:
                ediff = None
            # check if minimization oscillation
            if min_energy is None or min_energy > e:
                min_energy = e
                osc_loop = 0
                min_ediff = ediff
            else:
                osc_loop += 1

            logging.info(f'{i:<3} E: {e:<.6f} |F| {np.sqrt(np.linalg.norm(f)**2/len(f)):<.6f} dE: {Ei[current_state]-Ei[current_state-1]:<.6f} Emin: {min_energy:<.6f} Osc: {osc_loop}')
            if i >= self.max_opt_step-1:
                logging.info('Max loop exceeded, abort.')
                break
            # not converge, now check if we get MECI
            if osc_loop >= self.max_osc_loop and state > 0:
                logging.info('Max osc loop. Checking energy gap...')
                # in oscillation
                if min_ediff <= self.ediff_threshold:
                    logging.info('Switch to MECI. (Small energy gap)')
                    finished_flag = 2
                    break
                else:
                    logging.info(f'Min diffE: {min_ediff}')
                    # MAX_STEP = max(self.minimizer.MAX_STEP-0.02, 0.05)
                    # logging.info(f'Reduce MAX STEP: {self.minimizer.MAX_STEP} {MAX_STEP}')
                    # self.minimizer.MAX_STEP = MAX_STEP
                    osc_loop = 0
            if self.opt_type == 'bfgs':
                new_p = self.minimizer.get_next_coord(p, f, e)
                dx = new_p - p
                # L-BFGS step
                # g_list.append(-f)
                # x_list.append(np.copy(p))
                # while len(g_list) > max_save:
                #     g_list.pop(0)
                #     x_list.pop(0)
                # if len(x_list) > 1:
                #     dx_lbfgs = self.lbfgs_step(x_list, g_list, None)
                # else:
                #     dx_lbfgs = -g_list[-1] * 0.1
                # if np.max(np.abs(dx_lbfgs)) > 0.3:
                #     dx_lbfgs = dx / np.max(np.abs(dx)) * 0.3
                # dx = (dx + dx_lbfgs) / 2
            elif self.opt_type == 'lbfgs':
                g_list.append(-f)
                x_list.append(np.copy(p))
                while len(g_list) > max_save:
                    g_list.pop(0)
                    x_list.pop(0)
                if len(x_list) > 1:
                    dx = self.lbfgs_step(x_list, g_list, None)
                else:
                    dx = -g_list[-1] * 0.1
                if np.max(np.abs(dx)) > 0.3:
                    dx = dx / np.max(np.abs(dx)) * 0.3
            if len(self.minimizer.energy) >= 2:
                de = self.minimizer.energy[-1] - self.minimizer.energy[-2]
            if self.check_converge(f, dx, de):
                logging.info('Minimization finished.')
                finished_flag = 0
                break
            else:
                p = p + dx
            # if ele is not None:
            #     if i == 0:
            #         write_xyz('opt_traj.xyz', [ele], [p * self.Const.bohr2angs], mode='w')
            #     else:
            #         write_xyz('opt_traj.xyz', [ele], [p * self.Const.bohr2angs], mode='a')

        return (p * self.Const.bohr2angs, finished_flag, Ei)

    def init_branching_plane(self, g1, g2):
        """
        Each branching plane is defined by two vectors.
        """
        g_diff = g2 - g1
        g_sum = g1 + g2
        g_diff = g_diff / np.linalg.norm(g_diff)
        g_sum = g_sum / np.linalg.norm(g_sum)
        vec_x = g_diff
        vec_s = g_sum - np.dot(g_sum, g_diff) * g_diff
        vec_s = vec_s / np.linalg.norm(vec_s)
        I = np.identity(len(g1), np.float64)
        for i in range(len(I)):
            P = I - (np.outer(vec_x, vec_x) + np.outer(vec_s, vec_s))
            vec_y = I[i]
            vec_y = P.dot(vec_y)
            if np.linalg.norm(vec_y) > 1e-5:
                vec_y = vec_y / np.linalg.norm(vec_y)
                break
        self.branch_vec_list = [vec_x, vec_y]

    def update_branching_plane(self, g1, g2):
        """
        Update each two-state branching plane.
        """
        if len(self.branch_vec_list) == 0:
            self.init_branching_plane(g1, g2)
        else:
            last_branch_vec = self.branch_vec_list
            old_vec_x = last_branch_vec[0]
            old_vec_y = last_branch_vec[1]

            g_diff = g2 - g1
            g_diff = g_diff / np.linalg.norm(g_diff)
            vec_x = g_diff
            # BPU method
            vec_y = ((np.dot(old_vec_y, vec_x)*old_vec_x - np.dot(old_vec_x, vec_x)*old_vec_y) /
                    np.sqrt(np.dot(old_vec_y, vec_x)**2 + np.dot(old_vec_x, vec_x)**2))
            vec_y = vec_y - np.dot(vec_y, vec_x)*vec_x
            vec_y = vec_y / np.linalg.norm(vec_y)
            self.branch_vec_list = [vec_x, vec_y]

    def opt_MECI_old(self, init_p, ele, state=None):
        """
        Find MECI between state and state-1
        """
        if state is None:
            state = self.state
        max_save = 20
        p = np.copy(init_p) / self.Const.bohr2angs
        I = np.identity(len(p), dtype=np.float64)
        # 1 for composed gradient; 2 for composed step
        opt_type = 1
        f_list = []
        x_list = []
        e_list = []
        de_list = []
        for i in range(self.max_opt_step):
            if i == 0:
                write_xyz('opt.xyz', [ele], [p*self.Const.bohr2angs], [], mode='w')
            else:
                write_xyz('opt.xyz', [ele], [p*self.Const.bohr2angs], [], mode='a')
            Ei, Fi = self.get_force(p)
            e1 = Ei[state-1]
            e2 = Ei[state]
            g1 = -Fi[state-1]
            g2 = -Fi[state]
            de = e2 - e1
            dg = g2 - g1
            gm = (g1 + g2) / 2
            em = (e1 + e2) / 2
            self.update_branching_plane(g1, g2)
            x1, x2 = self.branch_vec_list
            P = I - np.outer(x1, x1) - np.outer(x2, x2)
            g_IS = P.dot(gm)
            # g_IS = gm
            # g_BS = 2 * de * x1
            g_BS = de * dg/np.linalg.norm(dg)
            # g_BS = g_BS * 0
            print("{:<3d}".format(i), "{:<.6f} {:<.6f} {:<.6f}".format(de, em, np.sqrt(np.linalg.norm(g_IS+g_BS)**2/len(g_IS)*3)))

            if self.check_converge_MECI(g_IS, g_BS):
                print('MECI finished.')
                return (p, 0)
            # if opt_type == 1 and i != 0:
            #     if de < 0.005 or last_de-de > 0.01:
            #         print('Type 2')
            #         opt_type = 2
            opt_type = 2
            if opt_type == 1:
                gc = g_IS + g_BS
                f_list.append(-gc)
                x_list.append(p)
                e_list.append(em)
                de_list.append(de)
                while len(f_list) > max_save:
                    f_list.pop(0)
                    x_list.pop(0)
                    e_list.pop(0)
                    de_list.pop(0)
                H = np.identity(len(p))
                if len(f_list) > 2:
                    if de_list[-1] > de_list[-2] and e_list[-1] > e_list[-2]:
                        dx = (x_list[-1] - x_list[-2]) / 2
                        dx = -gc / np.linalg.norm(gc) * np.linalg.norm(dx)
                        f_list.pop(-1)
                        x_list.pop(-1)
                        e_list.pop(-1)
                        de_list.pop(-1)
                    else:
                        H = self.bfgs_update_sequence(f_list, x_list, H)
                        dx = np.linalg.inv(H).dot(f_list[-1])
                else:
                    H = self.bfgs_update_sequence(f_list, x_list, H)
                    dx = np.linalg.inv(H).dot(f_list[-1])
            elif opt_type == 2:
                # type 2 gradient
                dx1 = -g_BS
                if np.max(np.abs(dx1)) > 0.1:
                    dx1 = dx1 / np.max(np.abs(dx1)) * 0.1
                # f_list.append(-g_IS)
                f_list.append(-gm)
                x_list.append(p)
                e_list.append(em)
                de_list.append(de)
                while len(f_list) > max_save:
                    f_list.pop(0)
                    x_list.pop(0)
                    e_list.pop(0)
                    de_list.pop(0)
                H = np.identity(len(p))
                if len(f_list) > 2:
                    if de_list[-1] > de_list[-2] and e_list[-1] > e_list[-2]:
                        dx2 = (x_list[-1] - x_list[-2]) / 2
                        dx2 = -g_IS / np.linalg.norm(g_IS) * np.linalg.norm(dx2)
                        f_list.pop(-1)
                        x_list.pop(-1)
                        e_list.pop(-1)
                        de_list.pop(-1)
                    else:
                        H = self.bfgs_update_sequence(f_list, x_list, H)
                        Hmod = P.dot(H).dot(P) + (I-P).dot(5000*I).dot(I-P)
                        H = P.dot(H).dot(P) + (I-P).dot(5000*I).dot(I-P)
                        dx2 = np.linalg.inv(Hmod).dot(P.dot(f_list[-1]) - P.dot(H.dot(dx1)))
                else:
                    H = self.bfgs_update_sequence(f_list, x_list, H)
                    H = P.dot(H).dot(P) + (I-P).dot(5000*I).dot(I-P)
                    Hmod = P.dot(H).dot(P) + (I-P).dot(5000*I).dot(I-P)
                    # dx2 = np.linalg.inv(H).dot(f_list[-1])
                    dx2 = np.linalg.inv(Hmod).dot(P.dot(f_list[-1]) - P.dot(H.dot(dx1)))
                dx2 = P.dot(dx2)
                dx = dx1 + dx2
            if np.max(np.abs(dx)) > 0.3:
                dx = dx / np.max(np.abs(dx)) * 0.3
            new_p = p + dx
            p = new_p
        print('Max loop exceeded.')
        return (p, 1)

    def lbfgs_step(self, x_list, g_list, Pm=None):
        m = len(g_list) - 1
        if Pm is None:
            s_list = [x_list[i+1]-x_list[i] for i in range(m)]
            y_list = [g_list[i+1]-g_list[i] for i in range(m)]
        else:
            s_list = [Pm.dot(x_list[i+1]-x_list[i]) for i in range(m)]
            y_list = [Pm.dot(g_list[i+1]-g_list[i]) for i in range(m)]
        # rho = [1.0/(y.dot(s)) for s,y in zip(s_list,y_list)]
        rho = []
        none_num = 0
        for i in range(len(s_list)):
            dot_val = s_list[i].dot(y_list[i])
            dot_val2 = s_list[i].dot(g_list[i])
            if dot_val > 1e-6 or dot_val2 < 0:
                rho.append(1 / dot_val)
            else:
                rho.append(None)
                none_num += 1
        # print('LBFGS update:', none_num, len(s_list))
        if Pm is None:
            q = g_list[-1].copy()
        else:
            q = Pm.dot(g_list[-1].copy())
        alpha = [0]*m
        for i in reversed(range(m)):
            if rho[i] is not None:
                alpha[i] = rho[i] * s_list[i].dot(q)
                q -= alpha[i] * y_list[i]
            else:
                alpha[i] = None
        gamma = s_list[-1].dot(y_list[-1]) / y_list[-1].dot(y_list[-1])
        r = gamma * q
        for i in range(m):
            if rho[i] is not None:
                beta = rho[i] * y_list[i].dot(r)
                r += s_list[i] * (alpha[i] - beta)
        if Pm is None:
            dx = -r
        else:
            dx = -Pm.dot(r)
        return dx

    def opt_MECI(self, init_p, ele=None, state=None):
        """
        Find MECI between state and state-1
        """
        if state is None:
            state = self.state
        state_index = self.interface.get_state_index(state)
        max_save = 50
        max_save = 10
        p = np.copy(init_p) / self.Const.bohr2angs
        I = np.identity(len(p), dtype=np.float64)
        g_list = []
        x_list = []
        e_list = []
        de_list = []
        dg_list = []
        ICI1 = self.interface.ICI1_org
        ICI2 = self.interface.ICI2_org

        i = 0
        self.branch_vec_list = []
        self.interface.remove_fort()
        de = None
        for i in range(self.max_MECI_step):
            self.interface.set_act_space(ICI1, ICI2)
            if ele is not None:
                if i == 0:
                    write_xyz('opt.xyz', [ele], [p*self.Const.bohr2angs], [], mode='w')
                else:
                    write_xyz('opt.xyz', [ele], [p*self.Const.bohr2angs], [], mode='a')
            Ei, Fi = self.get_force(p)
            e1 = Ei[state_index-1]
            e2 = Ei[state_index]
            g1 = -Fi[state_index-1]
            g2 = -Fi[state_index]
            de = e2 - e1
            dg = g2 - g1
            gm = (g1 + g2) / 2
            em = (e1 + e2) / 2
            self.update_branching_plane(g1, g2)
            x1, x2 = self.branch_vec_list
            P = I - np.outer(x1, x1) - np.outer(x2, x2)
            g_IS = P.dot(gm)
            # g_BS = de * dg / (np.linalg.norm(dg)**2 + 1e-4)
            g_BS = 2 * de * dg / (np.linalg.norm(dg) + 1e-5)
            # g_BS = 2 * de * dg / (abs(de) + 1e-6)
            # g_BS = 2 * de * dg / (np.linalg.norm(dg)**2 + 1e-5)
            # print("{:<3d}".format(i), "{:<.6f} {:<.6f} {:<.6f}".format(de, em, np.sqrt(np.linalg.norm(g_IS+g_BS)**2/len(g_IS)*3)))
            rmsF = np.sqrt(np.linalg.norm(g_IS+g_BS)**2/len(g_IS))
            # maxF = np.max(np.abs(g_IS+g_BS))
            logging.info(f'{i} dE: {de:<.6f} Eavg: {em:<.6f} |F| {rmsF:<.6f}')
            # g_list.append(g_IS)
            g_list.append(P.dot(gm))
            # g_list.append(gm)
            x_list.append(p)
            e_list.append(em)
            de_list.append(de)
            dg_list.append(g_BS)
            while len(g_list) > max_save:
                g_list.pop(0)
                x_list.pop(0)
                e_list.pop(0)
                de_list.pop(0)
                dg_list.pop(0)
            H = self.bfgs_update_sequence(g_list, x_list, np.copy(I))
            Hc = self.bfgs_update_sequence(dg_list, x_list, np.copy(I))
            Hmod = (P).dot(H).dot(P) + (I-P).dot(5000*I).dot(I-P)
            Hcmod = (I-P).dot(Hc).dot(I-P) + (P).dot(5000*I).dot(P)
            # dx1 = -g_BS
            dx1 = -np.linalg.solve(Hcmod, (I-P).dot(dg_list[-1]))
            dx1 = (I-P).dot(dx1)
            if np.max(np.abs(dx1)) > 0.1:
                dx1 = dx1 / np.max(np.abs(dx1)) * 0.1
            # dx2 = -np.linalg.solve(Hmod, P.dot(g_list[-1]) + P.dot(H.dot(dx1)))
            dx2 = -np.linalg.solve(Hmod, P.dot(g_list[-1]))
            dx2 = P.dot(dx2)
            if np.max(np.abs(dx2)) > 0.1:
                dx2 = dx2 / np.max(np.abs(dx2)) * 0.1
            dx = dx1 + dx2
            if np.max(np.abs(dx)) > 0.15:
                dx = dx / np.max(np.abs(dx)) * 0.15
            if len(e_list) >= 2:
                de = e_list[-1] - e_list[-2]
            if self.check_converge_MECI(g_IS, g_BS, dx, de):
                logging.info('MECI finished.')
                # print('MECI finished.')
                self.interface.reset_act_space()
                return (p*self.Const.bohr2angs, Ei)
            new_p = p + dx
            p = new_p
            if i // self.max_MECI_update_step > 0 and i % self.max_MECI_update_step == 0:
                if ICI1 == self.interface.ICI1_max and ICI2 == self.interface.ICI2_max:
                    pass
                else:
                    self.interface.remove_fort()
                if ICI1 != self.interface.ICI1_max or ICI2 != self.interface.ICI2_max:
                    ICI1 = min(ICI1+1, self.interface.ICI1_max)
                    ICI2 = min(ICI2+1, self.interface.ICI2_max)
                    logging.info(f'Try to extend act space to {ICI1} {ICI2}')
        logging.info(f'MECI failed.')
        self.interface.reset_act_space()
        return None

    def opt_MECP(self, init_p, inter_A, inter_B, state_A, state_B, ele=None):
        """
        Find MECP.
        """
        state_A_index = inter_A.get_state_index(state_A)
        state_B_index = inter_B.get_state_index(state_B)
        max_save = 30
        p = np.copy(init_p) / self.Const.bohr2angs
        I = np.identity(len(p), dtype=np.float64)
        g_list = []
        x_list = []
        e_list = []
        de_list = []
        dg_list = []
        inter_A.remove_fort()
        inter_B.remove_fort()
        self.branch_vec_list = []
        for i in range(self.max_MECI_step):
            if ele is not None:
                if i == 0:
                    write_xyz('opt.xyz', [ele], [p*self.Const.bohr2angs], [], mode='w')
                else:
                    write_xyz('opt.xyz', [ele], [p*self.Const.bohr2angs], [], mode='a')
            Ei_A, Fi_A, _ = inter_A.run(p * self.Const.bohr2angs, save_err=True, save_nor=True)
            Ei_B, Fi_B, _ = inter_B.run(p * self.Const.bohr2angs, save_err=True, save_nor=True)
            e1 = Ei_A[state_A_index]
            g1 = -Fi_A[state_A_index]
            e2 = Ei_B[state_B_index]
            g2 = -Fi_B[state_B_index]
            de = e2 - e1
            dg = g2 - g1
            gm = (g1 + g2) / 2
            em = (e1 + e2) / 2
            # self.update_branching_plane(g1, g2)
            # x1, x2 = self.branch_vec_list
            # P = I - np.outer(x1, x1) - np.outer(x2, x2)
            x1 = dg / np.linalg.norm(dg)
            P = I - np.outer(x1, x1)
            g_IS = P.dot(gm)
            # g_BS = 2 * de * dg / (np.linalg.norm(dg) + 1e-6)
            g_BS = de * dg / (abs(de) + 1e-6)
            # print("{:<3d}".format(i), "{:<.6f} {:<.6f} {:<.6f}".format(de, em, np.sqrt(np.linalg.norm(g_IS+g_BS)**2/len(g_IS)*3)))
            rmsFm = np.sqrt(np.linalg.norm(g_IS)**2/len(g_IS))
            rmsFc = np.sqrt(np.linalg.norm(g_BS)**2/len(g_IS))
            # maxF = np.max(np.abs(g_IS+g_BS))
            logging.info(f'{i} dE: {de:>10.6f} Eavg: {em:<.6f} |Fm| {rmsFm:<.6f} |Fc| {rmsFc:<.6f}')
            g_list.append(P.dot(gm))
            # g_list.append(gm)
            x_list.append(p)
            e_list.append(em)
            de_list.append(de)
            dg_list.append(g_BS)
            while len(g_list) > max_save:
                g_list.pop(0)
                x_list.pop(0)
                e_list.pop(0)
                de_list.pop(0)
                dg_list.pop(0)
            H = self.bfgs_update_sequence(g_list, x_list, np.copy(I))
            Hc = self.bfgs_update_sequence(dg_list, x_list, np.copy(I))
            Hmod = (P).dot(H).dot(P) + (I-P).dot(5000*I).dot(I-P)
            Hcmod = (I-P).dot(Hc).dot(I-P) + (P).dot(5000*I).dot(P)
            dx1 = -np.linalg.solve(Hcmod, (I-P).dot(dg_list[-1]))
            dx1 = (I-P).dot(dx1)
            if np.max(np.abs(dx1)) > 0.1:
                dx1 = dx1 / np.max(np.abs(dx1)) * 0.1
            # dx2 = -np.linalg.solve(Hmod, P.dot(g_list[-1]) + P.dot(H.dot(dx1)))
            # dx2 = -np.linalg.solve(Hmod, P.dot(g_list[-1]))
            dx2 = -np.linalg.solve(Hmod, g_list[-1])
            dx2 = P.dot(dx2)
            if np.max(np.abs(dx2)) > 0.1:
                dx2 = dx2 / np.max(np.abs(dx2)) * 0.1
            dx = dx1 + dx2
            if np.max(np.abs(dx)) > 0.15:
                dx = dx / np.max(np.abs(dx)) * 0.15
            if self.check_converge_MECI(g_IS, g_BS, dx):
                logging.info('MECP finished.')
                return (p*self.Const.bohr2angs, [e1, e2])
            new_p = p + dx
            p = new_p
        logging.info(f'MECP failed.')
        return None

    def opt_MECP_PF(self, init_p, inter_A, inter_B, state_A, state_B, ele=None):
        """
        Find MECP.
        """
        state_A_index = inter_A.get_state_index(state_A)
        state_B_index = inter_B.get_state_index(state_B)
        max_save = 50
        p = np.copy(init_p) / self.Const.bohr2angs
        I = np.identity(len(p), dtype=np.float64)
        g_list = []
        x_list = []
        e_list = []
        de_list = []
        inter_A.remove_fort()
        inter_B.remove_fort()
        self.branch_vec_list = []
        for i in range(self.max_MECI_step):
            if ele is not None:
                if i == 0:
                    write_xyz('opt.xyz', [ele], [p*self.Const.bohr2angs], [], mode='w')
                else:
                    write_xyz('opt.xyz', [ele], [p*self.Const.bohr2angs], [], mode='a')
            Ei_A, Fi_A, _ = inter_A.run(p * self.Const.bohr2angs, save_err=True, save_nor=True)
            Ei_B, Fi_B, _ = inter_B.run(p * self.Const.bohr2angs, save_err=True, save_nor=True)
            e1 = Ei_A[state_A_index]
            g1 = -Fi_A[state_A_index]
            e2 = Ei_B[state_B_index]
            g2 = -Fi_B[state_B_index]
            de = e2 - e1
            dg = g2 - g1
            gm = (g1 + g2) / 2
            em = (e1 + e2) / 2
            G = gm + de * dg * 2 * 5
            rmsF = np.sqrt(np.linalg.norm(G)**2/len(G)*3)
            # maxF = np.max(np.abs(g_IS+g_BS))
            logging.info(f'{i} dE: {de:<.6f} Eavg: {em:<.6f} |G| {rmsF:<.6f}')
            if self.check_converge_MECI(G, G*0):
                logging.info('MECP finished.')
                return (p*self.Const.bohr2angs, [e1, e2])
            # g_list.append(g_IS)
            # g_list.append(P.dot(gm))
            g_list.append(G)
            x_list.append(p)
            e_list.append(em)
            de_list.append(de)
            while len(g_list) > max_save:
                g_list.pop(0)
                x_list.pop(0)
                e_list.pop(0)
                de_list.pop(0)
            H = self.bfgs_update_sequence(g_list, x_list, np.copy(I))
            dx1 = -np.linalg.solve(H, g_list[-1])
            if np.max(np.abs(dx1)) > 0.1:
                dx1 = dx1 / np.max(np.abs(dx1)) * 0.1
            dx = dx1
            new_p = p + dx
            p = new_p
        logging.info(f'MECP failed.')
        return None

    def opt_MECI_PF(self, init_p, ele, state=None):
        """
        Find MECI between state and state-1
        """
        if state is None:
            state = self.state
        max_save = 10
        p = np.copy(init_p) / self.Const.bohr2angs
        I = np.identity(len(p), dtype=np.float64)
        g_list = []
        x_list = []
        e_list = []
        for i in range(self.max_opt_step):
            if i == 0:
                write_xyz('opt.xyz', [ele], [p*self.Const.bohr2angs], [], mode='w')
            else:
                write_xyz('opt.xyz', [ele], [p*self.Const.bohr2angs], [], mode='a')
            Ei, Fi = self.get_force(p)
            e1 = Ei[state-1]
            e2 = Ei[state]
            g1 = -Fi[state-1]
            g2 = -Fi[state]
            de = e2 - e1
            dg = g2 - g1
            gm = (g1 + g2) / 2
            em = (e1 + e2) / 2
            self.update_branching_plane(g1, g2)
            x1, x2 = self.branch_vec_list
            P = I - np.outer(x1, x1) - np.outer(x2, x2)
            G = gm + de * dg * 2
            # print("{:<3d}".format(i), "{:<.6f} {:<.6f} {:<.6f}".format(de, em, np.sqrt(np.linalg.norm(g_IS+g_BS)**2/len(g_IS)*3)))
            print("{:<3d}".format(i), "{:<.6f} {:<.6f} {:<.6f}".format(de, em, np.sqrt(np.linalg.norm(G)**2/len(G)*3)))
            if self.check_converge_MECI(G, G*0):
                print('MECI finished.')
                return (p, 0)
            # f_list.append(-g_IS)
            g_list.append(G)
            x_list.append(p)
            e_list.append(em)
            while len(g_list) > max_save:
                g_list.pop(0)
                x_list.pop(0)
                e_list.pop(0)
            H = np.copy(I)
            H = self.bfgs_update_sequence(g_list, x_list, H)
            dx = -np.linalg.inv(H).dot(g_list[-1])
            if np.max(np.abs(dx)) > 0.3:
                dx = dx / np.max(np.abs(dx)) * 0.3
            new_p = p + dx
            p = new_p
        print('Max loop exceeded.')
        return (p, 1)

    def opt_min_from_MECI(self, init_p, state=None):
        """
        Optimize minimum from MECI

        Parameters
        ----------
        p: numpy.array (1-D)
            Unit: Angs

        Returns
        -------
        p: numpy.array
            Final coordinates
        result flag: int
            Result type. 0: local minimum, 1: optimization failed, 2: energy degenerate
        """
        if state is None:
            if self.state > 0:
                state = self.state-1
            else:
                state = 0
        if state > 0:
            current_state = 1
        else:
            current_state = 0
        p = np.copy(init_p) / self.Const.bohr2angs
        osc_loop = 0
        min_energy = None
        min_ediff = None
        self.minimizer.reset()
        for i in range(self.max_opt_step):
            Ei, Fi = self.get_force(p)
            e = Ei[current_state]
            f = Fi[current_state]
            if state >= 1:
                ediff = Ei[current_state] - Ei[current_state-1]
            else:
                ediff = None
            # check if minimization oscillation
            if min_energy is None or min_energy > e:
                min_energy = e
                osc_loop = 0
                min_ediff = ediff
            else:
                osc_loop += 1

            rmsF = np.sqrt(np.linalg.norm(f)**2/len(f)*3)
            logging.info(f'{i} E: {e:<.6f} |F| {rmsF}:<.6f dE: {Ei[current_state]-Ei[current_state-1]:<.6f} Emin: {min_energy:<.6f} Osc: {osc_loop}')
            if self.check_converge(f):
                logging.info('Minimization finished.')
                return (p * self.Const.bohr2angs, 0, Ei)
            else:
                # not converge, now check if we get MECI
                if osc_loop >= self.max_osc_loop:
                    logging.info('Max osc loop. Checking energy gap...')
                    # in oscillation
                    if min_ediff <= self.ediff_threshold:
                        logging.info('Switch to MECI. (Small energy gap)')
                        return (p * self.Const.bohr2angs, 2, Ei)
                    else:
                        logging.info(f'Min diffE: {min_ediff}')
                        # MAX_STEP = max(self.minimizer.MAX_STEP-0.02, 0.05)
                        # logging.info(f'Reduce MAX STEP: {self.minimizer.MAX_STEP} {MAX_STEP}')
                        # self.minimizer.MAX_STEP = MAX_STEP
                        osc_loop = 0
            new_p = self.minimizer.get_next_coord(p, f)
            p = new_p
        logging.info('Max loop exceeded, abort.')
        return (p * self.Const.bohr2angs, 1, Ei)

    def opt_MECI_MNDO(self, init_p, save_filename=None):
        result = self.interface.run_MECI(init_p, save_err=True, save_nor=True, save_filename=save_filename)
        if result is not None:
            p, e_list = result
            logging.info('MECI finished.')
            return (p, e_list)
        else:
            logging.info('MECI failed.')
            return None

    def bfgs_update(self, dx, dg, old_H):
        H = old_H + np.outer(dg, dg) / np.dot(dg, dx) - np.outer(old_H.dot(dx), dx).dot(old_H) / np.dot(dx, old_H).dot(dx)
        return H

    def bfgs_update_sequence(self, g_list, x_list, init_H):
        H = np.copy(init_H)
        update_num = 0
        for i in range(1, len(g_list)):
            y = g_list[i]- g_list[i-1]
            s = x_list[i] - x_list[i-1]
            sy = s.dot(y)
            if sy < 1e-6:
                continue
            H = self.bfgs_update(s, y, H)
            H = (H + H.T) / 2
            update_num += 1
        # print('update:', update_num)
        return H


class BFGS():
    def __init__(self,
                 max_step=0.15,
                 trust_radius=0.25,
                 curvature_eps=1e-8,
                 damping_alpha=0.2,
                 restart_interval=20,
                 use_line_search=False,
                 energy_func=None,
                 armijo_c1=1e-4):
        """
        Parameters:
          max_step: maximum absolute displacement on any coordinate (Å)
          trust_radius: 2-norm trust radius; step scaled if exceed
          curvature_eps: minimum allowed s^T y before reset
          damping_alpha: Powell damping alpha (typical 0.2)
          restart_interval: after how many updates to reset H to scaled I
          use_line_search: if True and energy_func provided, do Armijo backtracking
          energy_func: callable E(coord) required if use_line_search True
          armijo_c1: Armijo constant
        """
        self.tau_max = 1.0
        self.tau_min = 0.1
        self.max_step = max_step
        self.trust_radius = trust_radius
        self.curvature_eps = curvature_eps
        self.damping_alpha = damping_alpha
        self.restart_interval = restart_interval
        self.use_line_search = use_line_search and (energy_func is not None)
        self.energy_func = energy_func
        self.armijo_c1 = armijo_c1
        self.max_history_step = 10

        self.reset()

    def reset(self):
        self.H = None
        self.step = []
        self.force = []
        self.energy = []
        self.step_num = 0
        self.tau_new = (self.tau_max + self.tau_min) / 2
        self.tau_old = self.tau_new

    def _symmetrize(self, M):
        return 0.5*(M + M.T)

    def _scale_by_max_step(self, step):
        max_abs = np.abs(step).max()
        if max_abs > 0 and max_abs > self.max_step:
            step = step * (self.max_step / max_abs)
        return step

    def _apply_trust_radius(self, step):
        norm = np.linalg.norm(step)
        if norm > self.tau_new and norm > 0:
            step = step * (self.tau_new / norm)
        return step

    def _reset_H(self, s=None, y=None, n=None):
        if s is not None and y is not None:
            denom = max(s.dot(y), 1e-12)
            gamma = max( (y.dot(y))/denom, 1e-6 )
        else:
            gamma = 1
        if n is None:
            n = self.H.shape[0] if self.H is not None else (s.size if s is not None else 1)
        self.H = np.eye(n, dtype=float) * gamma

    def update_trust_radius(self):
        if len(self.step) >= 2:
            dE = self.energy[-1] - self.energy[-2]
            dx = self.step[-1] - self.step[-2]
            f = self.force[-1]
            E_pred = -f.dot(dx)
            if abs(E_pred) <= 1e-12:
                ratio = 1
            else:
                ratio = dE / E_pred
            if ratio > 0.75:
                tau_new = 2 * self.tau_old
            elif ratio < 0.25:
                tau_new = self.tau_old / 4
            else:
                tau_new = self.tau_old
            self.tau_new = max(min(tau_new, self.tau_max), self.tau_min)
            self.tau_old = self.tau_new

    def update_inverse_Hessian(self, s, y):
        s = s.reshape(-1)
        y = y.reshape(-1)
        n = s.size
        if self.H is None:
            self._reset_H(s, y, n)
        sy = s.dot(y)
        if sy < 1e-8:
            return
        rho = 1.0 / sy
        I = np.eye(n)
        V = I - rho * np.outer(s, y)
        self.H = V.dot(self.H).dot(V.T) + rho * np.outer(s, s)
        self.H = self._symmetrize(self.H)
        try:
            vals = np.linalg.eigvalsh(self.H)
            if vals.min() <= 0:
                shift = max(1e-4, -vals.min() + 1e-4)
                self.H += np.eye(n) * shift
        except np.linalg.LinAlgError:
            self._reset_H()

    def get_next_coord(self, coord, force, energy):
        coord = coord.flatten()
        force = force.flatten()
        self.force.append(force.copy())
        self.step.append(coord.copy())
        self.energy.append(energy)
        if len(self.force) > self.max_history_step:
            self.force.pop(0)
            self.step.pop(0)
            self.energy.pop(0)
        self.sequence_update_Hessian(self.force, self.step)

        g = -force
        p = - self.H.dot(g)
        alpha = 1.0
        dx = alpha * p

        self.update_trust_radius()
        dx = self._apply_trust_radius(dx)
        dx = self._scale_by_max_step(dx)

        next_coord = coord + dx
        self.step_num += 1

        return next_coord

    def finalize_update(self, s, y):
        self.update_inverse_Hessian(s, y)
        self.H = self._symmetrize(self.H)

    def sequence_update_Hessian(self, f_list, x_list):
        self._reset_H(None, None, len(f_list[0]))
        for i in range(1, len(f_list)):
            y = -(f_list[i]-f_list[i-1])
            s = x_list[i] - x_list[i-1]
            self.finalize_update(s, y)

    def check_converge(self, dx, force,
                       max_force=4.5e-4,
                       rmsd_force=3.0e-4,
                       max_disp=1.8e-3,
                       rmsd_disp=1.2e-3):
        max_force_val = np.abs(force).max()
        rmsd_force_val = np.sqrt(np.mean(force**2))
        max_disp_val = np.abs(dx).max()
        rmsd_disp_val = np.sqrt(np.mean(dx**2))
        return (max_force_val <= max_force and rmsd_force_val <= rmsd_force and
                max_disp_val <= max_disp and rmsd_disp_val <= rmsd_disp)



def test():
    # ---------------------- quick sanity test ----------------------
    # minimize quadratic E = 0.5 x^T A x, grad = A x, force = -grad
    A = np.array([[3.0, 0.2], [0.2, 2.0]])
    def energy_q(x):
        return 0.5 * x.dot(A.dot(x))
    def grad_q(x):
        return A.dot(x)

    opt = BFGS(max_step=0.2, trust_radius=0.5, use_line_search=False, energy_func=energy_q)
    x = np.array([1.2, -1.0])
    e = energy_q(x)
    f = -grad_q(x)  # force = -grad
    history = []
    for i in range(40):
        x_new = opt.get_next_coord(x, f, e)
        # evaluate new gradient (as would happen in your driver)
        g_new = grad_q(x_new)
        f_new = -g_new
        # prepare s and y for finalize_update
        s = x_new - x
        y = g_new - grad_q(x)  # g_{k+1} - g_k
        opt.finalize_update(s, y)
        history.append((i, energy_q(x_new), np.linalg.norm(g_new)))
        x = x_new
        f = f_new
        print(i, np.linalg.norm(f_new))

    # print(history[:8], history[-1])


def init_MNDO_interface_test(state, mult=1, quant_num=None):
    interface = InterfaceMNDO()
    IMULT = mult
    ICI1 = 4
    ICI2 = 2
    ICI1_max = 6
    ICI2_max = 4
    # 0 for ground state and 1 for first exited state
    if state > 0:
        state_line = f' {state} {state+1}'
        quant_num = 2
    else:
        state_line = f' {state+1}'
        quant_num = 1
    interface.set_head_line('IOP=-6 JOP=-2 IGEOM=1 IFORM=1 KITSCF=1000 +\n'
                            'ICROSS=1 IPRINT=1 IOUTCI=2 +\n'
                            f'KHARGE=0 IMULT={IMULT} MULTCI=0 IUHF=-1 NPRINT=2 +\n'
                            f'NCIGRD={quant_num} KCI=5 MOVO=0 ICI1={ICI1} ICI2={ICI2} +\n'
                            # 'NCIREF=3 MCIREF=0 LEVEXC=2 IROOT=3 LROOT=0 imomap=3 mapthr=1 nsav15=3 +\n'
                            'NCIREF=3 MCIREF=0 LEVEXC=2 IROOT=3 LROOT=0 ICUTG=-1 ICUTS=-1 +\n'
                            'ktrial=11 ipubo=1 \n')
                            # 'NCIREF=3 MCIREF=0 LEVEXC=2 IROOT=3 LROOT=0 \n'
                            # )
    interface.set_end_line(state_line)
    interface.set_head_line_MECI('IOP=-6 JOP=0 IGEOM=1 IFORM=1 KITSCF=1000 +\n'
                                 'ICROSS=3 IPRINT=1 IOUTCI=2 +\n'
                                 f'KHARGE=0 IMULT={IMULT} MULTCI=0 IUHF=-1 NPRINT=2 +\n'
                                 f'NCIGRD={quant_num} KCI=5 MOVO=0 ICI1={ICI1} ICI2={ICI2} +\n'
                                 'NCIREF=3 MCIREF=0 LEVEXC=2 IROOT=3 LROOT=0 MAXRTL=3000\n')
    interface.set_end_line_MECI(state_line + '\n'
                                '5.000   5.000')
                                # '0.3000   0.5000   0.00000   0.00000')
                                # '0.000   0.000\n'
                                # '0')
    interface.set_act_space(ICI1, ICI2)
    interface.set_act_space_max(ICI1_max, ICI2_max)
    interface.init_run()
    interface.set_quant_num(quant_num)
    return interface


def init_xTB_interface_test(state, mult, file_name=1):
    interface = InterfaceXTB()
    interface.set_quant_num(1)
    interface.set_cwd(os.getcwd())
    interface.set_multiplicity(mult)
    interface.set_job_str(f'--grad --gfn 2 --uhf {mult-1}')
    if mult == 1:
        work_name = 'S'
    else:
        work_name = 'T'
    interface.set_working_filename('QM_' + work_name)
    # interface.init_run('/dev/shm', 'xtb_' + str(file_name))
    interface.init_run(os.getcwd(), 'QM_' + work_name)
    return interface


if __name__ == "__main__":
    # test()
    # exit()
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        filename=f'opt.log',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        format='%(asctime)s [%(levelname)s] %(message)s',
        filemode='w'
    )
    input_file = r'input.xyz'
    ele, p_list, _ = read_xyz(input_file)
    p = p_list[0]

    mass_amu = get_mass(ele)
    state = 1
    inter = init_MNDO_interface_test(state)
    inter.set_elements(ele)
    quant_num = 2
    inter.set_data_read_mode(read_mode=1)

    # opt = Optimizer(state)
    # opt.set_interface(inter)
    # opt.opt_min(p)
    # opt.opt_MECI(p, ele)
    # exit()
    opt = Optimizer(state=1)
    # opt.set_interface(inter)
    quant_num = 1
    inter_S = init_MNDO_interface_test(state=1, mult=1, quant_num=quant_num)
    inter_S.set_elements(ele)
    inter_S.set_data_read_mode(read_mode=1)
    inter_S.set_working_filename('QM_S')
    inter_T = init_MNDO_interface_test(state=0, mult=3, quant_num=quant_num)
    inter_T.set_elements(ele)
    inter_T.set_data_read_mode(read_mode=1)
    inter_T.set_working_filename('QM_T')

    inter_S = init_xTB_interface_test(state=1, mult=1)
    inter_S.set_elements(ele)
    inter_S.set_working_filename('QM_S')
    inter_T = init_xTB_interface_test(state=0, mult=3)
    inter_T.set_elements(ele)
    inter_T.set_working_filename('QM_T')
    # opt.opt_MECP_PF(p, inter_S, inter_T, state_A=1, state_B=1, ele=ele)
    opt.opt_MECP(p, inter_S, inter_T, state_A=1, state_B=1, ele=ele)
    exit()
    # OPT MECP
    result = opt.opt_MECI_MNDO(p, 'result.log')
    if result is not None:
        write_xyz('opt.xyz', [ele], [result[0]], [str(item) for item in result[1]])

