import numpy as np
import sys
import logging
import traceback


class BondGuidedBias():
    def __init__(self, target_value, type_list):
        self.cv_list = []
        # push forward distance
        # self.bias_center_offset = np.sqrt(0.05) # angs
        raise_energy = 5  # kcal/mol
        raise_distance = 0.2  # angs
        k0 = raise_energy / (raise_distance**2 / 2) # kcal/mol/angs
        self.bias_center_offset = raise_distance # angs
        self.bias_center_offset = self.bias_center_offset / 0.52
        # target bond length
        self.target_value = target_value
        # type_list: breoken: -1, formation: 1
        self.type_list = type_list
        # k0 value
        self.k0 = k0 / 627.5095 * (0.52)**2
        # angs
        self.rho = 0.1
        # restrain
        self.restrain_index = []
        self.restrain_length = []
        self.restrain_k = 500
        self.restrain_k = self.restrain_k / 627.5095 * (0.52)**2
        self.restrain_scale = 1.3
        self.restrain_bias_k = 1

    def get_force_constant(self, cv_value):
        k = 0
        # update force constant
        for cv_history in self.cv_list:
            # print('CV:', cv_history)
            k += 2 / (1+np.exp((cv_history-cv_value)/self.rho))
        # print('K before:', k)
        k = k * self.k0
        return k

    def get_cv_value(self, bond_length_list):
        cv = 0
        for i in range(len(bond_length_list)):
            dR = (bond_length_list[i] - self.target_value[i]) * self.type_list[i]
            if dR < 0:
                dR = 0
            cv += dR
        return cv

    def add_bias(self, cv):
        self.cv_list.append(cv)

    def get_bias_force_old(self, bond_length_list, bond_vec_list):
        cv = self.get_cv_value(bond_length_list)
        k = self.get_force_constant(cv)
        # print('k:', k)
        F_bias = np.zeros_like(bond_vec_list[0])
        if len(self.cv_list) == 0:
            E_bias = 0
            return (E_bias, F_bias)
        min_cv = min(self.cv_list)
        # E_bias = 1/2 * k * (cv-(self.cv_list[-1]-self.bias_center_offset))**2
        E_bias = 1/2 * k * (cv-(min_cv-self.bias_center_offset))**2
        # cv_distance = cv - (self.cv_list[-1]-self.bias_center_offset)
        cv_distance = cv - (min_cv-self.bias_center_offset)
        if cv_distance < 0:
            cv_distance = 0
        for i in range(len(bond_length_list)):
            dR = (bond_length_list[i] - self.target_value[i]) * self.type_list[i]
            if dR < 0:
                continue
            else:
                F_bias = F_bias + k * bond_vec_list[i] * self.type_list[i] * cv_distance
        return (E_bias, F_bias)

    def get_bias_force(self, bond_length_list, bond_vec_list):
        cv = self.get_cv_value(bond_length_list)
        k = self.k0
        E_bias = 0
        F_bias = np.zeros_like(bond_vec_list[0])
        if len(self.cv_list) == 0:
            E_bias = 0
            return (E_bias, F_bias)

        for cv_history in self.cv_list:
            if cv >= cv_history-self.bias_center_offset:
                # E_bias += 1/2 * k * (cv-(cv_history-self.bias_center_offset))**2 / len(bond_length_list)
                E_bias += 1/2 * k * (cv-(cv_history-self.bias_center_offset))**2
                for i in range(len(bond_length_list)):
                    if (bond_length_list[i] - self.target_value[i]) * self.type_list[i] > 0:
                        # F_bias = F_bias + k * bond_vec_list[i] * self.type_list[i] * (cv-(cv_history-self.bias_center_offset)) / len(bond_length_list)
                        F_bias = F_bias + k * bond_vec_list[i] * self.type_list[i] * (cv-(cv_history-self.bias_center_offset))

        return (E_bias, F_bias)

    def set_restrain_matrix(self, react_coord, restrain_mat):
        """
        Restrain all bonds in restrain_mat
        """
        res_mat = np.triu(restrain_mat, k=1)
        res_index = np.where(res_mat == 1)
        res_index = [[i, j] for i,j in zip(*res_index)]
        p = react_coord.reshape(-1, 3)
        res_length = []
        for i in range(len(res_index)):
            l = np.linalg.norm(p[res_index[i][0]] - p[res_index[i][1]])
            res_length.append(l * self.restrain_scale)
        self.restrain_index = res_index
        self.restrain_length = res_length

    def set_restrain_table(self, react_coord, restrain_table):
        """
        Restrain all bonds in restrain_mat
        """
        res_index = restrain_table
        p = react_coord.reshape(-1, 3)
        res_length = []
        for i in range(len(res_index)):
            l = np.linalg.norm(p[res_index[i][0]] - p[res_index[i][1]])
            res_length.append(l * self.restrain_scale)
        self.restrain_index = res_index
        self.restrain_length = res_length

    def get_restrain_force(self, coord):
        p = coord.reshape(-1, 3)
        F = np.zeros_like(coord)
        F = F.flatten()
        vec = np.zeros_like(p)
        for i in range(len(self.restrain_index)):
            length = np.linalg.norm(p[self.restrain_index[i][0]] - p[self.restrain_index[i][1]])
            if length > self.restrain_length[i]:
                vec[:] = 0
                vec = vec.reshape(p.shape)
                try:
                    vec[self.restrain_index[i][0]] = p[self.restrain_index[i][1]] - p[self.restrain_index[i][0]]
                except Exception:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    err_info = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    logging.info(' '.join(err_info))
                    logging.info('vec:')
                    logging.info(vec)
                    logging.info('p')
                    logging.info(p)
                    logging.info(self.restrain_index[i][0], self.restrain_index[i][1], self.restrain_index[i][0])
                    exit()
                vec[self.restrain_index[i][1]] = -vec[self.restrain_index[i][0]]
                vec = vec.flatten()
                vec = vec / np.linalg.norm(vec)
                F = F + vec * self.restrain_k * self.restrain_bias_k * (length - self.restrain_length[i])
        return F

    def add_restrain_bias(self):
        self.restrain_bias_k += 1



class BondGuidedBias_LongShortRange():
    def __init__(self, target_value, type_list):
        self.cv_list = []
        # push forward distance
        self.bias_center_offset_short_range = 0.2 # angs
        self.bias_center_offset_short_range = self.bias_center_offset_short_range/ 0.52
        self.bias_center_offset_long_range = 0.5 # angs
        self.bias_center_offset_short_range = self.bias_center_offset_long_range / 0.52
        self.bias_center_offset = max(self.bias_center_offset_long_range, self.bias_center_offset_short_range)
        # target bond length
        self.target_value = target_value
        # type_list: breoken: -1, formation: 1
        self.type_list = type_list
        # k0 value
        k_short = 100  # kcal/mol/(angs)^2
        k_long = 2  # kcal/mol/angs
        self.k_short = k_short / 627.5095 * (0.52)**2
        self.k_long = k_long / 627.5095 * 0.52
        # angs
        self.rho = 0.1
        # restrain
        self.restrain_index = []
        self.restrain_length = []
        self.restrain_k = 500
        self.restrain_k = self.restrain_k / 627.5095 * (0.52)**2
        self.restrain_scale = 1.3
        self.restrain_bias_k = 1

    def get_force_constant(self, cv_value):
        k = 0
        # update force constant
        for cv_history in self.cv_list:
            # print('CV:', cv_history)
            k += 2 / (1+np.exp((cv_history-cv_value)/self.rho))
        # print('K before:', k)
        k = k * self.k_short
        return k

    def get_cv_value(self, bond_length_list):
        cv = 0
        for i in range(len(bond_length_list)):
            dR = (bond_length_list[i] - self.target_value[i]) * self.type_list[i]
            if dR < 0:
                dR = 0
            cv += dR
        return cv

    def add_bias(self, cv):
        self.cv_list.append(cv)

    def get_bias_force(self, bond_length_list, bond_vec_list):
        cv = self.get_cv_value(bond_length_list)
        E_bias = 0
        E_long = 0
        E_short = 0
        F_bias = np.zeros_like(bond_vec_list[0])
        if len(self.cv_list) == 0:
            E_bias = 0
            return (E_bias, F_bias)

        for cv_history in self.cv_list:
            if cv >= cv_history-self.bias_center_offset_short_range:
                E_short = 1/2 * self.k_short* (cv-(cv_history-self.bias_center_offset_short_range))**2 / len(bond_length_list)
                E_bias += E_short
                for i in range(len(bond_length_list)):
                    if (bond_length_list[i] - self.target_value[i]) * self.type_list[i] > 0:
                        F_bias = F_bias + self.k_short * bond_vec_list[i] * self.type_list[i] * (cv-(cv_history-self.bias_center_offset_short_range)) / len(bond_length_list)
            if cv >= cv_history-self.bias_center_offset_long_range:
                E_long = self.k_long * (cv-(cv_history-self.bias_center_offset_long_range)) / len(bond_length_list)
                E_bias += E_long
                for i in range(len(bond_length_list)):
                    if (bond_length_list[i] - self.target_value[i]) * self.type_list[i] > 0:
                        F_bias = F_bias + self.k_long * bond_vec_list[i] * self.type_list[i] / len(bond_length_list)

        return (E_bias, F_bias)

    def set_restrain_matrix(self, react_coord, restrain_mat):
        """
        Restrain all bonds in restrain_mat
        """
        res_mat = np.triu(restrain_mat, k=1)
        res_index = np.where(res_mat == 1)
        res_index = [[i, j] for i,j in zip(*res_index)]
        p = react_coord.reshape(-1, 3)
        res_length = []
        for i in range(len(res_index)):
            l = np.linalg.norm(p[res_index[i][0]] - p[res_index[i][1]])
            res_length.append(l * self.restrain_scale)
        self.restrain_index = res_index
        self.restrain_length = res_length

    def set_restrain_table(self, react_coord, restrain_table):
        """
        Restrain all bonds in restrain_mat
        """
        res_index = restrain_table
        p = react_coord.reshape(-1, 3)
        res_length = []
        for i in range(len(res_index)):
            l = np.linalg.norm(p[res_index[i][0]] - p[res_index[i][1]])
            res_length.append(l * self.restrain_scale)
        self.restrain_index = res_index
        self.restrain_length = res_length

    def get_restrain_force(self, coord):
        p = coord.reshape(-1, 3)
        F = np.zeros_like(coord)
        F = F.flatten()
        vec = np.zeros_like(p)
        for i in range(len(self.restrain_index)):
            length = np.linalg.norm(p[self.restrain_index[i][0]] - p[self.restrain_index[i][1]])
            if length > self.restrain_length[i]:
                vec[:] = 0
                vec = vec.reshape(p.shape)
                try:
                    vec[self.restrain_index[i][0]] = p[self.restrain_index[i][1]] - p[self.restrain_index[i][0]]
                except Exception:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    err_info = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    logging.info(' '.join(err_info))
                    logging.info('vec:')
                    logging.info(vec)
                    logging.info('p')
                    logging.info(p)
                    logging.info(self.restrain_index[i][0], self.restrain_index[i][1], self.restrain_index[i][0])
                    exit()
                vec[self.restrain_index[i][1]] = -vec[self.restrain_index[i][0]]
                vec = vec.flatten()
                vec = vec / np.linalg.norm(vec)
                F = F + vec * self.restrain_k * self.restrain_bias_k * (length - self.restrain_length[i])
        return F

    def add_restrain_bias(self):
        self.restrain_bias_k += 1


if __name__ == "__main__":
    # test cv
    # test = BondGuidedBias([1], [1])
    test = BondGuidedBias_LongShortRange([1], [1])
    cv = test.get_cv_value([2])
    test.add_bias(cv)
    # test.add_bias(cv)
    e, f = test.get_bias_force([2.0], [np.array([1, 0, -1], 'f')])
    print(f)



