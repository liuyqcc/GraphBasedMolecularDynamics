from post_process import *


def plot_GMD():
    k = 627.5095
    # path to data.out
    path = r'data.out'
    quant_num = 1
    state_name_list, state_index_list, energy_list, geom_list = read_data(path, quant_num=quant_num)
    radius_table, ele_table = get_radius_table()
    ref_CH = radius_table['C'] + radius_table['H']
    ref_CC = radius_table['C'] + radius_table['C']
    # geom_1_org = geom_list[0]
    geom_1_org = -(geom_list[0]-1.8*ref_CC) + geom_list[1] + geom_list[2] - (geom_list[3]-1.8*ref_CH)

    e_list_time = energy_list[0]
    reactant_energy = e_list_time[0]
    e_list_time = e_list_time - reactant_energy
    sort_index = np.argsort(geom_1_org)[::-1]
    geom_1_geom = geom_1_org[sort_index]
    e_list_geom = e_list_time[sort_index]
    e_processed_list, correspond_index_geom = minimum_filter(e_list_geom, x_list=geom_1_geom, half_step_window=0.1)
    min_coord_list_geom, TS_coord_list_geom = scan_minimum_TS(e_processed_list)
    min_coord_list_geom = [correspond_index_geom[i] for i in min_coord_list_geom]
    TS_coord_list_geom = [correspond_index_geom[i] for i in TS_coord_list_geom]

    plt.margins(0)
    plt.tight_layout(pad=0)
    plt.gca().invert_xaxis()

    # plt.plot(x, e_list, c='black', alpha=0.5)

    # plt.plot(geom_1, e_list*k, c='black', alpha=0.5)
    e_list_time_2 = np.array(energy_list[0])
    e_list_time_2 = e_list_time_2 - reactant_energy
    e_list_geom_2 = e_list_time_2[sort_index]
    plt.plot(geom_1_geom, e_list_geom*k, c='black', alpha=0.5)
    plt.plot(geom_1_geom, e_list_geom_2*k, c='blue', alpha=0.5)
    plt.plot(geom_1_geom, np.array(e_processed_list)*k, c='red', alpha=0.5)

    e_diff = []
    CI_coord_list = []
    MECI_coord_list = scan_MECI(e_list_geom, CI_coord_list, TS_coord_list_geom)
    print('MECI coord index:', MECI_coord_list)

    result_list = analy_traj(e_list_geom, CI_coord_list, [], geom_1_geom, sort_index=sort_index, half_time_window=0.1, de_threshold=1)
    for i in range(len(result_list)):
        index_time, index_geom, species_name = result_list[i]
        print(result_list[i])
        coord_geom = geom_1_geom[index_geom]
        if species_name == 'Min' or species_name == 'MinCI':
            color = 'orange'
        elif species_name == 'MECI':
            color = 'red'
        elif species_name == 'TS':
            color = 'blue'
        plt.scatter(coord_geom, e_list_geom[index_geom]*k, c=color)
    plt.show()


if __name__ == "__main__":
    plot_GMD()