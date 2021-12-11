from functions import calc_approx_fun_acceleration, plot_drawing, create_vec_from_csv_file, calc_state

ideal_file_name = "real_cones_acceleration"  # file with ideal/real cones position and their colors
mapped_file_name = "file38"  # file with mapped cones position and their colors
mission_name = "Acceleration"

if __name__ == "__main__":
    # 38 git/7/48

    # Create matrix with mapped position of cones (with their colors)
    mat_mapped_cones = create_vec_from_csv_file(f'~/AGH_Racing_repo/results_in_csv/'
                                                f'acceleration_filtered/{mapped_file_name}.csv')
    mat_ideal_cones = create_vec_from_csv_file(f'~/PythonProjects/SLAM_map_creator/'
                                               f'{ideal_file_name}.csv')
    approx_acc_tuple = calc_approx_fun_acceleration(mat_mapped_cones)
    t = calc_state(mat_ideal_cones, mat_mapped_cones)
    #plot_drawing(mat_mapped_cones, mat_ideal_cones, approx_acc_tuple, mission_name, r='0.2m')
