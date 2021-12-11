import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.lines import Line2D
from typing import Tuple

operation_switcher = {
    0: '#85758a',  # undefined
    1: '#0050fc',  # BLUE
    3: '#ff7003',  # ORANGE
    2: '#f4fc00',  # YELLOW
    4: '#85758a'  # undefined
}


def create_vec_from_csv_file(_file_name: str) -> np.ndarray:
    data_frame = pd.read_csv(_file_name, delimiter='\n')
    values_one_row = data_frame.iloc[1:, 0].values  # values in one row together, 'lm x, lm y, color'
    lm_x_coordinate = []
    lm_y_coordinate = []
    color = []
    for i in values_one_row:
        t = i.split(',')
        lm_x_coordinate.append(float(t[0]))
        lm_y_coordinate.append(float(t[1]))
        color.append(int(t[2]))

    if len(lm_x_coordinate) != len(lm_y_coordinate) or len(lm_x_coordinate) != len(color):
        raise Exception('Length og vectors incorrect!')

    lm_x_coordinate = np.array(lm_x_coordinate)
    lm_y_coordinate = np.array(lm_y_coordinate)
    color = np.array(color)
    _mat_data = np.stack([lm_x_coordinate, lm_y_coordinate, color, np.array(range(len(color)))], axis=1)

    return _mat_data


def calc_approx_fun_acceleration(_mat_data: np.ndarray) -> Tuple[np.ndarray, np.poly1d, np.poly1d]:
    """
    Calculate approximation vector of points, which deals with polynomials. In other words: calculate the best
    approximation linear! function to vectors of blue and yellow cones.

    :param _mat_data: array with the following data:
             first col: lm_x_coordinates,
             second col: lm_y_coordinates,
             third col: color of cone,
             fourth col: index of cone.
    :return: _approx_points_vec_x: vector of x points (same for blue and yellow cones)
             _approx_points_vec_y_blue: vector of blue cone's y
             _approx_points_vec_y_yellow: vector of yellow cone's y
    """
    # Create tuple with x_coordinates, y_coordinates of blue and yellow cones
    blue_cones_xy = ([i[0] for i in _mat_data if i[2] == 1.0], [i[1] for i in _mat_data if i[2] == 1.0])
    yellow_cones_xy = ([i[0] for i in _mat_data if i[2] == 2.0], [i[1] for i in _mat_data if i[2] == 2.0])

    # Check length of x coordinates of yellow and blue cones, should be the same (but not always)
    if len(blue_cones_xy[0]) != len(yellow_cones_xy[0]):
        print('Note: Number of blue cones differs from number of yellow cones!')
    else:
        print("Note: Number of blue cones = number of yellow cones")

    # Calculate approx linear function to blue and yellow cones
    _approx_points_blue = np.poly1d(np.polyfit(blue_cones_xy[0], blue_cones_xy[1], 1))
    _approx_points_yellow = np.poly1d(np.polyfit(yellow_cones_xy[0], yellow_cones_xy[1], 1))

    # Vector of x coordinates which deals with polynomials (same for blue and yellow cones)
    _approx_points_vec_x = np.arange(round(min(_mat_data[:, 0]) - 1.0), round(max(_mat_data[:, 0]) + 1.0), step=0.1)

    # Calculate y values for x values
    _approx_points_vec_y_blue = _approx_points_blue(_approx_points_vec_x)
    _approx_points_vec_y_yellow = _approx_points_yellow(_approx_points_vec_x)

    return _approx_points_vec_x, _approx_points_vec_y_blue, _approx_points_vec_y_yellow


def calc_state(_mat_ideal_cones, _mat_mapped_cones):
    # Create tuple with x_coordinates, y_coordinates of blue and yellow cones
    blue_cones_xy_r = ([i[0] for i in _mat_ideal_cones if i[2] == 1.0],
                       [i[1] for i in _mat_ideal_cones if i[2] == 1.0])
    yellow_cones_xy_r = ([i[0] for i in _mat_ideal_cones if i[2] == 2.0],
                         [i[1] for i in _mat_ideal_cones if i[2] == 2.0])
    orange_cones_xy_r = ([i[0] for i in _mat_ideal_cones if i[2] == 3.0],
                         [i[1] for i in _mat_ideal_cones if i[2] == 3.0])
    blue_cones_xy_m = ([i[0] for i in _mat_mapped_cones if i[2] == 1.0],
                       [i[1] for i in _mat_mapped_cones if i[2] == 1.0])
    yellow_cones_xy_m = ([i[0] for i in _mat_mapped_cones if i[2] == 2.0],
                         [i[1] for i in _mat_mapped_cones if i[2] == 2.0])
    orange_cones_xy_m = ([i[0] for i in _mat_mapped_cones if i[2] == 3.0],
                         [i[1] for i in _mat_mapped_cones if i[2] == 3.0])

    distance_real_mapped_blue = np.sqrt((np.array(blue_cones_xy_r[0]) - np.array(blue_cones_xy_m[0])) ** 2 +
                                        (np.array(blue_cones_xy_r[1]) - np.array(blue_cones_xy_m[1])) ** 2)

    distance_real_mapped_yellow = np.sqrt((np.array(yellow_cones_xy_r[0]) - np.array(yellow_cones_xy_m[0])) ** 2 +
                                          (np.array(yellow_cones_xy_r[1]) - np.array(yellow_cones_xy_m[1])) ** 2)

    distance_real_mapped_orange = np.sqrt((np.array(orange_cones_xy_r[0]) - np.array(orange_cones_xy_m[0])) ** 2 +
                                          (np.array(orange_cones_xy_r[1]) - np.array(orange_cones_xy_m[1])) ** 2)

    number_states = len(np.concatenate((distance_real_mapped_orange,
                                        distance_real_mapped_yellow, distance_real_mapped_blue), axis=None))

    distance_real_mapped_blue = np.around(distance_real_mapped_blue, decimals=2, out=None)
    distance_real_mapped_yellow = np.around(distance_real_mapped_yellow, decimals=2, out=None)
    distance_real_mapped_orange = np.around(distance_real_mapped_orange, decimals=2, out=None)

    state_T_blue = [False]*len(distance_real_mapped_blue)
    state_T_yellow = [False]*len(distance_real_mapped_yellow)
    state_T_orange = [False]*len(distance_real_mapped_orange)

    for i, val in enumerate(distance_real_mapped_blue):
        if val <= 0.1:
            state_T_blue[i] = True
    for i, val in enumerate(distance_real_mapped_yellow):
        if val <= 0.11:
            state_T_yellow[i] = True
    for i, val in enumerate(distance_real_mapped_orange):
        if val <= 0.12:
            state_T_orange[i] = True

    return state_T_blue, state_T_yellow, state_T_orange


def plot_drawing(_mat_data: np.ndarray, _mat_data_ideal: np.ndarray, _approx_acc_tuple: Tuple, _mission_name: str,
                 r='0.2m') -> None:
    # Create list with colors (not number but color hex values)
    color_cones = _mat_data[:, 2]
    color_cones_list = []
    for col in _mat_data[:, 2]:
        for key, value in operation_switcher.items():
            if col == key:
                color_cones_list.append(value)

    if len(color_cones_list) != len(color_cones):
        raise Exception('Length of vectors incorrect!')
    else:
        del color_cones

    # Delete all undefined cones
    elem_to_del = []
    for cone in range(len(_mat_data[:, 0])):
        if color_cones_list[cone] == '#85758a' or color_cones_list[cone] == '#85758a':
            elem_to_del.append(cone)
    x_val_cones = np.delete(_mat_data[:, 0], elem_to_del)
    y_val_cones = np.delete(_mat_data[:, 1], elem_to_del)
    color_cones_list = np.delete(color_cones_list, elem_to_del)

    # Create tuple of dicts (name: vector of cone's coordinates, cone's color)
    data_cones = {
        'XY_x': x_val_cones,
        'XY_y': y_val_cones,
        'color': color_cones_list
    }

    # Visualise map and save it to .png file
    fig = plt.gcf()
    # Visualise approximation lines
    plt.plot(_approx_acc_tuple[0], _approx_acc_tuple[1], "-.k", label="Aproksymacja lin. rzeczywistych pachołków")
    plt.plot(_approx_acc_tuple[0], _approx_acc_tuple[2], "-.k")
    # Visualise approximation lines (lower and upper bound) (only for Acceleration !!!)
    plt.plot(_approx_acc_tuple[0] + 0.5, _approx_acc_tuple[1] + 0.5, '-', color='#f76161',
             label="Linia równoległa do aproskymowanej linii (d=0.5m)")
    plt.plot(_approx_acc_tuple[0] + 0.5, _approx_acc_tuple[2] - 0.5, '-', color='#f76161')
    # Visualise approximation lines (lower and upper bound) (only for Autocross !!!)
    # TODO
    # Visualise ideal position of cones
    for i, data in enumerate(zip(_mat_data_ideal[:, 0], _mat_data_ideal[:, 1])):
        j, k = data
        plt.scatter(j, k, marker="o", s=1200, alpha=0.3, color='k',
                    facecolors='none')  # For log ylabel scalling: 100 (1200)
    plt.plot(_mat_data_ideal[:, 0], _mat_data_ideal[:, 1], "*k", label='Rzeczywiste pachołki')
    # Plot one point from each type of cone to get good label
    first_orange_cone_pos = np.where(data_cones['color'] == '#ff7003')[0][0]
    first_blue_cone_pos = np.where(data_cones['color'] == '#0050fc')[0][0]
    first_yellow_cone_pos = np.where(data_cones['color'] == '#f4fc00')[0][0]
    plt.plot(data_cones['XY_x'][first_orange_cone_pos],
             data_cones['XY_y'][first_orange_cone_pos], '.', color='#ff7003', label="Pomarańczowe landmark'i")
    plt.plot(data_cones['XY_x'][first_blue_cone_pos],
             data_cones['XY_y'][first_blue_cone_pos], '.', color='#0050fc', label="Niebieskie landmark'i")
    plt.plot(data_cones['XY_x'][first_yellow_cone_pos],
             data_cones['XY_y'][first_yellow_cone_pos], '.', color='#f4fc00', label="Żółte landmark'i")
    # Plot landmarks
    plt.scatter('XY_x', 'XY_y', c='color', data=data_cones)

    # Create all labels
    handles, _ = plt.gca().get_legend_handles_labels()
    handles = handles[0:6]  # Delete label for scatter (sth is wrong, so that's why i'm using additional 3 plots above)
    line_circle = Line2D([0], [0], label=f'Linia wyznaczająca okrąg o R={r}', color='grey')
    handles.extend([line_circle])

    fig.set_size_inches(25.5, 15.5, forward=True)
    plt.grid()
    scale_type_name = "skala nieliniowa względem osi X"
    plt.title(f"Zbudowana mapa XY dla najlepszej cząsteczki ({scale_type_name})")
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend(handles=handles)
    plt.xticks(np.arange(round(min(x_val_cones) - 1.0), round(max(x_val_cones) + 7.0), step=1))
    plt.yticks(np.arange(round(min(y_val_cones) - 1.0), round(max(y_val_cones) + 1.0), step=1))
    text_coord_x = 1.1
    text_coord_y = [-4.7, -4.9]  # For log ylabel scalling: [-4.7, -4.9] ([-4.7, -5.1])
    plt.text(text_coord_x, text_coord_y[0], r'Misja (trasa): Acceleration', fontsize=17)
    plt.text(text_coord_x, text_coord_y[1], r'Średnia prędkość przejazdu: ...$\frac{m}{s}$', fontsize=17)
    # plt.gca().set_aspect('equal', adjustable='box')  # Same scale of xlabel and ylabel
    plt.show()
    scale_type = "no_linear"
    fig.savefig(f'{_mission_name}_{scale_type}_low_vel.png', dpi=300)
