import numpy as np


def rotate(x, y, degrees):
    vector = x + y * 1j
    return vector * np.exp(complex(0, np.deg2rad(degrees)))


def ant_step(current_location, last_location, current_potential, last_potential):
    xy_curr, xy_last, p_curr, p_last = [np.array(arg) for arg in (current_location, last_location, current_potential, last_potential)]

    dxy_last = xy_curr - xy_last
    direction_last = dxy_last / np.linalg.norm(dxy_last)

    dp_last = p_curr - p_last
    reverse = int(dp_last > 0) * 2 - 1

    direction_curr = rotate(*(-reverse * direction_last), 30)

    return direction_curr.real * 0.1, direction_curr.imag * 0.1
