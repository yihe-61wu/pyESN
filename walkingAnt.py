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

    new_direction = rotate(*(-reverse * direction_last), 30)
    direction_curr = np.array([new_direction.real, new_direction.imag])

    return direction_curr * 0.1
