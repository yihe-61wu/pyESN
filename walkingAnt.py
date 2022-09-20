import numpy as np


def rotate(x, y, degrees):
    vector = x + y * 1j
    return vector * np.exp(complex(0, np.deg2rad(degrees)))


def ant_step(current_location, last_location, current_potential, last_potential, rotation=90, stepsize=0.1, method='fliprot'):
    xy_curr, xy_last, p_curr, p_last = [np.array(arg) for arg in (current_location, last_location, current_potential, last_potential)]

    dxy_last = xy_curr - xy_last
    direction_last = dxy_last / np.linalg.norm(dxy_last)

    dp_last = p_curr - p_last

    if method == 'reverse':
        reverse = int(dp_last < 0) * 2 - 1

        new_direction = rotate(*(reverse * direction_last), rotation)
        direction_curr = np.array([new_direction.real, new_direction.imag])

    elif method == 'fliprot':
        if dp_last > 0:
            new_direction = rotate(*direction_last, rotation)
        else:
            new_direction = -rotate(*direction_last, -rotation)

        direction_curr = np.array([new_direction.real, new_direction.imag]) #+ np.random.rand(2) * 0.01


    return direction_curr * stepsize, dp_last < 0


if __name__ == "__main__":
    from scipy.stats import multivariate_normal
    import matplotlib.pyplot as plt

    landscape = multivariate_normal(mean=[0, 0], cov=[[2, 0], [0, 1]])
    x, y = np.mgrid[-3:3.01:.01, -3:3.01:.01]
    pos = np.dstack((x, y))

    x0 = np.random.randint(2) * 5 - 2.5
    y0 = np.random.rand() * 5 - 2.5
    init_offset = np.random.rand(2) * .2 - .1
    x1, y1 = x0 + init_offset[0], y0 + init_offset[1]

    duration = 100
    xt, yt = np.zeros(duration), np.zeros(duration)
    degree = 20
    pt = np.zeros(duration)
    for t in range(duration):
        xt[t], yt[t] = x1, y1
        p0, p1 = [landscape.pdf(xy) for xy in ([x0, y0], [x1, y1])]
        #p0, p1 = y0, y1
        s, flip = ant_step([x1, y1], [x0, y0], p0, p1, rotation=degree)
        if flip:
            degree *= -1

        x0, y0 = x1, y1
        x1 += s[0]
        y1 += s[1]

        pt[t] = p1


    print(xt[0], yt[0])
    plt.contourf(x, y, landscape.pdf(pos))
    plt.plot(xt, yt, c='r')
    plt.axis('square')
    plt.figure()
    plt.plot(pt)
    plt.show()