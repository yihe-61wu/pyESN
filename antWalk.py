import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def rotate(vector, angle):
    v_old = vector[0] + vector[1] * 1j
    v_new = v_old * np.exp(complex(0, angle))
    return np.array([v_new.real, v_new.imag])


class field():
    def __init__(self, title, noise=0):
        self.title = title
        if self.title == 'Gaussian':
            self.landscape = multivariate_normal(mean=[0, 0], cov=[[2, -1], [1, 1]])
            self.norm_coeff = 1 / self.landscape.pdf([0, 0])
            self.potential = self.potential_gaussian

        self.randomise = noise > 0
        if self.randomise:
            self.noise = noise
        else:
            self.noise = 0

    def plot_field(self, xrange=[-3, 3], yrange=[-3, 3], precision=0.01):
        x, y = [np.arange(r[0], r[1] + precision, precision) for r in (xrange, yrange)]
        xv, yv = np.meshgrid(x, y)
        xy = np.dstack((xv, yv))

        plt.contourf(x, y, self.potential(xy))
        plt.axis('square')
        plot_title = "{} field - local noise level: {}".format(self.title, self.noise)
        plt.title(plot_title)

    def potential_gaussian(self, locations):
        return self.landscape.pdf(locations) * self.norm_coeff



class ant():
    def __init__(self):
        pass


def ant_step(current_location, last_location, current_potential, last_potential, rotation=90, stepsize=0.1, method='reverse'):
    xy_curr, xy_last, p_curr, p_last = [np.array(arg) for arg in (current_location, last_location, current_potential, last_potential)]

    dxy_last = xy_curr - xy_last
    direction_last = dxy_last / np.linalg.norm(dxy_last)

    dp_last = p_curr - p_last

    if method == 'reverse':
        goahead = int(dp_last > 0) * 2 - 1
        new_direction = rotate(goahead * direction_last, rotation)

    elif method == 'switch':
        if dp_last > 0:
            new_direction = rotate(direction_last, rotation)
        else:
            new_direction = rotate(direction_last, -rotation)





    return new_direction * stepsize, dp_last < 0


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

    duration = 200
    xt, yt = np.zeros(duration), np.zeros(duration)
    degree = np.pi / 4
    pt = np.zeros(duration)
    for t in range(duration):
        xt[t], yt[t] = x1, y1
        p0, p1 = [landscape.pdf(xy) + np.random.randn() / 1000 for xy in ([x0, y0], [x1, y1])]
        #p0, p1 = y0, y1
        s, flip = ant_step([x1, y1], [x0, y0], p1, p0, rotation=degree)
        if flip:
            degree *= -1
            print('change')

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