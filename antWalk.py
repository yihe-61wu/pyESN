import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def rotate(vector, angle):
    v_old = vector[0] + vector[1] * 1j
    v_new = v_old * np.exp(complex(0, angle))
    return np.array([v_new.real, v_new.imag])


class Field():
    def __init__(self, title, noise=0):
        self.title = title
        if self.title == 'Gaussian':
            self.landscape = multivariate_normal(mean=[0, 0], cov=[[2, -1], [1, 1]])
            self.norm_coeff = 1 / self.landscape.pdf([0, 0])
            self.get_potential = self.potential_gaussian

        self.randomise = noise > 0
        if self.randomise:
            self.noise = noise
        else:
            self.noise = 0

    def plot_field(self, xrange=[-3, 3], yrange=[-3, 3], precision=0.01):
        x, y = [np.arange(r[0], r[1] + precision, precision) for r in (xrange, yrange)]
        xv, yv = np.meshgrid(x, y)
        xy = np.dstack((xv, yv))

        plt.contourf(x, y, self.get_potential(xy))
        plt.axis('square')
        plot_title = "{} field - local noise level: {}".format(self.title, self.noise)
        plt.title(plot_title)

    def potential_gaussian(self, locations):
        return self.landscape.pdf(locations) * self.norm_coeff


class AntInField():
    def __init__(self, lifespan, method, intrinsic_rotation, intrinsic_stepsize, field, init_location, init_direction=None):
        self.method = method
        self.rotation, self.step_size = intrinsic_rotation, intrinsic_stepsize

        self.field = field
        self.location = init_location
        self.potential = self.field.get_potential(self.location)
        if init_direction is None:
            self.direction = rotate([1, 0], np.random.rand() * 2 * np.pi)
        else:
            self.direction = init_direction

        self.lifespan = lifespan
        self.age = 0
        self.record = {}
        for key, val in zip(('location', 'direction'), (self.location, self.direction)):
            self.record[key] = np.full((self.lifespan, 2), val)
        self.record['potential'] = np.full(self.lifespan, self.potential)

    def plot_step_arrow(self, epoch, length=0.1):
        plt.arrow(*self.record['location'][epoch], *self.record['direction'][epoch] * length,
                  head_width=0.1, head_length=0.1, fc='r', ec='r')

    def step_reverse(self):
        old_loc = self.record_location[self.age - 1]
        old_pot = self.record_potential[self.age - 1]
        new_pot = self.field.get_potential(self.location)




if __name__ == "__main__":
    landscape = Field('Gaussian')
    ant = AntInField(10, 'reverse', np.pi/6, 0.01, landscape, [1, 1])
    print(ant.record)

    landscape.plot_field()
    ant.plot_step_arrow(0, 0.5)
    plt.show()