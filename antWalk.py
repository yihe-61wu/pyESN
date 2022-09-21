import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def rotate(vector, angle):
    v_old = vector[0] + vector[1] * 1j
    v_new = v_old * np.exp(complex(0, angle))
    return np.array([v_new.real, v_new.imag])


class Field():
    def __init__(self, title, noise_level=0):
        self.title = title
        if self.title == 'Gaussian2d':
            self.landscape = multivariate_normal(mean=[0, 0], cov=[[2, -1], [1, 1]])
            self.norm_coeff = 1 / self.landscape.pdf([0, 0])
        elif self.title == 'Gaussian1d':
            self.landscape = multivariate_normal(mean=[0, 0], cov=[[2000, 0], [0, 1]])
            self.norm_coeff = 1 / self.landscape.pdf([0, 0])
        elif self.title == 'TwoTower':
            self.landscape0 = multivariate_normal(mean=[1, 0], cov=[[2, -1], [1, 1]])
            self.landscape1 = multivariate_normal(mean=[-1, -1], cov=[[1, -1], [2, 3]])
            self.norm_coeff = 1 / (self.landscape0.pdf([1, 0]) + self.landscape0.pdf([-1, -1]))
        self.init_potential()

        self.randomise = noise_level > 0
        if self.randomise:
            self.noise_level = noise_level
        else:
            self.noise_level = 0

    def plot_field(self, xrange=[-3, 3], yrange=[-3, 3], precision=0.01):
        x, y = [np.arange(r[0], r[1] + precision, precision) for r in (xrange, yrange)]
        xv, yv = np.meshgrid(x, y)
        xy = np.dstack((xv, yv))

        plt.contourf(x, y, self.get_potential(xy))
        plt.axis('square')
        plot_title = "{} field - local noise level: {}".format(self.title, self.noise_level)
        plt.title(plot_title)

    def init_potential(self):
        self.get_potential = {'Gaussian2d': self.potential_gaussian,
                              'Gaussian1d': self.potential_gaussian,
                              'TwoTower': self.potential_2tower}[self.title]

    def potential_gaussian(self, locations):
        noise = self.prepare_noise()
        return self.landscape.pdf(locations) * self.norm_coeff + noise

    def potential_2tower(self, locations):
        noise = self.prepare_noise()
        return self.landscape0.pdf(locations) + self.landscape1.pdf(locations) + noise

    def prepare_noise(self):
        if self.randomise:
            noise = np.random.randn() * self.noise_level
        else:
            noise = 0
        return noise


class AntInField():
    def __init__(self, lifespan, method, intrinsic_rotation, intrinsic_stepsize, field, init_location, init_direction=None):
        self.method = method
        self.step = {'reverse': self.step_reverse,
                     'turn': self.step_turn}[self.method]
        self.rotation, self.step_size = intrinsic_rotation, intrinsic_stepsize

        self.potential_decrease_count = 0
        self.potential_decrease_upper_limit = 2 * np.pi // self.rotation

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
        self.record_list = 'location', 'direction', 'potential'
        for key, val in zip(self.record_list[:2], (self.location, self.direction)):
            self.record[key] = np.full((self.lifespan, 2), val)
        self.record['potential'] = np.full(self.lifespan, self.potential)

    def plot_step_arrow(self, epoch, length=0.1):
        plt.arrow(*self.record['location'][epoch], *self.record['direction'][epoch] * length,
                  head_width=0.01, head_length=0.01, fc='r', ec='r', alpha=0.3)

    def plot_trajectory(self, plot_arrows=False):
        plt.plot(*self.record['location'].T, c='r')
        plt.scatter(*self.record['location'][0], s=80, facecolors='none', edgecolors='r')
        if plot_arrows:
            duration = self.lifespan
            length = 0.1
        else:
            duration = 1
            length = 0.3

        for epoch in range(duration):
            self.plot_step_arrow(epoch, length)

    def plot_potentials(self):
        plt.plot(self.record['potential'])

    def update_record(self):
        self.age += 1
        if self.age >= self.lifespan:
            print('end of life')
        else:
            for key, val in zip(self.record_list, (self.location, self.direction, self.potential)):
                self.record[key][self.age] = val

    def step_reverse(self):
        ispotincrease = self.potential >= self.record['potential'][self.age - 1]
        self.direction = rotate((int(ispotincrease) * 2 - 1) * self.direction, self.rotation)
        self.location += self.direction * self.step_size
        self.potential = self.field.get_potential(self.location)
        self.update_record()

    def step_turn(self):
        ispotincrease = self.potential >= self.record['potential'][self.age - 1]
        if not ispotincrease:
            self.rotation *= -1
            self.potential_decrease_count +=1
        else:
            self.potential_decrease_count = 0
        if self.potential_decrease_count <= self.potential_decrease_upper_limit:
            self.direction = rotate(self.direction, self.rotation)
            self.location += self.direction * self.step_size
            self.potential = self.field.get_potential(self.location)
            self.update_record()
        else:
            self.step_reverse()
            self.potential_decrease_count = 0

    def walk(self):
        for _ in range(self.lifespan):
            self.step()


if __name__ == "__main__":
    def random_start(distance_range_from_origin=[2, 2.5]):
        low, high = distance_range_from_origin
        distance = np.random.rand() * (high - low) + low
        angle = np.random.rand() * 2 * np.pi
        return rotate([distance, 0], angle)

    field_type = 'Gaussian2d'
    landscape = Field(field_type, noise_level=0)

    duration = 200
    ant = AntInField(duration, 'turn', np.pi/np.e/3, 0.02, landscape, random_start())
    ant.walk()

    plt.figure()
    landscape.plot_field()
    ant.plot_trajectory(True)

    plt.figure()
    ant.plot_potentials()
    plt.grid()
    plt.show()