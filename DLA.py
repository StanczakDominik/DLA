import numpy as np
from scipy.spatial import cKDTree
import tqdm
import random
from multiprocessing import Pool
import json
import os
import numba
import math
import random

N_particles = 5000

dtype = np.float32

@numba.njit
def radius(particle):
    x, y = particle
    return math.sqrt(x**2 + y**2)

@numba.njit
def position_on_circle(radius):
    theta = random.random() * 2 * np.pi
    x = math.cos(theta) * radius
    y = math.sin(theta) * radius
    particle = np.array([x, y], dtype=dtype)
    return particle

class DLA2D:
    DIM = 2
    @classmethod
    def displacement(cls, loc = 1):
        return np.random.normal(scale=loc, size=cls.DIM).astype(dtype)

    @classmethod
    def displacement_multiple(cls, N, loc = 1):
        return np.random.normal(scale=loc, size=(N, cls.DIM)).astype(dtype)

    def position_on_circle(self, N, radius, off_center = True):
        theta = np.random.random(N) * 2 * np.pi
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        new_samples = np.vstack([x, y]).T
        if off_center:
            new_samples += np.mean(self.particles, axis=0)
        return new_samples

    def __init__(self, num_starters = 1,
                 R = 1/20,
                 Rmin = 1.2,
                 Rmax = 30
                 ):
        self.num_starters = num_starters
        self.connections = []
        self.R = R
        self.Rmin = Rmin
        self.Rmax = Rmax

        if num_starters == 1:
            self.particles = [np.zeros(self.DIM)]
            self.max_distance = 0
        else:
            self.particles = list(np.random.normal(scale = num_starters * R * 5,
                                                   size=(num_starters, self.DIM)))
            self.max_distance = np.linalg.norm(self.particles, axis=1).max()
        self.tree = cKDTree(self.particles)

    def iterate_particle(self):
        spawn_distance = self.max_distance + 5 
        particle = position_on_circle(spawn_distance)

        while True:
            displaced_particle = particle + self.displacement()
            particle_radius = radius(displaced_particle)
            neighbors = self.tree.query_ball_point(displaced_particle, self.R)

            if neighbors:
                if self.max_distance < particle_radius:
                    self.max_distance = particle_radius
                neighbor_index = neighbors[0]
                return displaced_particle, neighbor_index

            elif spawn_distance + 5 < particle_radius:
                particle = position_on_circle(spawn_distance)

            else:  # run next iteration
                particle = displaced_particle

    def make_fractal(self, N_particles = N_particles, ):
        for N in tqdm.tqdm(range(N_particles)):
            particle, neighbor_index = self.iterate_particle()
            self.particles.append(particle)
            self.tree = cKDTree(self.particles)
            self.connections.append((N+self.num_starters, neighbor_index))

    def iterate_multiple(self, N_particles, bunch_size):
        print(f"Bunch size is {bunch_size}")
        self.max_distance = np.max(np.linalg.norm(self.particles, axis=1))
        spawn_distance = self.max_distance + 5 
        # particles = np.vstack([position_on_circle(spawn_distance) for _ in range(bunch_size)])
        particles = self.position_on_circle(bunch_size, spawn_distance)
        indices = np.arange(bunch_size)

        if isinstance(self.particles, np.ndarray):
            self.particles = self.particles.tolist()
        added_particles = 0
        progressbar = tqdm.tqdm(total=N_particles)
        with progressbar:
            while added_particles < N_particles:
                particles += self.displacement_multiple(bunch_size)
                particle_radius = np.linalg.norm(particles, axis=1)
                neighbors = self.tree.query_ball_point(particles, self.R)

                have_neighbors = neighbors.astype(bool)
                if have_neighbors.any():
                    # breakpoint()
                    number_added = have_neighbors.sum()
                    for i, p, r, n in zip(indices[have_neighbors],
                                            particles[have_neighbors],
                                            particle_radius[have_neighbors],
                                            neighbors[have_neighbors]):
                        self.particles.append(p)
                        if self.max_distance < r:
                            self.max_distance = r

                        spawn_distance = self.max_distance + 5 
                        particles[i] = position_on_circle(spawn_distance)
                        for neighbor_index in n:
                            self.connections.append((len(self.particles) + added_particles, neighbor_index))
                        added_particles += 1
                        # progressbar.update(1)
                    # particles[have_neighbors] = self.position_on_circle(number_added, spawn_distance)
                    progressbar.update(number_added)
                    self.tree = cKDTree(self.particles)

                out_of_bounds = (spawn_distance + 5) < particle_radius
                particles[out_of_bounds] = self.position_on_circle(out_of_bounds.sum(), spawn_distance)

        self.particles = np.array(self.particles)

    def plot_mass_distribution(self, Rmin=0.1, Rmax=0.9):
        import matplotlib.pyplot as plt
        center = np.mean(self.particles, axis=0)
        recentered_particles = self.particles - center
        recentered_tree = cKDTree(recentered_particles)
        distances = np.linalg.norm(recentered_particles, axis=1)
        minimum = np.log(distances.min() / 10)
        maximum = np.log(distances.max())
        span = maximum - minimum
        R1 = np.exp(Rmin * span + minimum)
        R2 = np.exp(Rmax * span + minimum)
        R1, R2 = np.quantile(np.log(distances), [0.01, 0.9999999999])
        Rs = np.logspace(minimum, maximum, 1000)
        Ns = np.array([len(recentered_tree.query_ball_point(center, R)) for R in Rs])
        indices = (R1 < Rs) & (Rs < R2)
        plt.loglog(Rs, Ns)
        a_fit, b_fit = np.polyfit(np.log(Rs[indices]), np.log(Ns[indices]), 1)
        title = f"Estimated dimension: {a_fit:.3f}"
        plt.title(title)
        plt.loglog(Rs[indices], Ns[indices], "r", label=title)
        plt.axvline(R1, color='r', linestyle="--")
        plt.axvline(R2, color='r', linestyle="--")
        plt.legend(loc='best')
        plt.xlabel("Radius R")
        plt.ylabel("Number of particles within R")

        plt.show()

    def plot_particles(self):
        import matplotlib.pyplot as plt
        x, y = np.vstack(self.particles).T
        plt.plot(x, y, "k.", alpha=0.2) 
        plt.plot(x[:self.num_starters], y[:self.num_starters], "*", markersize=20)
        # plt.grid()
        plt.show()

    def plot_connections(self):
        import matplotlib.pyplot as plt
        x, y = np.vstack(self.particles).T
        # plt.plot(x[self.num_starters:], y[self.num_starters:], "o")
        tqdm.tqdm.write("Plotting...")
        lines = [(self.particles[par1], self.particles[par2]) for par1, par2 in self.connections]
        for (x1, y1), (x2, y2) in tqdm.tqdm(lines):
            plt.plot([x1, x2], [y1, y2]) 
        plt.plot(x[:self.num_starters], y[:self.num_starters], "*", markersize=20)
        # plt.grid()
        plt.show()

    def save(self, filename):
        with open(filename, "w") as f:
            d = dict(particles=np.vstack(self.particles).tolist(),
                     connections=self.connections,
                     num_starters = self.num_starters,
                     R=self.R,
                     Rmin=self.Rmin,
                     Rmax=self.Rmax)
            json.dump(d, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            d = json.load(f)
        dla = cls(d['num_starters'], d['R'], d['Rmin'], d['Rmax'])
        dla.particles = np.array(d['particles'])
        dla.connections = d['connections']
        dla.tree = cKDTree(dla.particles)
        return dla

class MapDLA(DLA2D):
    def __init__(self, N_procs):
        super().__init__()
        self.N_procs = N_procs

    def generate_particle(self, pool):
        while True:
            particles = list(filter(None, pool.map(self.iterate_particle, range(pool._processes))))
            if len(particles):
                return random.choice(particles)

    def make_fractal(self, N_particles = N_particles):
        with Pool(self.N_procs) as p:
            for N in tqdm.tqdm(range(N_particles)):
                particle, neighbor_index = self.generate_particle(p)
                self.particles.append(particle)
                self.tree = cKDTree(self.particles)
                self.connections.append((N+1, neighbor_index))

def create_fractal(n_starters = 2, n_particles = 5000, force_new = False,
                   *args, **kwargs):
    filename = f"2d_{n_starters}_{n_particles}.json"
    if not force_new and os.path.isfile(filename):
        tqdm.tqdm.write("reading from file...")
        d = DLA2D.load(filename)
    else:
        tqdm.tqdm.write(f"Creating fractal {filename} from scratch")
        d = DLA2D(n_starters, *args, **kwargs)
        d.make_fractal(n_particles)
        d.save(filename)
    return d

def main(plot = False):
    # d = create_fractal(1,
    #                    int(5e3)+5,
    #                    R = 1/2,
    #                    # force_new = True,
    #                    )
    # d.iterate_multiple(int(2e4 - len(d.particles)), 4000)
    d = DLA2D.load("2d_1_multi_20008.json")
    if plot:
        d.plot_particles()
        d.plot_mass_distribution(0.06, 0.6)
    filename = f"2d_{d.num_starters}_multi_{len(d.particles)}.json"
    d.save(filename)
    return d

if __name__ == "__main__":
    main(True)


