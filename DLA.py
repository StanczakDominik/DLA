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


class DLA2D:
    DIM = 2
    @classmethod
    def displacement(cls, loc = 1):
        distance = loc
        theta= random.random()*2*np.pi
        x = math.cos(theta) * distance
        y = math.sin(theta) * distance
        return np.array([x, y], dtype=np.float16)

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

    def iterate_particle(self,
                         NT = 100000,
                         ):
        spawn_distance = self.max_distance + 5 
        theta = random.random() * 2 * np.pi
        x = math.cos(theta) * spawn_distance
        y = math.sin(theta) * spawn_distance
        particle = np.array([x, y], dtype=np.float16)

        while True:
            displaced_particle = particle + self.displacement()
            particle_radius = (displaced_particle**2).sum()**0.5
            distance, neighbor_index = self.tree.query(displaced_particle, 1)

            if distance < self.R:  # we have found a neighbour!
                if self.max_distance < particle_radius:
                    self.max_distance = particle_radius
                return displaced_particle, neighbor_index
            elif spawn_distance < particle_radius:
                theta = random.random() * 2 * np.pi
                x = math.cos(theta) * spawn_distance
                y = math.sin(theta) * spawn_distance
                particle = np.array([x, y], dtype=np.float16)
            else:  # run next iteration
                particle = displaced_particle

    def make_fractal(self, N_particles = N_particles, ):
        for N in tqdm.tqdm(range(N_particles)):
            particle, neighbor_index = self.iterate_particle()
            self.particles.append(particle)
            self.tree = cKDTree(self.particles)
            self.connections.append((N+self.num_starters, neighbor_index))

    def plot_mass_distribution(self, Rmin=0.1, Rmax=0.9):
        import matplotlib.pyplot as plt
        distances = np.linalg.norm(self.particles, axis=1)
        minimum = -2
        maximum = np.log(distances.max())
        span = maximum - minimum
        R1 = np.exp(Rmin * span + minimum)
        R2 = np.exp(Rmax * span + minimum)
        Rs = np.logspace(minimum, maximum, 1000)
        Ns = np.array([len(self.tree.query_ball_point(np.zeros(self.DIM), R)) for R in Rs])
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

class DLA3D(DLA2D):
    DIM=3
    @classmethod
    def displacement(cls, loc = 3, scale=1/10):
        distance = np.abs(np.random.normal(loc=loc, scale=scale))
        theta= np.random.random()*2*np.pi
        phi = np.random.random() * np.pi    
        x = np.cos(theta) * np.sin(phi) * distance
        y = np.sin(theta) * np.sin(phi) * distance
        z = np.cos(phi) * distance
        return np.array([x, y, z])
    
    def plot_positions(self):
        from mayavi import mlab
        tqdm.tqdm.write("Plotting...")
        lines = [(self.particles[par1], self.particles[par2]) for par1, par2 in self.connections]
        x, y, z = np.vstack(self.particles).T
        for (x1, y1, z1), (x2, y2, z2) in tqdm.tqdm(lines):
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2]) 
        mlab.show()


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

def main():
    d = create_fractal(1, int(4e4)+1,
                       R = 1/2,
                       # force_new = True,
                       )
    d.plot_particles()
    d.plot_mass_distribution(0.06, 0.6)

if __name__ == "__main__":
    main()


