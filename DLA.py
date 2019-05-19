import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm as tqdm
import random
from multiprocessing import Pool
import json
import os

N_particles = 5000


class DLA2D:
    DIM = 2
    @classmethod
    def displacement(cls, loc = 3, scale=1/10, center = (0, 0)):
        distance = np.abs(np.random.normal(loc=loc, scale=scale))
        theta= np.random.random()*2*np.pi
        x = np.cos(theta) * distance
        y = np.sin(theta) * distance
        return np.array([x, y]) + np.array(center)

    def __init__(self, num_starters = 1,
                 R = 0.3,
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
        else:
            self.particles = list(np.random.normal(scale = num_starters * R * 5,
                                                   size=(num_starters, self.DIM)))
        self.tree = cKDTree(self.particles)

    def generate_particle(self):
        while True:
            appended_particle = self.iterate_particle()
            if appended_particle is not None:
                return appended_particle

    def iterate_particle(self,
                         NT = 100000,
                         ):
        while True:
            particle = self.displacement(loc=np.linalg.norm(self.particles,
                                                            axis=1).max()*1.05,
                                         scale=1/2,
                                         # center = np.mean(self.particles, axis=0),
                                         )
            if not self.tree.query_ball_point(particle, self.Rmin):
                # make sure we're spawning far away from existing particles
                break

        for i in range(NT):
            displaced_particle = particle + self.displacement()
            distance, neighbor_index = self.tree.query(displaced_particle, 1)

            if distance < self.R:  # we have found a neighbour!
                return displaced_particle, neighbor_index
            elif distance > self.Rmax:  # Particle going too far!
                return None
            else:  # run next iteration
                particle = displaced_particle
        else:
            # Could not find neighbor in allotted time.
            return None

    def make_fractal(self, N_particles = N_particles, ):
        for N in tqdm(range(N_particles)):
            particle, neighbor_index = self.generate_particle()
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

    def plot_positions(self):
        import matplotlib.pyplot as plt
        x, y = np.vstack(self.particles).T
        # plt.plot(x[self.num_starters:], y[self.num_starters:], "o")
        tqdm.write("Plotting...")
        lines = [(self.particles[par1], self.particles[par2]) for par1, par2 in self.connections]
        for (x1, y1), (x2, y2) in tqdm(lines):
            plt.plot([x1, x2], [y1, y2]) 
        plt.plot(x[:self.num_starters], y[:self.num_starters], "*", markersize=20)
        plt.grid()
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
        tqdm.write("Plotting...")
        lines = [(self.particles[par1], self.particles[par2]) for par1, par2 in self.connections]
        x, y, z = np.vstack(self.particles).T
        for (x1, y1, z1), (x2, y2, z2) in tqdm(lines):
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
            for N in tqdm(range(N_particles)):
                particle, neighbor_index = self.generate_particle(p)
                self.particles.append(particle)
                self.tree = cKDTree(self.particles)
                self.connections.append((N+1, neighbor_index))

def create_fractal(n_starters = 2, n_particles = 5000):
    filename = f"2d_{n_starters}_{n_particles}.json"
    if os.path.isfile(filename):
        tqdm.write("reading from file...")
        d = DLA2D.load(filename)
    else:
        tqdm.write(f"Creating fractal {filename} from scratch")
        d = DLA2D(n_starters)
        d.make_fractal(n_particles)
        d.save(filename)
    return d
if __name__ == "__main__":
    for n_starters in range(2, 5):
        d = create_fractal(n_starters, 50000)
        # d.plot_positions()
