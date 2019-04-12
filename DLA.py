import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm as tqdm
import random
from multiprocessing import Pool

np.random.seed(0)
DIM = 2
N_particles = 5000

def displacement(loc = 3, scale=1/10):
    distance = np.abs(np.random.normal(loc=loc, scale=scale))
    theta= np.random.random()*2*np.pi
    if DIM == 3:
        phi = np.random.random() * np.pi    
        x = np.cos(theta) * np.sin(phi) * distance
        y = np.sin(theta) * np.sin(phi) * distance
        z = np.cos(phi) * distance
        return np.array([x, y, z])
    elif DIM == 2:
        x = np.cos(theta) * distance
        y = np.sin(theta) * distance
        return np.array([x, y])
    else:
        raise ValueError(f"Invalid dimension {DIM}!")

class DLA:
    def __init__(self, DIM):
        self.particles = [np.zeros(DIM)]
        self.tree = cKDTree(self.particles)
        self.connections = []
        self.DIM = DIM

    def generate_particle(self):
        while True:
            appended_particle = self.iterate_particle()
            if appended_particle is not None:
                return appended_particle

    def iterate_particle(self,
                         NT = 100000,
                         R = 0.3,
                         Rmin = 1.2,
                         Rmax = 5
                         ):
        while True:
            particle = displacement(loc=np.linalg.norm(self.particles, axis=1).max()*1.05, scale=1/2)
            if not self.tree.query_ball_point(particle, Rmin):
                # make sure we're spawning far away from existing particles
                break

        for i in range(NT):
            displaced_particle = particle + displacement()
            distance, neighbor_index = self.tree.query(displaced_particle, 1)

            if distance < R:  # we have found a neighbour!
                return displaced_particle, neighbor_index
            elif distance > Rmax:  # Particle going too far!
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
            self.connections.append((N+1, neighbor_index))

    def plot_mass_distribution(self):
        import matplotlib.pyplot as plt
        distances = np.linalg.norm(self.particles, axis=1)
        Rs = np.linspace(0, distances.max(), 100)
        Ns = [len(self.tree.query_ball_point(np.zeros(DIM), R)) for R in Rs]
        plt.plot(Rs, Ns)
        plt.show()

    def plot_positions(self):
        lines = [(self.particles[par1], self.particles[par2]) for par1, par2 in self.connections]
        if self.DIM == 2:
            import matplotlib.pyplot as plt
            x, y = np.vstack(self.particles).T
            # plt.scatter(x, y)
            for (x1, y1), (x2, y2) in lines:
                plt.plot([x1, x2], [y1, y2]) 
            plt.grid()
            plt.show()
        elif self.DIM == 3:
            from mayavi import mlab
            x, y, z = np.vstack(self.particles).T
            for (x1, y1, z1), (x2, y2, z2) in lines:
                mlab.plot3d([x1, x2], [y1, y2], [z1, z2]) 
            mlab.show()
        else:
            raise NotImplementedError(self.DIM)

class MapDLA(DLA):
    def __init__(self, DIM, N_procs):
        super().__init__(DIM)
        self.N_procs = N_procs

    def generate_particle(self, pool):
        # To koniecznie trzeba zrównoleglić...
        # można puścić 4 cząstki naraz i jeśli któraś nie jest None, wybrać losową z nie-None
        # parallel_iterate = lambda *args, **kwargs: iterate_particle(tree)
        # trial_particles = p.map(parallel_iterate, range(4))
        # if len(trial_partic
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

if __name__ == "__main__":
    d = DLA(2)
    # d = MapDLA(2, 4)
    d.make_fractal()
    d.plot_positions()
