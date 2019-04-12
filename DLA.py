import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm as tqdm
import random

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

def iterate_particle(tree,
                     NT = 100000,
                     R = 0.3,
                     Rmin = 1.2,
                     Rmax = 5
                     ):
    while True:
        particle = displacement(loc=np.linalg.norm(tree.data, axis=1).max()*1.05, scale=1/2)
        if not tree.query_ball_point(particle, Rmin):
            # make sure we're spawning far away from existing particles
            break

    for i in range(NT):
        displaced_particle = particle + displacement()
        distance, neighbor_index = tree.query(displaced_particle, 1)

        if distance < R:  # we have found a neighbour!
            return displaced_particle, neighbor_index
        elif distance > Rmax:  # Particle going too far!
            return None
        else:  # run next iteration
            particle = displaced_particle
    else:
        # Could not find neighbor in allotted time.
        return None

def generate_particle_sequential(tree):
    while True:
        appended_particle = iterate_particle(tree)
        if appended_particle is not None:
            return appended_particle

def generate_particle_parallel(tree, pool):
    # To koniecznie trzeba zrównoleglić...
    # można puścić 4 cząstki naraz i jeśli któraś nie jest None, wybrać losową z nie-None
    # parallel_iterate = lambda *args, **kwargs: iterate_particle(tree)
    # trial_particles = p.map(parallel_iterate, range(4))
    # if len(trial_partic
    func = lambda *args, **kwargs: iterate_particle(tree)
    while True:
        particles = list(filter(None, pool.map(func, range(pool._processes))))
        if len(particles):
            return random.choice(particles)

def make_fractal(N_particles = N_particles, N_procs=None):
    particles = [np.zeros(DIM)]
    tree = cKDTree(particles)
    connections = []
    if N_procs is not None:
        p = Pool(N_procs)

    for N in tqdm(range(N_particles)):
        if N_procs is not None:
            particle, neighbor_index = generate_particle_parallel(tree, p)
        else:
            particle, neighbor_index = generate_particle_sequential(tree)
        particles.append(particle)
        tree = cKDTree(particles)
        connections.append((N+1, neighbor_index))

    if N_procs is not None:
        p.close()

    particles = np.vstack(particles)
    return particles, connections

def plot_positions(particles, connections):
    lines = [(particles[par1], particles[par2]) for par1, par2 in connections]
    if DIM == 2:
        import matplotlib.pyplot as plt
        x, y = particles.T
        # plt.scatter(x, y)
        for (x1, y1), (x2, y2) in lines:
            plt.plot([x1, x2], [y1, y2]) 
        plt.grid()
        plt.show()
    elif DIM == 3:
        from mayavi import mlab
        x, y, z = particles.T
        for (x1, y1, z1), (x2, y2, z2) in lines:
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2]) 
        mlab.show()

def plot_mass_distribution(particles):
    import matplotlib.pyplot as plt
    tree = cKDTree(particles)
    distances = np.linalg.norm(particles, axis=1)
    Rs = np.linspace(0, distances.max(), 100)
    Ns = [len(tree.query_ball_point(np.zeros(DIM), R)) for R in Rs]
    plt.plot(Rs, Ns)
    plt.show()


if __name__ == "__main__":
    particles, connections = make_fractal(N_particles, N_procs = 4)
    plot_positions(particles, connections)
    # plot_mass_distribution(particles)
