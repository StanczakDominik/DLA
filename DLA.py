import numpy as np
from scipy.spatial import cKDTree
import tqdm
import json
import os
import numba
import math
import random
from textwrap import dedent
import matplotlib.pyplot as plt
import inspect
import pandas


DTYPE = np.float32

@numba.njit
def radius(particle):
    x, y = particle
    return math.sqrt(x**2 + y**2)

@numba.njit
def position_on_circle(radius):
    theta = random.random() * 2 * np.pi
    x = math.cos(theta) * radius
    y = math.sin(theta) * radius
    particle = np.array([x, y], dtype=DTYPE)
    return particle

class DLA2D:
    DIM = 2
    @classmethod
    def displacement(cls, loc = 1):
        return np.random.normal(scale=loc, size=cls.DIM).astype(DTYPE)

    @classmethod
    def displacement_multiple(cls, N, loc = 1):
        return np.random.normal(scale=loc.reshape(N, 1), size=(N, cls.DIM)).astype(DTYPE)

    def position_on_circle(self, N, radius, off_center = True):
        theta = np.random.random(N) * 2 * np.pi
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        new_samples = np.vstack([x, y]).T
        if off_center:
            new_samples += np.mean(self.particles, axis=0)
        return new_samples

    def __init__(self,
                 num_starters = 1,
                 R = 1/20,
                 ):
        self.num_starters = num_starters
        self.R = R

        if num_starters == 1:
            self.particles = [np.zeros(self.DIM)]
            self.max_distance = 0
        else:
            self.particles = list(np.random.normal(scale = num_starters * R * 5,
                                                   size=(num_starters, self.DIM)))
            self.max_distance = np.linalg.norm(self.particles, axis=1).max()
        self.tree = cKDTree(self.particles)

    @property
    def N_particles(self):
        return len(self.particles)

    @property
    def description(self):
        desc = f"""DLA with {self.N_particles}, starting from {self.num_starters} seeds
                   Particle interaction distance: {self.R:.2e} 
                   """
        return "\n".join([x.strip() for x in desc.splitlines()])

    def iterate_particle(self):
        spawn_distance = self.max_distance + 5 
        particle = position_on_circle(spawn_distance)


        while True:
            distance, _ = self.tree.query(particle, 1)
            if distance < (self.R * 3):
                step_size = self. R / 2
            else:
                step_size = distance / 2

            displaced_particle = particle + self.displacement(step_size)
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

    def make_fractal(self, N_particles):
        for N in tqdm.tqdm(range(N_particles), unit="particles"):
            particle, neighbor_index = self.iterate_particle()
            self.particles.append(particle)
            self.tree = cKDTree(self.particles)

    def iterate_multiple(self, N_particles, bunch_size):
        self.max_distance = np.max(np.linalg.norm(self.particles, axis=1))
        spawn_distance = self.max_distance + 5 
        particles = self.position_on_circle(bunch_size, spawn_distance)
        num_steps = np.zeros(bunch_size, dtype=int)
        full_num_steps = []
        indices = np.arange(bunch_size)

        if isinstance(self.particles, np.ndarray):
            self.particles = self.particles.tolist()
        added_particles = 0
        progressbar = tqdm.tqdm(total=N_particles, initial=self.N_particles, unit="particles")
        with progressbar:
            while self.N_particles < N_particles:
                distances, _ = self.tree.query(particles, 1)
                step_size = distances / 2
                step_size[distances < (self.R * 3)] = self.R / 2
                progressbar.set_postfix(**{"r0": distances[0], "n0": num_steps[0], "step": step_size[0]})
                particles += self.displacement_multiple(bunch_size, step_size)
                num_steps += 1
                particle_radius = np.linalg.norm(particles, axis=1)
                neighbors = self.tree.query_ball_point(particles, self.R)

                have_neighbors = neighbors.astype(bool)
                if have_neighbors.any():
                    number_added = have_neighbors.sum()
                    self.particles.extend(particles[have_neighbors].tolist())
                    rmax = particle_radius[have_neighbors].max()
                    if self.max_distance < rmax:
                        self.max_distance = rmax
                        spawn_distance = self.max_distance + 5

                    particles[have_neighbors] = self.position_on_circle(number_added, spawn_distance)
                    full_num_steps.extend(num_steps[have_neighbors].tolist())
                    num_steps[have_neighbors] = 0
                    added_particles += number_added

                    progressbar.update(number_added)
                    self.tree = cKDTree(self.particles)

                out_of_bounds = (spawn_distance + 5) < particle_radius
                particles[out_of_bounds] = self.position_on_circle(out_of_bounds.sum(), spawn_distance)

        self.particles = np.array(self.particles[:N_particles])
        self.tree = cKDTree(self.particles)
        return full_num_steps
        

    def make_in_steps(self, final_N_particles, bunchexponent = 0.75):
        while self.N_particles < final_N_particles:
            try:
                dN = final_N_particles - self.N_particles
                bunch_size = int(self.N_particles**0.75)
                num_particles_for_iteration = 100 * bunch_size if (100 * bunch_size) < dN else dN
                tqdm.tqdm.write(f"Currently at {self.N_particles}, going up to {num_particles_for_iteration + dN} with bunch size {bunch_size}")
                self.iterate_multiple(num_particles_for_iteration + self.N_particles, bunch_size)
            except KeyboardInterrupt:
                break


    def get_off_center_distances(self, center = None):
        if center is None:
            center = np.mean(self.particles, axis=0)
        recentered_particles = self.particles - center
        distances = np.linalg.norm(recentered_particles, axis=1)
        return distances

    def plot_mass_distribution(self,
                               filename = None,
                               ax = None,
                               plot = True,
                               minimum = None,
                               maximum=None,
                               min_distance = None,
                               max_distance = None,
                               ):
        center = np.mean(self.particles, axis=0)
        distances = self.get_off_center_distances(center)
        recentered_tree = cKDTree(self.particles - center)

        if minimum is None:
            minimum = np.quantile(distances, 0.01)
        if maximum is None:
            maximum = np.quantile(distances, 0.6)

        if min_distance is None:
            min_distance = distances.min()

        if max_distance is None:
            max_distance = distances.max()

        Rs = np.logspace(np.log10(min_distance), np.log10(max_distance), 10000)
        Ns = np.array([len(recentered_tree.query_ball_point(center, R)) for R in Rs])

        indices = (minimum < Rs) & (Rs < maximum)
        (a_fit, b_fit), cov = np.polyfit(np.log10(Rs[indices]), np.log10(Ns[indices]), 1, cov=True)
        std_a = cov[0,0]**0.5

        if plot:
            title = fr"Estimated dimension: ${a_fit:.2f} \pm {std_a:.2f}$"
            if ax is None:
                fig, ax = plt.subplots()
            ax.set_title(title)
            ax.loglog(Rs, Ns)
            ax.loglog(Rs[indices], Ns[indices], "r", label=title)
            ax.axvline(minimum, color='r', linestyle="--")
            ax.axvline(maximum, color='r', linestyle="--")
            ax.legend(loc='best')
            ax.set_xlabel("Radius R")
            ax.set_ylabel("Number of particles within R")
            if filename is not None:
                plt.savefig(filename)
                plt.close()
            else:
                plt.show()
        return Rs, Ns

    def plot_particles(self, filename = None, ax = None):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=(16, 12),
            )
        x, y = np.vstack(self.particles).T
        r = range(len(y))
        plt.style.use('dark_background')
        points = ax.scatter(x, y, c = r, marker='.', s=1, cmap='Blues', alpha=0.6) 
        cbar = plt.colorbar(points)
        cbar.ax.set_ylabel('Time of arrival', rotation=270)
        ax.plot(x[:self.num_starters], y[:self.num_starters], "*", markersize=20)
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        ax.set_title(self.description)
        if filename is not None:
            plt.savefig(filename, dpi=400)
            plt.close()
        else:
            plt.show()

    def save(self, filename = None):
        if filename is None:
            filename = f"2d_{self.num_starters}_{len(self.particles)}.json"

        with open(filename, "w") as f:
            d = dict(particles=np.vstack(self.particles).tolist(),
                     num_starters = self.num_starters,
                     R=self.R,
                     code=inspect.getsource(inspect.getmodule(inspect.currentframe())),
                     )
            json.dump(d, f)
        return filename

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            d = json.load(f)
        dla = cls(d['num_starters'], d['R'])
        dla.particles = np.array(d['particles'])
        dla.tree = cKDTree(dla.particles)
        return dla

    @classmethod
    def create_fractal(cls, n_starters = 2, n_particles = 5000, force_new = False,
                       filename = None, *args, **kwargs):
        if filename is None:
            filename = f"2d_{n_starters}_{n_particles}.json"
        if not force_new and os.path.isfile(filename):
            tqdm.tqdm.write(f"reading from file {filename}...")
            d = cls.load(filename)
        else:
            tqdm.tqdm.write(f"Creating fractal {filename} from scratch")
            d = cls(n_starters, *args, **kwargs)
            d.make_fractal(n_particles)
            d.save(filename)
        return d

def plot_all(directory = "."):
    import glob
    for filename in glob.glob(os.path.join(directory, "*.json")):
        d = DLA2D.load(filename)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
        d.plot_particles(filename.replace(".json", "_particles.png"))
        d.plot_mass_distribution(filename.replace(".json", "_distribution.png"))
        # fig.savefig(dpi=600, fname=filename.replace("json", "png"))

def plot_all_dimensions(directory = "."):
    import glob
    fractals = []

    for filename in tqdm.tqdm(glob.glob(os.path.join(directory, "*.json"))):
        d = DLA2D.load(filename)
        if d.N_particles >= 1e5:
            fractals.append(d)

    global_min_distance = np.inf
    global_max_distance = -np.inf
    for fractal in tqdm.tqdm(fractals):
        distances = fractal.get_off_center_distances()
        min_distance = distances.min()
        if min_distance < global_min_distance:
            global_min_distance = min_distance
        max_distance = distances.max()
        if global_max_distance < max_distance:
            global_max_distance = max_distance

    dataframes = []
    for fractal in tqdm.tqdm(fractals):
        Rs, Ns = fractal.plot_mass_distribution(min_distance=global_min_distance,
                                       max_distance=global_max_distance,
                                       )
        dataframes.append(pandas.DataFrame({"R": Rs, "N": Ns}))
    return dataframes




def main(plot = False, initsize=int(5e3), gotosize=[int(1e4)], bunchexponent=0.5):
    with tqdm.trange(0, 3) as progressbar:
        for seed in progressbar:
            np.random.seed(seed)
            progressbar.set_postfix(seed=seed)
            d = DLA2D.create_fractal(1,
                                     int(initsize),
                                     R = 1/2,
                                     force_new = False
                                     )
            for uptosize in gotosize:
                d.make_in_steps(int(uptosize), bunchexponent)
                filename = f"2d_seed_{seed}_{len(d.particles)}.json"
                filename = d.save(filename)
                if plot:
                    d.plot_particles(filename.replace(".json", "_particles.png"))
                    d.plot_mass_distribution(filename.replace(".json", "_distribution.png"))
    return d

if __name__ == "__main__":
    main(True, initsize=5e4, gotosize = [5e5])
    dimensions = plot_all_dimensions()
    meandf = sum(dimensions) / len(dimensions)
    off_center_dimensions = [(df - meandf)**2 for df in dimensions]
    stddf = np.sqrt(sum(off_center_dimensions)) / len(dimensions)
    # meandf.plot('R', 'N', logy=True, logx=True)
    # plt.show()

    x= (0.5, 80)
    print(x)
    xmin, xmax = x
    indices = (xmin < meandf['R']) & (meandf['R'] < xmax)
    middle = meandf[indices]
    middlestd = stddf[indices]
    (a, b), cov = np.polyfit(np.log10(middle['R']),
                             np.log10(middle['N']),
                             1,
                             w = 1/np.log10(middlestd['N']),
                             cov=True)
    stda = cov[0,0]**0.5
    meandf['Nfit'] = 10**np.polyval((a, b), np.log10(meandf['R']))
    fig, ax = plt.subplots()
    meandf.plot('R', 'N', logx=True, logy=True, ax=ax)
    meandf.plot('R', 'Nfit', logx=True, logy=True, ax=ax)
    ax.set_title(fr"y = {a:.4f} $\pm$ {stda:.3f}")
    ax.axvline(xmin)
    ax.axvline(xmax)
    for seed, df in enumerate(dimensions):
        df.plot('R', 'N', logx=True, logy=True, ax=ax, alpha=0.1, label=f"Seed: {seed}")
    plt.show()


