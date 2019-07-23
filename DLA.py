import numpy as np
from scipy.spatial import cKDTree
import tqdm
import json
import os
import numba
import math
import random
from multiprocessing import Pool

DTYPE = np.float32


@numba.njit
def radius(particle):
    x, y = particle
    return math.sqrt(x ** 2 + y ** 2)


@numba.njit
def position_on_circle(radius):
    theta = random.random() * 2 * np.pi
    x = math.cos(theta) * radius
    y = math.sin(theta) * radius
    particle = np.array([x, y], dtype=DTYPE)
    return particle

@numba.vectorize
def calculate_step_size(distance, R):
    if distance < (R * 3):
        step_size = R * 0.75
    elif distance < (R * 6):
        step_size = R * 1.2
    else:
        step_size = distance * 0.75
    return step_size

class DLA2D:
    DIM = 2

    @classmethod
    def displacement(cls, loc=1):
        theta = random.random() * 2 * np.pi
        x = math.cos(theta) * loc
        y = math.sin(theta) * loc
        particle = np.array([x, y], dtype=DTYPE)
        return particle

    @classmethod
    def displacement_multiple(cls, N, loc=1):
        theta = np.random.random(N) * 2 * np.pi
        x = np.cos(theta) * loc
        y = np.sin(theta) * loc
        particle = np.vstack([x, y]).T
        return particle

    def position_on_circle(self, N, radius, off_center=True):
        theta = np.random.random(N) * 2 * np.pi
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        new_samples = np.asarray(np.vstack([x, y]).T, order="C")
        if off_center:
            new_samples += np.mean(self.particles, axis=0)
        return new_samples

    def __init__(self, num_starters=1, R=1 / 20):
        self.num_starters = num_starters
        self.R = R

        if num_starters == 1:
            self.particles = [np.zeros(self.DIM)]
            self.max_distance = 0
        else:
            self.particles = list(
                np.random.normal(
                    scale=num_starters * R * 5, size=(num_starters, self.DIM)
                )
            )
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
        spawn_distance = self.max_distance + 100 * self.R
        particle = position_on_circle(spawn_distance)

        while True:
            distance, _ = self.tree.query(particle, 1)
            step_size = calculate_step_size(distance, self.R)

            displaced_particle = particle + self.displacement(step_size)
            particle_radius = radius(displaced_particle)
            neighbors = self.tree.query_ball_point(displaced_particle, self.R)

            if neighbors:
                if self.max_distance < particle_radius:
                    self.max_distance = particle_radius
                neighbor_index = neighbors[0]
                return displaced_particle, neighbor_index

            elif (spawn_distance + 100 * self.R) < particle_radius:
                particle = position_on_circle(spawn_distance)

            else:  # run next iteration
                particle = displaced_particle

    def iterate_multiple(self, N_particles, bunch_size, progressbar=None):
        self.max_distance = np.max(np.linalg.norm(self.particles, axis=1))
        spawn_distance = self.max_distance + 100 * self.R
        particles = self.position_on_circle(bunch_size, spawn_distance)
        num_steps = np.zeros(bunch_size, dtype=int)
        full_num_steps = []
        indices = np.arange(bunch_size)

        if isinstance(self.particles, np.ndarray):
            self.particles = self.particles.tolist()
        added_particles = 0
        progressbar_needs_closing = False

        if progressbar is None:
            progressbar = tqdm.tqdm(
                total=N_particles, initial=self.N_particles, unit="particles"
            )
            progressbar_needs_closing = True

        while self.N_particles < N_particles:
            distances, _ = self.tree.query(particles, 1)
            step_size = calculate_step_size(distances, self.R)

            progressbar.set_postfix(
                **{"r0": distances[0], "n0": num_steps[0], "step": step_size[0]}
            )
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
                    spawn_distance = self.max_distance + 100 * self.R

                particles[have_neighbors] = self.position_on_circle(
                    number_added, spawn_distance
                )
                full_num_steps.extend(num_steps[have_neighbors].tolist())
                num_steps[have_neighbors] = 0
                added_particles += number_added

                progressbar.update(number_added)
                self.tree = cKDTree(self.particles)

            out_of_bounds = (spawn_distance + 5) < particle_radius
            particles[out_of_bounds] = self.position_on_circle(
                out_of_bounds.sum(), spawn_distance
            )

        if progressbar_needs_closing:
            progressbar.close()

        self.particles = np.array(self.particles[:N_particles])
        self.tree = cKDTree(self.particles)
        return full_num_steps

    def make_in_steps(self, final_N_particles, bunchexponent=0.5):
        with tqdm.tqdm(
            total=final_N_particles, initial=self.N_particles, unit="particles"
        ) as progressbar:
            while self.N_particles < final_N_particles:
                try:
                    dN = final_N_particles - self.N_particles
                    bunch_size = int(self.N_particles ** bunchexponent)
                    num_particles_for_iteration = (
                        100 * bunch_size if (100 * bunch_size) < dN else dN
                    )
                    tqdm.tqdm.write(
                        f"Currently at {self.N_particles}, going up to {num_particles_for_iteration + dN} with bunch size {bunch_size}"
                    )
                    self.iterate_multiple(
                        num_particles_for_iteration + self.N_particles,
                        bunch_size,
                        progressbar,
                    )
                    self.save()

                except KeyboardInterrupt:
                    break

    def get_off_center_distances(self, center=None):
        if center is None:
            center = np.mean(self.particles, axis=0)
        recentered_particles = self.particles - center
        distances = np.linalg.norm(recentered_particles, axis=1)
        return distances

    def plot_mass_distribution(
        self,
        filename=None,
        ax=None,
        plot=True,
        minimum=None,
        maximum=None,
        min_distance=None,
        max_distance=None,
    ):
        import matplotlib.pyplot as plt

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
        (a_fit, b_fit), cov = np.polyfit(
            np.log10(Rs[indices]), np.log10(Ns[indices]), 1, cov=True
        )
        std_a = cov[0, 0] ** 0.5

        if plot:
            title = fr"Estimated dimension: ${a_fit:.2f} \pm {std_a:.2f}$"
            if ax is None:
                fig, ax = plt.subplots()
            ax.set_title(title)
            ax.loglog(Rs, Ns)
            ax.loglog(Rs[indices], Ns[indices], "r", label=title)
            ax.axvline(minimum, color="r", linestyle="--")
            ax.axvline(maximum, color="r", linestyle="--")
            ax.legend(loc="best")
            ax.set_xlabel("Radius R")
            ax.set_ylabel("Number of particles within R")
            if filename is not None:
                plt.savefig(filename)
                plt.close()
            else:
                plt.show()
        return Rs, Ns

    def plot_particles(self, filename=None, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 12))
        x, y = np.vstack(self.particles).T
        r = range(len(y))
        plt.style.use("dark_background")
        points = ax.scatter(x, y, c=r, marker=".", s=1, cmap="Blues", alpha=0.6)
        cbar = plt.colorbar(points)
        cbar.ax.set_ylabel("Time of arrival", rotation=270)
        ax.plot(x[: self.num_starters], y[: self.num_starters], "*", markersize=20)
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        ax.set_title(self.description)
        if filename is not None:
            plt.savefig(filename, dpi=400)
            plt.close()
        else:
            plt.show()

    def save(self, filename=None):
        if filename is None:
            filename = f"2d_{self.num_starters}_{len(self.particles)}.json"

        with open(filename, "w") as f:
            d = dict(
                particles=np.vstack(self.particles).tolist(),
                num_starters=self.num_starters,
                R=self.R,
            )
            json.dump(d, f)
        return filename

    def make_fractal(self, N_particles):
        for N in tqdm.tqdm(range(N_particles), unit="particles"):
            particle, neighbor_index = self.iterate_particle()
            self.particles.append(particle)
            self.tree = cKDTree(self.particles)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            d = json.load(f)
        dla = cls(d["num_starters"], d["R"])
        dla.particles = np.array(d["particles"])
        dla.tree = cKDTree(dla.particles)
        return dla

    @classmethod
    def create_fractal(
        cls,
        n_starters=2,
        n_particles=5000,
        force_new=False,
        filename=None,
        *args,
        **kwargs,
    ):
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


def main_single(
    seed, plot=False, initsize=int(5e3), gotosize=[1e4, 5e4, 1e5], bunchexponent=0.5
):
    np.random.seed(seed)
    d = DLA2D.create_fractal(1, int(initsize), R=1 / 2, force_new=True)
    for uptosize in gotosize:
        d.make_in_steps(int(uptosize), bunchexponent)
        filename = f"2d_seed_{seed}_{len(d.particles)}.json"
        filename = d.save(filename)
        if plot:
            d.plot_particles(filename.replace(".json", "_particles.png"))
            d.plot_mass_distribution(filename.replace(".json", "_distribution.png"))
    return d


def main(plot=False, initsize=int(5e3), gotosize=[int(1e4)], bunchexponent=0.5):
    with Pool(3) as p:
        r = list(tqdm.tqdm(p.imap(main_single, range(22, 37)), total=15))


if __name__ == "__main__":
    d = main_single(1, gotosize=[1e4, 5e4])
    d.plot_particles()
    d.plot_mass_distribution()
