# DLA

![Sample cluster](images/2d_seed_36_100000_particles.png)

A Pythonic, optimized, multiply-parallelized simulation of diffusion-limited aggregation on continuous space.

## The sequential algorithm

* The algorithm starts with a cluster consisting of a single particle in the middle. 
    * The particle has an effective interaction radius of R - a parameter, 1/20 by default. 
* Another, random one is sampled on a circle around the existing particle. 
    * By default, the circle's radius is defined by TODO
* The sampled particle a random walk whose steps have random angles and radii dependent on the distance from the nearest particle.
    * If the particle reaches a distance of 100 R from * TODO absorbing boundary condition on big radius
* Once the particle reaches the existing cluster, it sticks to it, remaining attached and static for the rest of the simulation.
    * "Reaching the existing cluster" is defined by the particle's distance to its closest particle in the cluster falling below 2R. The particles can be thought of as "soft".
* Every particle's moment of attachment triggers a rebuild of `scipy.spatial.cKDTree` which lets us find distances rapidly.
* Another particle is sampled from a larger circle and the cycle repeats itself.

## Parallelizations

The simulation is parallized in two ways.

Firstly and simply, `multiprocessing.pool` is used so as to build multiple fractals at the same time and get better statistics via ensemble averaging.

Secondly, after an initial sequential iteration builds a "core" cluster, multiple random walkers are simulated at the same time. This is currently hard-coded in `iterate_multiple`.
