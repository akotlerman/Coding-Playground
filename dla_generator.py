import numpy as np
import random
from PIL import Image


def generate_dla_grid(size):
    """ Generates a 2D grid of booleans defining a DLA.

    DLA (Diffusion Limited Aggregate) are 3-dimensional fractals that occur
    in nature as a result of random processes. This program aims to recreate
    the stochastic nature behind these structures in a easy-to-read python script.

    More information behind the algorithm can be found here:
    http://alumnus.caltech.edu/~jsu/dla/dla.html

    Summary of the algorithm:
        A seed particle is set in the center of a square grid. All other particles
        are generated at the edge of a very small circle. If particles are close
        enough to the seed or they leave the grid, a new particle is generated.
        Any particles that end up close to another particle "stick". When the
        distance of any "stuck" particle comes close to the radius of our circle,
        the circle is expanded.


    This python code uses numpy arrays to help speed up operations and clean up
    code. Numpy arrays have the advantage of providing clean ways of applying
    element-wise matrix addition.
    """

    class C(np.ndarray):
        """Helper class with extra methods to clean code.

        Should be called with a 2D ndarray like this:

        array.view(C)
        """

        _possible_directions = np.array(
            list(map(np.array, [(+1, +0), (+1, +1), (+0, +1), (-1, +1),
                                (-1, +0), (-1, -1), (+0, -1), (+1, -1)])))
        """ 8 possible directions particles can travel on a 2D grid.

        Todo: Distance of (1,1) could be
        normalized to (1./sqrt(2), 1./sqrt(2))
        """

        _radius = 10
        """ Radius of inner circle inscribed in the grid. """

        @property
        def possible_directions(self):
            """ Access protected variable _possible_directions"""
            return self._possible_directions

        @property
        def radius(self):
            """ Access protected variable _radius """
            return self._radius

        def center(self):
            """ Returns the center of the 2D grid. """
            return (np.array(self.shape) / 2).round().astype(int)

        def set_radius(self, radius):
            """ Set inner radius of a circle inscribed within the grid.

            Radius referenced here is used in function "out_of_bounds"
            """
            self._radius = radius

        def max_radius(self):
            """ Returns max radius as defined by the size of the grid. """
            return min(*self.shape)/2 - 2

        # Create a copy of the grid with padded zeros
        # in the indicated directions
        def pad_matrix(self, amount):
            """ Expand inner radius of a circle inscribed within a grid.

            Radius referenced is used in function "out_of_bounds"
            Parameter amount corresponds to integer values.
            """
            self.set_radius(self._radius + amount)

        def has_neighbors(self, coordinate):
            """ Checks for any neighbors around a 2D coordinate. """
            return any(self[int(c[0]), int(c[1])]
                       for c in (self._possible_directions + coordinate))

        def out_of_bounds(self, position):
            """ Return boolean if point is outside circle defined by self.radius """
            return ((position - self.center())**2).sum()**0.5 > self._radius

    # Make sure the grid is created with the custom class view
    grid = np.zeros((size, size), dtype=np.bool).view(C)

    # Set the seed
    grid[grid.center()[0], grid.center()[1]] = True

    while grid.radius < grid.max_radius():
        # Generate a particle in a random position on our entry circle
        entry = random.random() * 2 * np.pi
        particle_position = grid.center() + [round(np.cos(entry) * grid.radius),
                                             round(np.sin(entry) * grid.radius)]
        # Random walk
        for _ in range(200):
            # Go one of 8 directions
            particle_position += grid.possible_directions[random.randint(0, 7)]

            if particle_position[0] < 0 or particle_position[0] >= grid.shape[0]-1 or \
               particle_position[1] < 0 or particle_position[1] >= grid.shape[1]-1:
                # Break out of the loop if the particle hits the edge of the grid
                break

            # Look at neighbors and see if there is
            # another particle to stick to
            if grid.has_neighbors(particle_position):
                grid[int(particle_position[0]), int(particle_position[1])] = True

                # If placed on the edge of the circle, increase grid size to new radius
                # This important or the DLA doesn't come out right
                if grid.out_of_bounds(particle_position):
                    grid.set_radius(((particle_position[0] - grid.center()[0])**2.0 +
                                     (particle_position[1] - grid.center()[1])**2.0)**0.5)
                    # print("Increased size to: {}".format(grid.radius))
                break

    return grid


def generate_dla_image(size):
    """ Generate a PIL image of size "size x size" of a DLA

    New PIL image with mode L http://effbot.org/imagingbook/concepts.htm#mode

    """

    dla = generate_dla_grid(size)
    image = Image.new("L", tuple(dla.shape))

    for i in range(dla.shape[0]):
        for j in range(dla.shape[1]):
            if dla[i, j]:
                image.putpixel((int(i), int(j)), 600)

    return image

generate_dla_image(100).show()
