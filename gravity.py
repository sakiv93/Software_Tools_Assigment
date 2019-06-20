from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

# tqdm is used for a progress bar
from tqdm import tqdm

# parameters
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2
POSITIONS = np.array([[-1, 0], [1, 0]])
VELOCITIES = np.array([[0, -1], [0, 1]])
MASSES = np.array([4 / GRAVITATIONAL_CONSTANT, 4 / GRAVITATIONAL_CONSTANT])
TIME_STEP = 0.0001  # s
NUMBER_OF_TIME_STEPS = 2
PLOT_INTERVAL = 2

# derived variables
number_of_planets = len(POSITIONS)
number_of_dimensions = 2

# make sure the number of planets is the same for all quantities
assert len(POSITIONS) == len(VELOCITIES) == len(MASSES)
for position in POSITIONS:
    assert len(position) == number_of_dimensions
for velocity in VELOCITIES:
    assert len(velocity) == number_of_dimensions


for step in tqdm(range(NUMBER_OF_TIME_STEPS + 1)):
    #print(POSITIONS)
    # plotting every single configuration does not make sense
    # if step % PLOT_INTERVAL == 0:
    #     fig, ax = plt.subplots()
    #     x = []
    #     y = []
    #     for position in POSITIONS:
    #         x.append(position[0])
    #         y.append(position[1])
    #     ax.scatter(x, y)
    #     ax.set_aspect("equal")
    #     ax.set_xlim(-1.5, 1.5)
    #     ax.set_ylim(-1.5, 1.5)
    #     ax.set_title("t = {:8.4f} s".format(step * TIME_STEP))
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     output_file_path = Path("positions", "{:016d}.png".format(step))
    #     output_file_path.parent.mkdir(exist_ok=True)
    #     fig.savefig(output_file_path)
    #     plt.close(fig)

    # the accelerations for each planet are required to update the velocities
    #$$accelerations = []
    accelerations = np.array([])
    #for i in range(number_of_planets):
        #$$acceleration = [0, 0]
    acceleration = np.zeros(2)
        #for j in range(number_of_planets):
            #if i == j:
               # continue

            #$$distance_vector = []
    distance_vector = np.array([])
            #for coordinate_i, coordinate_j in zip(POSITIONS[i], POSITIONS[j]):
                #distance_vector=np.append(distance_vector,(coordinate_j - coordinate_i))
    distance_vector=POSITIONS[1]-POSITIONS[0]
    #print(distance_vector)

    #sum_of_squares = 0
    #for coordinate in distance_vector:
        #sum_of_squares += coordinate ** 2
    distance_vector_length = np.linalg.norm(distance_vector)

            #$$acceleration_contribution = []
    acceleration_contribution = np.array([])
    #for coordinate in distance_vector:
                # acceleration_contribution.append(
                #     GRAVITATIONAL_CONSTANT
                #     * MASSES[j]
                #     / distance_vector_length ** 2
                #     * coordinate
                # )
    acceleration_contribution=(GRAVITATIONAL_CONSTANT* MASSES/ distance_vector_length ** 2* distance_vector)
            #$$for i in range(number_of_dimensions):
                #$$acceleration[i] += acceleration_contribution[i]
    acceleration=acceleration_contribution

        #$$accelerations.append(acceleration)
    accelerations=np.append(accelerations,acceleration)
    print(accelerations)
    # position=np.array([])
    # velocity=np.array([])
    # for position, velocity in zip(POSITIONS, VELOCITIES):
    #     for i in range(number_of_dimensions):
    #         position[i] += TIME_STEP * velocity[i]

    # for velocity, acceleration in zip(VELOCITIES, accelerations):
    #     for i in range(number_of_dimensions):
    #         velocity[i] += TIME_STEP * acceleration[i]
    
