from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

# tqdm is used for a progress bar
from tqdm import tqdm

# parameters
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2
POSITIONS = np.array([[-1, 0], [1, 0]])
VELOCITIES = np.array([[0, -1], [0, 1]])
MASSES = np.array([[4 / GRAVITATIONAL_CONSTANT],[4 / GRAVITATIONAL_CONSTANT]])
TIME_STEP = 0.0001  # s
NUMBER_OF_TIME_STEPS = 100000
PLOT_INTERVAL = 1

# derived variables
number_of_planets = len(POSITIONS)
number_of_dimensions = 2

# make sure the number of planets is the same for all quantities
assert len(POSITIONS) == len(VELOCITIES) == len(MASSES)
for position in POSITIONS:
	assert len(position) == number_of_dimensions
for velocity in VELOCITIES:
	assert len(velocity) == number_of_dimensions


position_traces=np.reshape(POSITIONS,(1,np.product(POSITIONS.shape)))
#print(position_traces)
for step in tqdm(range(NUMBER_OF_TIME_STEPS)):
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
	accelerations=np.array([[0,0]])
	for i in range(number_of_planets):
		distance_vector=np.sum(POSITIONS-POSITIONS[i],axis=0)
		# print('Distance_vector:',distance_vector)
		distance_vector_length = np.array([[np.linalg.norm(distance_vector)]])
		# print('Distance_vector_Length:',distance_vector_length)
		acceleration=(GRAVITATIONAL_CONSTANT* (MASSES[i]/ distance_vector_length ** 2)* distance_vector)
		# print('Acceleration:',acceleration)
		accelerations=np.append(accelerations,acceleration,axis=0)
		# print('Accelerations:',accelerations)


	##Evaluated Velocity before calculating positions which is semi implicit scheme.##
	POSITIONS =np.add(POSITIONS,(TIME_STEP*VELOCITIES))
	#print('positions:',POSITIONS)
	VELOCITIES =np.add(VELOCITIES,(TIME_STEP*accelerations[1:]))
	# print('velocities:',VELOCITIES)
	##For plotting purpose##
	position_trace=np.reshape(POSITIONS,(1,np.product(POSITIONS.shape)))
	#print('position_trace:',position_trace)
	position_traces=np.append(position_traces,position_trace,axis=0)
	#print('position_traces:',position_traces)


#print(position_traces)

# plot_traces=position_traces.transpose()
# print(plot_traces)

# for i in range(0,(2*len(POSITIONS)),2):
# 	fig,ax=plt.subplots()
# 	ax.plot(position_traces[:,i],position_traces[:,i+1])
# 	plt.show()


fig,ax=plt.subplots()
ax.plot(position_traces[:,0],position_traces[:,1],position_traces[:,2],position_traces[:,3])
plt.show()

