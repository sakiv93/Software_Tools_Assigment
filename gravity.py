from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import scipy

# tqdm is used for a progress bar
from tqdm import tqdm

# parameters
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2
POSITIONS = np.array([[-1, 0], [1, 0]])
VELOCITIES = np.array([[0, -1], [0, 1]])
MASSES = np.array([[4 / GRAVITATIONAL_CONSTANT],[4 / GRAVITATIONAL_CONSTANT]])
TIME_STEP = 0.0001  # s
NUMBER_OF_TIME_STEPS = 1
PLOT_INTERVAL = 1000

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
for step in tqdm(range(NUMBER_OF_TIME_STEPS+1)):
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

	distance_vector=POSITIONS[:,np.newaxis,:] - POSITIONS[np.newaxis,:,:]
	distance_vector_length =np.linalg.norm(distance_vector,axis=-1)
	print(distance_vector_length)
	# acceleration=(GRAVITATIONAL_CONSTANT* (MASSES/ distance_vector_length ** 2)* distance_vector)
	# print(acceleration)
	# POSITIONS =np.add(POSITIONS,(TIME_STEP*VELOCITIES))
	# VELOCITIES =np.add(VELOCITIES,(TIME_STEP*acceleration))
	
	
	
	
	
	# print('velocities:',VELOCITIES)
	# ##For plotting purpose##
	# position_trace=np.reshape(POSITIONS,(1,np.product(POSITIONS.shape)))
	# #print('position_trace:',position_trace)
	# position_traces=np.append(position_traces,position_trace,axis=0)
	# print('position_traces:',position_traces)





# fig,ax=plt.subplots()
# ax.plot(position_traces[:,0],position_traces[:,1],position_traces[:,2],position_traces[:,3])
# plt.show()

# Things to look for
# np.newaxis
# scipy.spacial.pdist












#print(position_traces)

# plot_traces=position_traces.transpose()
# print(plot_traces)

# for i in range(0,(2*len(POSITIONS)),2):
# 	fig,ax=plt.subplots()
# 	ax.plot(position_traces[:,i],position_traces[:,i+1])
# 	plt.show()

