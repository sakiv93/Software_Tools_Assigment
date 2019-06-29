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
NUMBER_OF_TIME_STEPS = 10000
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

def potential_at_mesh(x,y,M,P):
	V=-GRAVITATIONAL_CONSTANT*M/np.linalg.norm([x-P[0],y-P[1]],axis=0)
	return V

#Co-ordinates of plot
x_start=-1.5
x_end=1.5
y_start=-1.5
y_end=1.5
# Number of little rectangles
no_pixels_x=10
no_pixels_y=10
#pixel size
dx=(x_end-x_start)/no_pixels_x
dy=(y_end-y_start)/no_pixels_y
#co-ordinates of vertices.
xv=np.linspace(x_start,x_end,no_pixels_x+1)
yv=np.linspace(y_start,y_end,no_pixels_y+1)
#center points of pixels
xc=np.linspace(x_start+(dx/2),x_end-(dx/2),no_pixels_x)
yc=np.linspace(y_start+(dy/2),y_end-(dy/2),no_pixels_y)
#create 2D co-ordinate array
xv_2d,yv_2d=np.meshgrid(xv,yv)
xc_2d,yc_2d=np.meshgrid(xc,yc)

#potentials=potential(0.,0.,MASSES[0],POSITIONS[0])

positions_trace=[POSITIONS]
for step in tqdm(range(NUMBER_OF_TIME_STEPS+1)):
	#plotting every single configuration does not make sense
	if step % PLOT_INTERVAL == 0:
		#Plotting pcolor mesh
		v_sum=np.zeros_like(xc_2d)
		for i in range(number_of_planets):
			v=potential_at_mesh(xc_2d,yc_2d,MASSES[i],POSITIONS[i])
			#v_sum is used to add potentials due to each planet
			v_sum=v_sum+v
		fig, ax = plt.subplots()
		ax.pcolormesh(xv_2d,yv_2d,v_sum)
		ax.set(xlabel='Position x (m)',ylabel='Position y (m)')
		im=plt.imshow(v_sum,origin='lower',extent=(x_start,x_end,y_start,y_end),aspect='equal',vmin=-0.1e9,cmap='RdBu')
		##plt.colorbar();
		plt.colorbar()
		output_file_path = Path("Potentials", "{:016d}.png".format(step))
		output_file_path.parent.mkdir(exist_ok=True)
		fig.savefig(output_file_path)
		plt.close(fig)


	distance_vector_3d=POSITIONS[:,np.newaxis,:] - POSITIONS[np.newaxis,:,:]
	distance_vector=np.sum(distance_vector_3d,axis=0)
	distance_vector_length =np.linalg.norm(distance_vector,axis=-1)
	acceleration=(GRAVITATIONAL_CONSTANT* (MASSES/ distance_vector_length ** 2)* distance_vector)
	POSITIONS =np.add(POSITIONS,(TIME_STEP*VELOCITIES))
	positions_trace.append(POSITIONS)
	VELOCITIES =np.add(VELOCITIES,(TIME_STEP*acceleration))

