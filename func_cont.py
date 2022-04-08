#!/usr/bin/env python
"""Calculate the values of surface height (h) and east-west velocity
(u) in a dish of water where a point disturbance of h initiates waves.
Use the simplified shallow water equations on a non-staggered grid.

This is an implementation of lab7 section 4.3.

Example usage from the notebook::

from numlabs.lab7 import rain
# Run 5 time steps on a 9 point grid
rain.rain(5,9)

Example usage from the shell::

  # Run 5 time steps on a 9 point grid
  $ rain.py 5 9

The graph window will close as soon as the animation finishes.  And
the default run for 5 time steps doesn't produce much of interest; try
at least 100 steps.

Example usage from the Python interpreter::

  $ python
  ...
  >>> import rain
  >>> # Run 200 time steps on a 9 point grid
  >>> rain.rain((200, 9))
"""
from __future__ import division
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.colorbar as colorbar
import os,glob
import math

class Quantity(object):
    """Generic quantity to define the data structures and method that
    are used for both u and h.

    u and h objects will be instances of this class.
    """
    def __init__(self, n_grid, n_time):
        """Initialize an object with prev, now, and next arrays of
        n_grid points, and a store array of n_time time steps.
        """
        self.n_grid = n_grid
        # Storage for values at previous, current, and next time step
        self.prev = np.empty(n_grid)
        self.now = np.empty(n_grid)
        self.next = np.empty(n_grid)
        # Storage for results at each time step.  In a bigger model
        # the time step results would be written to disk and read back
        # later for post-processing (such as plotting).
        self.store = np.empty((n_grid, n_time))

    def store_timestep(self, time_step, attr='next'):
        """Copy the values for the specified time step to the storage
        array.

        The `attr` argument is the name of the attribute array (prev,
        now, or next) that we are going to store.  Assigning the value
        'next' to it in the function def statement makes that the
        default, chosen because that is the most common use (in the
        time step loop).
        """
        # The __getattribute__ method let us access the attribute
        # using its name in string form;
        # i.e. x.__getattribute__('foo') is the same as x.foo, but the
        # former lets us change the name of the attribute to operate
        # on at runtime.
        self.store[:, time_step] = self.__getattribute__(attr)


    def shift(self):
        """Copy the .now values to .prev, and the .next values to .new.

        This reduces the storage requirements of the model to 3 n_grid
        long arrays for each quantity, which becomes important as the
        domain size and model complexity increase.  It is possible to
        reduce the storage required to 2 arrays per quantity.
        """
        # Note the use of the copy() method from the copy module in
        # the standard library here to get a copy of the array, not a
        # copy of the reference to it.  This is an important and
        # subtle aspect of the Python data model.
        self.prev = copy.copy(self.now)
        self.now = copy.copy(self.next)


def initial_conditions(c1,c, n_grid):
    """Set the initial condition values.
    """
    c.now[0:n_grid - 1] = 0
    c.now[int(n_grid/3)] = c1
#     c.now[int(n_grid/3)+1] = c1
#     c.now[int(n_grid/3)+2] = c1
#     c.now[int(n_grid/3)+3] = c1
    

def boundary_conditions(c_array, n_grid):
    """Set the boundary condition values.
    """
    c_array[0] = c_array[1]
    c_array[n_grid-1] = c_array[n_grid-2]


def first_time_step(c,c1, n_grid):
    """Calculate the first time step values from the analytical
    predictor-corrector derived from equations 4.18 and 4.19.
    """
    ##maybe if we actually use leapfrog


def scheme(c, u, k, n_grid, dt, dx):
    """Calculate the next time step values using the leap-frog scheme
    derived from equations 4.16 and 4.17.
    """
    for pt in np.arange(1, n_grid - 1):

        c.next[pt] = c.now[pt] - u*(dt/dx)*(c.now[pt] - c.now[pt-1])  + k* (dt/(dx**2))*(c.now[pt + 1] - 2*c.now[pt] + c.now[pt - 1]) 
    

#     Alternate vectorized implementation:
#     u.next[1:n_grid - 1] = (u.prev[1:n_grid - 1]
#                             - gu * (h.now[2:n_grid] - h.now[:n_grid - 2]))
#     h.next[1:n_grid - 1] = (h.prev[1:n_grid - 1]
#                             - gh * (u.now[2:n_grid] - u.now[:n_grid - 2]))

def analytic(c_an,u,k,t,x,n_grid,c1):
    for i in range(0,len(x)):
        if i<int(n_grid/3):
            c =0
        else:
            c = (c1/2)*math.erfc((x[i]-(u*t))/(2*k*t))
        c_an.next[i] = c



def Crank_Nicolson(c, u, k, n_grid, dt, dx,c1):
    
    a = (u*dt)/(4*dx)
    b = (k*dt)/(2*(dx**2))
    
    cof1 = [-a-b, 1+2*b, a-b]
    cof2 = [a+b, 1-2*b, b-a]
    

    cof1_all = np.zeros((n_grid,n_grid))
    cof2_all = np.zeros((n_grid,n_grid))
    init = np.zeros(n_grid)
    cof2_all[0,:] = init
    init[0] = 1
    init[1] = -1
    cof1_all[0,:] = init
    for pt in np.arange(1, n_grid - 1):
        c_cur = np.zeros(n_grid)
        c_cur[pt-1] = -a-b
        c_cur[pt] = 1+2*b
        c_cur[pt+1] = a-b
        cof1_all[pt,:]=c_cur
        c_cur1 = np.zeros(n_grid)
        c_cur1[pt-1] = a+b
        c_cur1[pt] = 1-2*b
        c_cur1[pt+1] = b-a
        cof2_all[pt,:] = c_cur1
    lstit = np.zeros(n_grid)
    cof2_all[-1,:]= lstit
    lstit[-1] = 1
    lstit[-2] = -1
    cof1_all[-1,:] = lstit
    c_mid = c.now.copy()
    rhs = np.matmul(cof2_all,np.transpose(c_mid))
    nxt = np.matmul(np.linalg.inv(cof1_all),rhs)
    for pt in np.arange(1, n_grid - 1):
        c.next[pt] = max(0,nxt[pt])
    c.next[int(n_grid/3)] =  c1
#     c.next[int(n_grid/3)+1] =  c1
#     c.next[int(n_grid/3)+2] =  c1
#     c.next[int(n_grid/3)+3] =  c1
    
#     while True:
#         c_old = c.next
#         for pt in np.arange(1, n_grid - 1):
#             c.next[pt] = c.now[pt] - ((u*dt)/(4*dx))*(c_old[pt+1] - c_old[pt-1] + c.now[pt+1] - c.now[pt-1]) + ((k*dt)/(2*(dx**2)))*(c_old[pt-1] - 2*c_old[pt] + c_old[pt+1] + c.now[pt-1] - 2*c.now[pt] + c.now[pt+1])
        
#         change = (c.next - c_old)/max(c_old)
#         if (max(change) <0.05):               
#             break


def Upstream(c, u, k, n_grid, dt, dx,c1):

#    for pt in np.arange(1, n_grid - 1):
#        c.next[pt] = c.now[pt] - u*(dt/dx)*(c.now[pt + 1] - c.now[pt])  + k* (dt/(dx**2))*(c.now[pt + 1] - 2*c.now[pt] + c.now[pt - 1]) 

    for pt in np.arange(1, n_grid - 1):
        c.next[pt] = c.now[pt] - u*(dt/dx)*(c.now[pt] - c.now[pt - 1])  + ((k*dt)/(dx**2))*(c.now[pt + 1] - 2*c.now[pt] + c.now[pt - 1])         
    c.next[int(n_grid/3)] = c1


def FCTS(c, u, k, n_grid, dt, dx):

#    for pt in np.arange(1, n_grid - 1):
#        c.next[pt] = c.now[pt] - u*(dt/dx)*(c.now[pt + 1] - c.now[pt])  + k* (dt/(dx**2))*(c.now[pt + 1] - 2*c.now[pt] + c.now[pt - 1]) 

    for pt in np.arange(1, n_grid - 1):
        c.next[pt] = c.now[pt] - u*(dt/(2*dx))*(c.now[pt+1] - c.now[pt - 1]) # + ((k*dt)/(dx**2))*(c.now[pt + 1] - 2*c.now[pt] + c.now[pt - 1])       
    #print((k*dt)/(dx**2))

    
def nsdf(c,u,K, n_grid,dt,dx, c1):
  """The non-standard finite difference method given by (Appadu, 2013, doi: 10.1155/2013/734374)"""
  k = 5 # given by paper as most accurate coefficient
  h = 20  # given by paper as most accurate coeficient
  a1 = k/h  # coefficient to tidy up equation
  b1 = a1/(np.exp(h/K)-1) # coefficient to tidy up equation
 
  for pt in np.arange(1, n_grid-1):
    c.next[pt] = b1*c.now[pt+1]+(1-a1*u-2*b1)*c.now[pt]+(a1*u+b1)*c.now[pt-1]   
    
  c.next[int(n_grid/3)] =  c1

def Lax_Wendroff(c, u, k, n_grid, dt, dx, c1):
    """Calculate the next time step values using the leap-frog scheme
    derived from equations 4.16 and 4.17.
    """
    for pt in np.arange(2, n_grid - 2): #note that the grid range is changed
        c.next[pt] = c.now[pt] - u*(dt/(2*dx))*(c.now[pt + 1] - c.now[pt-1])  + (u**2*dt+2*k) * (dt/(4*dx**2))*(c.now[pt + 1] - 2*c.now[pt] + c.now[pt - 1]) - (k*u*dt**2)/(2*dx**3)*(-c.now[pt-2]+2*c.now[pt-1]-2*c.now[pt+1]+c.now[pt+2])
#         c.next[pt] = c.now[pt] - u*(dt/(2*dx))*(c.now[pt + 1] - c.now[pt-1])  + (u**2*dt) * (dt/(2*dx^2))*(c.now[pt + 1] - 2*c.now[pt] + c.now[pt - 1]) 
        if (c.next[pt]<0):
            c.next[pt]=0
    c.next[int(n_grid/3)] = c1
    
def make_graph(c, dt, n_time):
    """Create graphs of the model results using matplotlib.

    You probably need to run the rain script from within ipython,
    in order to see the graphs.  And
    the default run for 5 time steps doesn't produce much of interest;
    try at least 100 steps.
    """

    # Create a figure with 2 sub-plots
    fig = plt.figure(figsize=(14,7))
    
    
    # Set the figure title, and the axes labels.
    ax_c = fig.add_subplot(121)
    ax_c.set_title(f'Results from numerical dt = {dt}s & dx =1000m')
    ax_c.set_ylabel('c [mg/m^3]')
    #ax_h.set_ylabel('h [cm]')
    ax_c.set_xlabel('Grid Point')

    # We use color to differentiate lines at different times.  Set up the color map
    cmap = plt.get_cmap('viridis')
    cNorm  = colors.Normalize(vmin=0, vmax=1.*n_time)
    cNorm_inseconds = colors.Normalize(vmin=0, vmax=1.*n_time*dt)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    # Only try to plot 20 lines, so choose an interval if more than that (i.e. plot
    # every interval lines
    interval = np.int(np.ceil(n_time/500))
    print('where')
    # Do the main plot
    for time in range(0, n_time, interval):
        colorVal = scalarMap.to_rgba(time)
        ax_c.plot(c[0].store[:, time], color=colorVal)
        #ax_h.plot(h.store[:, time], color=colorVal)

    ax_c = fig.add_subplot(122)
    ax_c.set_title(f'Results from analytical solution')
    ax_c.set_ylabel('c [mg/m^3]')
    #ax_h.set_ylabel('h [cm]')
    ax_c.set_xlabel('Grid Point')
    
    # Do the main plot
    for time in range(0, n_time, interval):
        colorVal = scalarMap.to_rgba(time)
        ax_c.plot(c[1].store[:, time], color=colorVal)
    
    # Add the custom colorbar
    ax2 = fig.add_axes([0.95, 0.05, 0.05, 0.9])
    cb1 = colorbar.ColorbarBase(ax2, cmap=cmap, norm=cNorm_inseconds)
    cb1.set_label('Time (s)')
    
    return

def numeric(args):
    """Run the model.

    args is a 2-tuple; (number-of-time-steps, number-of-grid-points)
    """
    n_time = int(args[0])
    n_grid = int(args[1])
#     Alternate implementation:
#     n_time, n_grid = map(int, args)

    # Constants and parameters of the model
    u = 3.39 #wind speed in x direction in m/s
    k = 2.0   #eddy diffusivity coefficient in m^2/s
    c1 = 600.0   #initial pollution amount g/m3
    domain_length = 360000 #360km
 
    dx = 1000   #stepsize
    courant = 0.339 #keep less than 1 for advection diffusion
    dt = 30                  # time step [s]
    # Create velocity and surface height objects
    c = Quantity(n_grid, n_time)
    c_an = Quantity(n_grid, n_time)

    # Set up initial conditions and store them in the time step
    # results arrays

    initial_conditions(c1,c, n_grid)
    initial_conditions(c1,c_an, n_grid)
    #c.store_timestep(0, 'now')

    # Calculate the first time step values from the
    # predictor-corrector, apply the boundary conditions, and store
    # the values in the time step results arrays
    
    #first_time_step(u, h, g, H, dt, dx, ho, gu, gh, n_grid)
    boundary_conditions(c.now, n_grid)
    boundary_conditions(c_an.now, n_grid)
    
    c.store_timestep(0, 'now')
    c_an.store_timestep(0, 'now')
    

    x = np.zeros(n_grid)
    dxx = 0
    for i in range(n_grid):
        if(i < int(n_grid/3)):
            pass
        else:
            x[i]=x[i]+dxx
            dxx = dxx + dx
      
    # Time step loop using leap-frog scheme
    t_val = 0
    for t in np.arange(1, n_time):
        # Advance the solution and apply the boundary conditions
        Upstream(c, u, k, n_grid, dt, dx,c1)
        t_val = t_val + dt
        analytic(c_an,u,k,t_val,x,n_grid,c1)
        boundary_conditions(c.next, n_grid)
        boundary_conditions(c_an.next, n_grid)
        # Store the values in the time step results arrays, and shift
        # .now to .prev, and .next to .now in preparation for the next
        # time step

        c.store_timestep(t)
        c_an.store_timestep(t)
        c.shift()
        c_an.shift()
    
    cs = [c,c_an]
    # Plot the results as colored graphs
    make_graph(cs, dt, n_time)
    return


if __name__ == '__main__':
    # sys.argv is the command-line arguments as a list. It includes
    # the script name as its 0th element. Check for the degenerate
    # cases of no additional arguments, or the 0th element containing
    # `sphinx-build`. The latter is a necessary hack to accommodate
    # the sphinx plot_directive extension that allows this module to
    # be run to include its graph in sphinx-generated docs.
    #
    #  the following command, executed in the plotfile directory makes a movie on ubuntu called
    #   outputmplt.avi
    #  which can be
    #  looped with mplayer -loop 0
    #
    #  mencoder mf://*.png -mf type=png:w=800:h=600:fps=25 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o outputmplt.avi
    #
    if len(sys.argv) == 1 or 'sphinx-build' in sys.argv[0]:
        # Default to 50 time steps, and 9 grid points
        rain((50, 9))
        plt.show()
    elif len(sys.argv) == 3:
        # Run with the number of time steps and grid point the user gave
        rain(sys.argv[1:])
        plt.show()
    else:
        print ('Usage: rain n_time n_grid')
        print ('n_time = number of time steps; default = 5')
        print ('n_grid = number of grid points; default = 9')
