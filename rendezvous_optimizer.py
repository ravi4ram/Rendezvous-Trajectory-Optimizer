# Implemented Rendezvous and docking trajectory optimization problem
# based on the paper using convex optimization techniques:
#     Probabilistic Trajectory Optimization Under Uncertain Path Constraints
#     for Close Proximity Operations
#     [ JOURNAL OF GUIDANCE, CONTROL, AND DYNAMICS ]
#     Christopher Jewison and David W. Miller
#     Massachusetts Institute of Technology, Cambridge, Massachusetts 02139
#
# Author: ravi_ram

import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt


# mu - gravitational parameter (m^3/s^2)
mu = 3.986e14
# radius of Earth (m)
r_e = 6.371e6

# Clohessy Wiltshire Equations in state space form
def getAB(n, dt):
    nt = n*dt
    snt, cnt = np.sin(nt), np.cos(nt)
    
    # Ad matrix
    Ad = np.array([[    4.0 - 3.0 * cnt, 0.0,      0.0,         (1.0 / n) * snt,              (2.0 / n) * (1 - cnt),             0.0],
                 [     6.0 * (snt - nt), 1.0,      0.0, (2.0 / n) * (cnt - 1.0), (1.0 / n) * (4.0 * snt - 3.0 * nt),             0.0],
                 [                  0.0, 0.0,      cnt,                     0.0,                                0.0, (1.0 / n) * snt],
                 [        3.0 * n * snt, 0.0,      0.0,                     cnt,                          2.0 * snt,             0.0],
                 [6.0 * n * (cnt - 1.0), 0.0,      0.0,              -2.0 * snt,                    4.0 * cnt - 3.0,             0.0],
                 [                  0.0, 0.0, -n * snt,                     0.0,                                0.0,             cnt]
                ])
    # Bd matrix
    Bd = np.array([[ (1/n*n) * (1 - cnt),      (2/n*n) * (nt - snt),                 0 ],
                  [ -(2/n*n) * (nt - snt),    ((4/n*n) * (1 - cnt)) - 3*dt*dt/2,    0 ],
                  [ 0,                        0,                                    (1/n*n) * (1 - cnt)],
                  [ 1/n * snt,                (2/n) * (1 - cnt),                    0 ],
                  [ -(2/n) * (1 - cnt),       ((4/n) * snt) - 3*dt,                 0 ],
                  [ 0,                        0,                                    1/n * snt]
                  ])
    return Ad, Bd

def estimate():
    # # Goal is the target position - Target Spacecraft State (m)
    xT = np.array([0.,0.,0.,0.,0.,0.]) #
    # print(x0_target)

    # Chaser Spacecraft # initial position in target center coordinate frame
    x0 = np.array([-10103.780725584133, 19588.531139346305, 16220.16171565,
                   85.37754802465861, -89.64878002435307, -94.88549429088607])

    
    # mass of the satellite (kg)
    m = 220.0 # kg
    # Thrusters: 1N (9 Nos)
    max_thrust = 4.0 # (N)
    min_thrust = 0.5 # (N)

    # radius of the target body's circular orbit (m)
    a = 6848.26 * 1E3 # meters
    # constant mean motion of target n = sqrt(mu/a^3) (rad/s)
    n = np.sqrt(mu / a**3)

    # states (x, y, z, vx, vy, vz)
    nx = 6
    # control inputs (Fx, Fy, Fz)
    nu = 3

    # for 4 minutes control points
    N = (60 * 4) + 1    
    # timestep (seconds)
    dt = 1
      
    # A and B matrix
    A, B = getAB(n, dt)
    
    # state variable   - (x, y, z, vx, vy, vz)
    x = cp.Variable((nx, N+1))     
    # control variable - (Fx, Fy, Fz)
    u = cp.Variable((nu, N))
    # objective
    objective = cp.Minimize(cp.norm(u, 1) )   
    # constraints
    constraints = []
    # initial state vector ( relative to target centered frame )
    constraints.append( x[:,0] == x0  )
    # target state vector ( target centered frame )
    constraints.append( x[:,N-1] == xT )
    
    for k in range(N):
        # set dynamics constraints
        constraints.append( x[:,k+1] == A @ x[:,k] + B @ u[:,k] )
        # set thrust limit constraint
        constraints.append( cp.norm(u[:,k] / m, 2) <= min_thrust )
    # assemble Problem
    prob = cp.Problem(objective, constraints)
    # solve
    prob.solve(solver=cp.CLARABEL)
    
    # on success plot the result
    if prob.status == 'optimal' :
        plot(x.value, u.value)        
    else:
        print(f'STATUS : {prob.status}')
    
    # return
    return

# The Clohessy–Wiltshire–Hill (CWH) frame is centered on the target spacecraft,
# with the i^ direction pointing radially outward from the Earth;
# the j^ direction pointing in the intrack, orbital velocity direction of the
# target satellite and the k^ direction pointing in the crosstrack direction,
# out of the orbital plane, to complete the orthogonal set.
def plot(X, U):
    # km conversion for display
    X = np.round(X, 2)/1E3
    
    x, y, z    = X[2, :], X[1, :], X[0, :]
    xi, yi, zi = X[2, 0], X[1, 0], X[0, 0]
    xf, yf, zf = X[2, -1], X[1, -1], X[0, -1]
    
    #U = control data
    ux, uy, uz = U[2, :].T, U[1, :].T, U[0, :].T
    
    # Set 3D plot axes to equal scale. 
    # Required since `ax.axis('equal')` and `ax.set_aspect('equal')` don't work on 3D.
    # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    def set_axes_equal_3d(ax: plt.Axes):    
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        # set axes
        x, y, z = origin
        ax.set_xlim3d([x - radius, x + radius])
        ax.set_ylim3d([y - radius, y + radius])
        ax.set_zlim3d([z - radius, z + radius])
        #
        return
    
    # setup plot canvas
    fig = plt.figure(figsize=(8, 5), constrained_layout=True)
    fig.suptitle('Rendezvous and docking Trajectory Optimization', fontsize=10)
    gs = fig.add_gridspec(3,3)

    # 1st col
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ux, '-b', linewidth=0.8)
    ax1.set_title('Control History', fontsize=7)
    ax1.set_ylabel(r'Fx ($m/s^2$)', fontsize=7)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(uy, '-b', linewidth=0.8)
    ax2.set_ylabel(r'Fy ($m/s^2$)', fontsize=7)
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(uz, '-b', linewidth=0.8)   
    ax3.set_ylabel(r'Fz ($m/s^2$)', fontsize=7)
    ax3.set_xlabel('Timestep', fontsize=9)

    # 2nd col - 3-rows
    ax4 = fig.add_subplot(gs[:, 1:3], projection='3d') 
    ax4.view_init(azim=40., elev=12.)
    ax4.plot(x, y, z, '-b', linewidth=1.5)
    ax4.scatter(xi, yi, zi, color='green')
    ax4.scatter(xf, yf, zf, color='red')    
    ax4.set_title('Chaser Trajectory', fontsize=7)
    ax4.set(xlabel='Z-Out of Plane (km)',
            ylabel='Y-Along Track (km)',
            zlabel='X-Radial (km)')
    ax4.legend(['Trajectory', 'Start Point', 'End Point'], loc='upper center',
              bbox_to_anchor=(0.5, -0.1), frameon=True, borderaxespad=0, ncol = 3)

    # set correct aspect ratio
    ax4.set_box_aspect([1,1,1])
    set_axes_equal_3d(ax4)    

    # set ticks and grid lines
    for ax in [ax1, ax2, ax3]:
        # set grid lines
        ax.grid(True, which='major', color='k', linestyle='-', lw=0.2, alpha=0.5)
        ax.grid(True, which='minor', color='k', linestyle='--', lw=0.2, alpha=0.5)
        # set ticks
        ax.minorticks_on()
        ax.xaxis.set_tick_params(labelsize=7, rotation=90)
        ax.yaxis.set_tick_params(labelsize=7)    
    # end-for
    
    plt.show()
    # return
    return
    
# __main method__ 
if __name__=="__main__":   
    # solve   
    estimate()
