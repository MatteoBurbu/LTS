from casadi import *
import numpy as np

# declaring some parameters
# turning radius
R = 20

# car parameter
lf = 0.8  # front wheel base
lr = 0.8  # rear wheel base
m = 250  # vehicle mass

# tire parameters
A = 1800
B = 1.5
C = 25
D = 1
E = 20

# declaring symbolic variables
x1 = SX.sym("x1", 5)

# easier name for symbolic variables
delta = x1[0]  # steering angle of the front wheel
beta = x1[1]  # car side slip angle
yaw_rate = x1[2]  # car yaw rate
slip_ratio_front = x1[3]  # front longitudinal slip ratio
slip_ratio_rear = x1[4]  # rear longitudinal slip ratio

# declaring a symbolic parameter
ay1 = SX.sym("ay1", 1)

# starting to create the symbolic expression for the acceleration
vel = sqrt(ay1/R)
vx = vel * cos(beta)
vy = vel * sin(beta)

Saf = -delta + atan((vy + lf * yaw_rate) / vx)  # front slip angle
Sar = atan((vy - lr * yaw_rate) / vx)  # rear slip angle

# tire force calculation
# first we calculate the pure tire force the the combined
Fxpf = A * sin(B * atan(C * slip_ratio_front))
Fypf = -A * sin(B * atan(C * tan(Saf)))
Fxpr = A * sin(B * atan(C * slip_ratio_rear))
Fypr = -A * sin(B * atan(C * tan(Sar)))

# combined tire force
Fxf = Fxpf * cos(D * atan(E * tan(Saf)))
Fyf = Fypf * cos(D * atan(E * slip_ratio_front))
Fxr = Fxpr * cos(D * atan(E * tan(Sar)))
Fyr = Fypr * cos(D * atan(E * slip_ratio_rear))

# force calculation on the vehicle
Fy = Fyf * cos(delta) + Fxf * sin(delta) + Fyr
Fx = Fxf * cos(delta) - Fyf * sin(delta) + Fxr
Mz = lf * (Fyf * cos(delta) + Fxf * sin(delta)) - lr * Fyr

# acceleration along the trajectory
ay = 1/m * (Fy * cos(beta) - Fx * sin(beta))
ax = 1/m * (Fy * sin(beta) + Fx * cos(beta))

# calculating the power to use as a constraint
Power = Fxf * (vx * cos(delta) + vy * sin(delta)) * (1 + slip_ratio_front) + Fxr * vx * (1 + slip_ratio_rear)

# variable to optimize are grouped together
opt_var = [delta, beta, yaw_rate, slip_ratio_front, slip_ratio_rear]

# a casadi function is created to evaluate quickly the car speed
f_vel = Function('vel', [ay1],  [vel])

# create the nlp problem
# ay1 (used to calculate the speed) is used as parameter
nlp = {'x': x1, 'f': -ay, 'g': Power, 'p': ay1}

# Pick an NLP solver
MySolver = "ipopt"
# MySolver = "worhp"
# MySolver = "sqpmethod"

# Solver options
opts = {}
if MySolver == "sqpmethod":
    opts["qpsol"] = "qpoases"
    opts["qpsol_options"] = {"printLevel": "none"}

# Allocate a solver
solver = nlpsol("solver", MySolver, nlp, opts)

# declaring the starting point
x0 = [0.01, 0.01, 0.01, 0.01, 0.01]

# declaring the lower and upper bound of the variable to optimize
lbx = [-0.52, -0.34, -1.57, -0.1, -0.1]
ubx = [0.52, 0.34, 1.57, 0.1, 0.1]

# initializing some list used to check the loop
opt_x = [0, 0, 0, 0, 0]
sol_x = [0, 0, 0, 0, 0]

# max difference between two optimal variable to stop the while loop
tol = 0.01
go_ahead = True

# declaring a starting value for the parameter ay1 (ay1 is the symbolic variable, ay2 is its value)
ay2 = 0.1

while go_ahead:
    # solving the problem
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=[0], ubg=[80000], p=ay2)

    # print("vel is", f_vel(ay2))
    # print(ay2)

    # casadi variables need to be converted to loop through them
    sol_x[0] = sol["x"][0].__float__()
    sol_x[1] = sol["x"][1].__float__()
    sol_x[2] = sol["x"][2].__float__()
    sol_x[3] = sol["x"][3].__float__()
    sol_x[4] = sol["x"][4].__float__()

    # print("sol x before for loop is", sol_x)
    # print("opt x before for loop is", opt_x)

    # loop to check if the optimal variables found differ with the previous one more than tol
    # if the difference is 'big' another optimization will take place with the new value of ay1
    for x0, x1 in zip(opt_x, sol_x):
        if np.abs(x0-x1) < tol:
            go_ahead = False
            # print(go_ahead)
            # print("keep checking")
        else:
            go_ahead = True
            opt_x = sol_x
            ay2 = -sol["f"][0].__float__()
            # print(go_ahead)
            # print("break")
            break

    # print("sol x after for loop is", sol_x)
    # print("opt x after for loop is", opt_x)


# Print solution
print("-----")
print("objective at solution = ", sol["f"])
print("primal solution = ", sol["x"])
print("dual solution (x) = ", sol["lam_x"])
print("dual solution (g) = ", sol["lam_g"])
print("constraints value = ", sol["g"])
print("-----")

# print(f_vel(ay2))
# print(ay2)
