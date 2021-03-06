from casadi import *
import numpy as np

# declaring some parameters
# turning radius
R = 20

# car parameter
lf = 0.8  # front wheel base
lr = 0.8  # rear wheel base
m = 250  # vehicle mass
h = 0.3  # cog height
track = 1.2

# tire parameters
A = 1800
B = 1.5
C = 25
D = 1
E = 20

# declaring symbolic variables
x1 = SX.sym("x1", 4)

# easier name for symbolic variables
delta = x1[0]  # steering angle of the front wheel
beta = x1[1]  # car side slip angle
slip_ratio_front = x1[2]  # front longitudinal slip ratio
slip_ratio_rear = x1[3]  # rear longitudinal slip ratio

# declaring a symbolic parameter
ay1 = SX.sym("ay1", 1)

# starting to create the symbolic expression for the acceleration
vel = sqrt(ay1 * R)
vx = vel * cos(beta)
vy = vel * sin(beta)

yaw_rate = vel / R

# casadi function to calculate final car yaw rate
f_yaw_rate = Function("f_yaw_rate", [ay1], [yaw_rate])

Saf = -delta + atan((vy + lf * yaw_rate) / vx)  # front slip angle
Sar = atan((vy - lr * yaw_rate) / vx)  # rear slip angle

# casadi function to calculate slip angles
f_saf = Function("f_saf", [delta, beta, ay1], [Saf])
f_sar = Function("f_sar", [delta, beta, ay1], [Sar])

# move ay ( which is an) to the right reference system to calculate load transfers
# ax1 = - ay1 * sin(beta)
ay3 = ay1 * cos(beta)

fz_fr = m * 9.81 * lr / ( (lr + lf) * 2 ) - ay3 * ( h / track )
fz_fl = m * 9.81 * lr / ( (lr + lf) * 2 ) + ay3 * ( h / track )
fz_rr = m * 9.81 * lf / ( (lr + lf) * 2 ) - ay3 * ( h / track )
fz_rl = m * 9.81 * lf / ( (lr + lf) * 2 ) + ay3 * ( h / track )

f_fzfr = Function("f_fzfr", [delta, beta, ay1], [fz_fr])
f_fzfl = Function("f_fzfl", [delta, beta, ay1], [fz_fl])
f_fzrr = Function("f_fzrr", [delta, beta, ay1], [fz_rr])
f_fzrl = Function("f_fzrl", [delta, beta, ay1], [fz_rl])

load_f = ( 1.5 * atan( fz_fr / 900 ) + 1.5 * atan( fz_fl / 900 ) ) / 2
load_r = ( 1.5 * atan( fz_rr / 900 ) + 1.5 * atan( fz_rl / 900 ) ) / 2

f_loadf = Function("f_loadf", [delta, beta, ay1], [load_f])
f_loadr = Function("f_loadr", [delta, beta, ay1], [load_r])

# tire force calculation
# first we calculate the pure tire force the the combined
Fxpf = load_f * A * sin(B * atan(C * slip_ratio_front))
Fypf = - load_f * A * sin(B * atan(C * tan(Saf)))
Fxpr = load_r * A * sin(B * atan(C * slip_ratio_rear))
Fypr = - load_r * A * sin(B * atan(C * tan(Sar)))


# casadi function to calculate lateral pure tire forces
f_fypf = Function("f_fypf", [delta, beta, ay1], [Fypf])
f_fypr = Function("f_fypr", [delta, beta, ay1], [Fypr])

# combined tire force
Fxf = Fxpf * cos(D * atan(E * tan(Saf)))
Fyf = Fypf * cos(D * atan(E * slip_ratio_front))
Fxr = Fxpr * cos(D * atan(E * tan(Sar)))
Fyr = Fypr * cos(D * atan(E * slip_ratio_rear))

# casadi function to calculate lateral combined tire forces
f_fyf = Function("f_fyf", [delta, beta, slip_ratio_front, ay1], [Fyf])
f_fyr = Function("f_fyr", [delta, beta, slip_ratio_rear, ay1], [Fyr])

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
opt_var = [delta, beta, slip_ratio_front, slip_ratio_rear]

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
x0 = [0.01, 0.01, 0.01, 0.01]

# declaring the lower and upper bound of the variable to optimize
lbx = [-0.52, -0.34, -0.1, -0.1]
ubx = [0.52, 0.34, 0.1, 0.1]

# initializing some list used to check the loop
opt_x = [0, 0, 0, 0]
sol_x = [0, 0, 0, 0]

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

    # print("sol x before for loop is", sol_x)
    # print("opt x before for loop is", opt_x)

    # loop to check if the optimal variables found differ with the previous one more than the tolerance
    # if the difference is 'big' another optimization will take place with the new value of ay1 (ay1 is starting ay to
    # get the car velocity)
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
print("objective at solution = ", sol["f"], "(normal acceleration)")
print("primal solution = ", sol["x"], "(Steering angle, slip angle, front and rear slip ratio)")
print("dual solution (x) = ", sol["lam_x"])
print("dual solution (g) = ", sol["lam_g"])
print("constraints value = ", sol["g"])
print("-----")
print("Vel is [m/s]", f_vel(ay2))
print("ay2 is [m/s^2]", ay2)
print("-----")
print("Front slip angle [rad] ", f_saf(sol["x"][0], sol["x"][1], ay2))
print("Rear slip angle [rad] ", f_sar(sol["x"][0], sol["x"][1], ay2))
print("-----")
print("Front lateral pure force [N]", f_fypf(sol["x"][0], sol["x"][1], ay2))
print("Rear lateral pure force [N]", f_fypr(sol["x"][0], sol["x"][1], ay2))
print("-----")
print("Front lateral combined force [N]", f_fyf(sol["x"][0], sol["x"][1], sol["x"][2], ay2))
print("Rear lateral combined force [N]", f_fyr(sol["x"][0], sol["x"][1], sol["x"][3], ay2))
print("-----")
print("Yaw rate is [1/s]", f_yaw_rate(ay2))
print("-----")
print("Front right vertical load ", f_fzfr(sol["x"][0], sol["x"][1], ay2))
print("Front left vertical load ", f_fzfl(sol["x"][0], sol["x"][1], ay2))
print("Rear right vertical load ", f_fzrr(sol["x"][0], sol["x"][1], ay2))
print("Real left vertical load ", f_fzrl(sol["x"][0], sol["x"][1], ay2))
print("-----")
print("Front load coefficent ", f_loadf(sol["x"][0], sol["x"][1], ay2))
print("Rear load coefficent ", f_loadr(sol["x"][0], sol["x"][1], ay2))
