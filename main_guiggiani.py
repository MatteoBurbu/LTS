from casadi import *
import numpy as np

# declaring some parameters
# turning radius
Rg = 20

# car parameter
lf = 0.81  # front wheel base
lr = 0.79  # rear wheel base
m = 250  # vehicle mass
h = 0.3  # cog height
track = 1.2
q1 = 0.04  # front roll center height [m]
q2 = 0.1  # rear roll center height [m]
qb = (lf * q1 + lr * q2) / (lf + lr)

# tire parameters
A = 1800
B = 1.5
C = 25
D = 1
E = 20

k_susp_f = 50 * 1000  # Nm/rad (just spring and ARB, rigid tyres)
k_susp_r = 50 * 1000  # Nm/rad
k_chassis = 100 * 1000  # Nm/rad

p_tire = 120 * 1000  # N/m
k_tyre_f = (p_tire * track ** 2) / 2  # Nm/rad (just tyres, rigid susp and ARB)
k_tyre_r = (p_tire * track ** 2) / 2

k_f = k_susp_f * k_tyre_f / (k_susp_f + k_tyre_f)  # Nm/rad (tyres and suspension)
k_r = k_susp_r * k_tyre_r / (k_susp_r + k_tyre_r)

k_roll = 1 / (1 / k_chassis + 1 / k_f + 1 / k_r)  # Nm/rad

# declaring symbolic variables
x1 = SX.sym("x1", 4)

# easier name for symbolic variables
delta = x1[0]  # steering angle of the front wheel
beta = x1[1]  # car side slip angle
slip_ratio_front = x1[2]  # front longitudinal slip ratio
slip_ratio_rear = x1[3]  # rear longitudinal slip ratio

# declaring a symbolic parameter

an1 = SX.sym("an1", 1)
vel = sqrt(Rg * an1)
R = Rg * cos(beta)
vx = vel * cos(beta)
vy = vel * sin(beta)
yaw_rate = vx / R
ax = - vy * yaw_rate
ay1 = an1 * cos(beta)
# casadi function to calculate final car yaw rate
f_yaw_rate = Function("f_yaw_rate", [beta, an1], [yaw_rate])
f_ax = Function("f_ax", [beta, an1], [ax])

wr_11 = ((vx - yaw_rate * track * 0.5) * cos(delta) + (vy + yaw_rate * lf * 0.5) * sin(delta)) / (1 + slip_ratio_front)
wr_12 = ((vx + yaw_rate * track * 0.5) * cos(delta) + (vy + yaw_rate * lf * 0.5) * sin(delta)) / (1 + slip_ratio_front)
wr_21 = (vx - yaw_rate * track * 0.5) * cos(delta) / (1 + slip_ratio_rear)
wr_22 = (vx + yaw_rate * track * 0.5) * cos(delta) / (1 + slip_ratio_rear)

slip_y_11 = ((vy + yaw_rate * lf) * cos(delta) - (vx - yaw_rate * track * 0.5) * sin(delta)) / wr_11
slip_y_12 = ((vy + yaw_rate * lf) * cos(delta) - (vx + yaw_rate * track * 0.5) * sin(delta)) / wr_12
slip_y_21 = (vy - yaw_rate * lr) / wr_21
slip_y_22 = (vy - yaw_rate * lr) / wr_22

f_sy_11 = Function("f_sy_11", [delta, beta, slip_ratio_front, an1], [slip_y_11])
f_sy_12 = Function("f_sy_12", [delta, beta, slip_ratio_front, an1], [slip_y_12])
f_sy_21 = Function("f_sy_12", [delta, beta, slip_ratio_rear, an1], [slip_y_21])
f_sy_22 = Function("f_sy_12", [delta, beta, slip_ratio_rear, an1], [slip_y_22])

# move ay ( which is an) to the right reference system to calculate load transfers
# ax1 = - ay1 * sin(beta)
# ay3 = ay1 * cos(beta)

Y1 = m * ay1 / (1 + lf / lr)
Y2 = Y1 * (lf / lr)

Delta_Z_1 = (1 / track) * ((k_f / k_roll) * m * ay1 * (h - qb) + Y1 * q1 + (k_f * k_r / k_roll) * (
            Y2 * q2 / k_tyre_r - Y1 * q1 / k_tyre_f))
Delta_Z_2 = (1 / track) * ((k_r / k_roll) * m * ay1 * (h - qb) + Y2 * q2 + (k_f * k_r / k_roll) * (
            Y1 * q1 / k_tyre_f - Y2 * q2 / k_tyre_r))

fz_11 = 0.5 * (m * 9.81 * lr / (lr + lf) - m * ax * h / (lr + lf)) - Delta_Z_1
fz_12 = 0.5 * (m * 9.81 * lr / (lr + lf) - m * ax * h / (lr + lf)) + Delta_Z_1
fz_21 = 0.5 * (m * 9.81 * lf / (lr + lf) + m * ax * h / (lr + lf)) - Delta_Z_2
fz_22 = 0.5 * (m * 9.81 * lf / (lr + lf) + m * ax * h / (lr + lf)) + Delta_Z_2

fz_11 = if_else(fz_11 > 0, fz_11, 0)
fz_12 = if_else(fz_12 > 0, fz_12, 0)
fz_21 = if_else(fz_21 > 0, fz_21, 0)
fz_22 = if_else(fz_22 > 0, fz_22, 0)

f_fz11 = Function("f_fz11", [delta, beta, an1], [fz_11])
f_fz12 = Function("f_fz12", [delta, beta, an1], [fz_12])
f_fz21 = Function("f_fz21", [delta, beta, an1], [fz_21])
f_fz22 = Function("f_fz22", [delta, beta, an1], [fz_22])

load_11 = 1.5 * atan(fz_11 / 900)
load_12 = 1.5 * atan(fz_12 / 900)
load_21 = 1.5 * atan(fz_21 / 900)
load_22 = 1.5 * atan(fz_22 / 900)

f_load_11 = Function("f_load_11", [delta, beta, an1], [load_11])
f_load_12 = Function("f_load_12", [delta, beta, an1], [load_12])
f_load_21 = Function("f_load_21", [delta, beta, an1], [load_21])
f_load_22 = Function("f_load_22", [delta, beta, an1], [load_22])

# tire force calculation
# first we calculate the pure tire force the the combined
Fxp11 = 0.5 * load_11 * A * sin(B * atan(C * slip_ratio_front))
Fxp12 = 0.5 * load_12 * A * sin(B * atan(C * slip_ratio_front))
Fxp21 = 0.5 * load_21 * A * sin(B * atan(C * slip_ratio_rear))
Fxp22 = 0.5 * load_22 * A * sin(B * atan(C * slip_ratio_rear))

Fyp11 = - 0.5 * load_11 * A * sin(B * atan(C * tan(slip_y_11)))
Fyp12 = - 0.5 * load_12 * A * sin(B * atan(C * tan(slip_y_12)))
Fyp21 = - 0.5 * load_21 * A * sin(B * atan(C * tan(slip_y_21)))
Fyp22 = - 0.5 * load_22 * A * sin(B * atan(C * tan(slip_y_22)))

# combined tire force
Fx11 = 0.5 * Fxp11 * cos(D * atan(E * tan(slip_y_11)))
Fx12 = 0.5 * Fxp12 * cos(D * atan(E * tan(slip_y_12)))
Fx21 = 0.5 * Fxp21 * cos(D * atan(E * tan(slip_y_21)))
Fx22 = 0.5 * Fxp22 * cos(D * atan(E * tan(slip_y_22)))

Fy11 = 0.5 * Fyp11 * cos(D * atan(E * slip_ratio_front))
Fy12 = 0.5 * Fyp11 * cos(D * atan(E * slip_ratio_front))
Fy21 = 0.5 * Fyp21 * cos(D * atan(E * slip_ratio_rear))
Fy22 = 0.5 * Fyp22 * cos(D * atan(E * slip_ratio_rear))

f_fxp11 = Function("f_fxp11", [delta, beta, slip_ratio_front, an1], [Fxp11])
f_fxp12 = Function("f_fxp12", [delta, beta, slip_ratio_front, an1], [Fxp12])
f_fxp21 = Function("f_fxp21", [delta, beta, slip_ratio_rear, an1], [Fxp21])
f_fxp22 = Function("f_fxp22", [delta, beta, slip_ratio_rear, an1], [Fxp22])

f_fyp11 = Function("f_fyp11", [delta, beta, slip_ratio_front, an1], [Fyp11])
f_fyp12 = Function("f_fyp12", [delta, beta, slip_ratio_front, an1], [Fyp12])
f_fyp21 = Function("f_fyp21", [delta, beta, slip_ratio_rear, an1], [Fyp21])
f_fyp22 = Function("f_fyp22", [delta, beta, slip_ratio_rear, an1], [Fyp22])

# casadi function to calculate lateral combined tire forces
f_fy11 = Function("f_fy11", [delta, beta, slip_ratio_front, an1], [Fy11])
f_fy12 = Function("f_fy12", [delta, beta, slip_ratio_front, an1], [Fy12])
f_fy21 = Function("f_fy21", [delta, beta, slip_ratio_rear, an1], [Fy21])
f_fy22 = Function("f_fy22", [delta, beta, slip_ratio_rear, an1], [Fy22])

f_fx11 = Function("f_fx11", [delta, beta, slip_ratio_front, an1], [Fx11])
f_fx12 = Function("f_fx12", [delta, beta, slip_ratio_front, an1], [Fx12])
f_fx21 = Function("f_fx21", [delta, beta, slip_ratio_rear, an1], [Fx21])
f_fx22 = Function("f_fx22", [delta, beta, slip_ratio_rear, an1], [Fx22])

# force calculation on the vehicle
Fy = (Fy11 + Fy12) * cos(delta) + (Fx11 + Fx12) * sin(delta) + Fy21 + Fy22
Fx = (Fx11 + Fx12) * cos(delta) - (Fy11 + Fy12) * sin(delta) + Fx21 * Fx22
Mz = lf * ((Fy11 + Fy12) * cos(delta) + (Fx11 + Fx12) * sin(delta)) - lr * (Fy21 + Fy22)

# acceleration along the trajectory
# ay = 1 / m * (Fy * cos(beta) - Fx * sin(beta))
# ax = 1 / m * (Fy * sin(beta) + Fx * cos(beta))

ay = Fy / m
ax = Fx / m

an = -ax * sin(beta) + ay * cos(beta)

# calculating the power to use as a constraint
Power = (Fx11 + Fx12) * (vx * cos(delta) + vy * sin(delta)) * (1 + slip_ratio_front) + (Fx21 + Fx22) * vx * (
            1 + slip_ratio_rear)

# variable to optimize are grouped together
opt_var = [delta, beta, slip_ratio_front, slip_ratio_rear]

# a casadi function is created to evaluate quickly the car speed
f_vel = Function('vel', [an1], [vel])

g = vertcat(Power, Mz)

# create the nlp problem
# ay1 (used to calculate the speed) is used as parameter
nlp = {'x': x1, 'f': -an, 'g': g, 'p': an1}

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
an2 = 0.1

while go_ahead:
    # solving the problem
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=[0, -0.1], ubg=[80000, 0.1], p=an2)

    # print("vel is", f_vel(ay2))
    # print(ay2)

    # casadi variables need to be converted to loop through them
    sol_x[0] = sol["x"][0].__float__()
    sol_x[1] = sol["x"][1].__float__()
    sol_x[2] = sol["x"][2].__float__()
    sol_x[3] = sol["x"][3].__float__()

    print("sol x before for loop is", sol_x)
    print("opt x before for loop is", opt_x)

    # loop to check if the optimal variables found differ with the previous one more than the tolerance
    # if the difference is 'big' another optimization will take place with the new value of ay1 (ay1 is starting ay to
    # get the car velocity)
    for x0, x1 in zip(opt_x, sol_x):
        if np.abs(x0 - x1) < tol:
            go_ahead = False
            # print(go_ahead)
            # print("keep checking")
        else:
            go_ahead = True
            opt_x = sol_x
            an2 = -sol["f"][0].__float__()
            # print(go_ahead)
            # print("break")
            break

    # print("sol x after for loop is", sol_x)
    # print("opt x after for loop is", opt_x)

# Print solution
print("-----")
print("objective at solution = ", sol["f"][0], "(normal acceleration)")
print("primal solution = ", sol["x"][0], "(Steering angle Delta)")
print("primal solution = ", sol["x"][1], "(Side slip angle Beta)")
print("primal solution = ", sol["x"][2], "(Front slip ratio)")
print("primal solution = ", sol["x"][3], "(Rear slip ratio)")
print("dual solution (x) = ", sol["lam_x"])
print("dual solution (g) = ", sol["lam_g"])
print("constraints value = ", sol["g"])
print("-----")
print("Vel is [m/s]", f_vel(an2))
print("an2 is [m/s^2]", an2)
print("ax is [m/s^2]", f_ax(sol["x"][1], an2))
print("-----")
print("11 slip angle [rad] ", f_sy_11(sol["x"][0], sol["x"][1], sol["x"][2], an2))
print("12 slip angle [rad] ", f_sy_12(sol["x"][0], sol["x"][1], sol["x"][2], an2))
print("21 slip angle [rad] ", f_sy_21(sol["x"][0], sol["x"][1], sol["x"][3], an2))
print("22 slip angle [rad] ", f_sy_22(sol["x"][0], sol["x"][1], sol["x"][3], an2))
print("-----")
print("11 lateral pure force [N]", f_fyp11(sol["x"][0], sol["x"][1], sol["x"][2], an2))
print("12 lateral pure force [N]", f_fyp12(sol["x"][0], sol["x"][1], sol["x"][2], an2))
print("21 lateral pure force [N]", f_fyp21(sol["x"][0], sol["x"][1], sol["x"][3], an2))
print("22 lateral pure force [N]", f_fyp22(sol["x"][0], sol["x"][1], sol["x"][3], an2))
print("-----")
print("11 longitudinal pure force [N]", f_fxp11(sol["x"][0], sol["x"][1], sol["x"][2], an2))
print("12 longitudinal pure force [N]", f_fxp12(sol["x"][0], sol["x"][1], sol["x"][2], an2))
print("21 longitudinal pure force [N]", f_fxp21(sol["x"][0], sol["x"][1], sol["x"][3], an2))
print("22 longitudinal pure force [N]", f_fxp22(sol["x"][0], sol["x"][1], sol["x"][3], an2))
print("-----")
print("11 lateral combined force [N]", f_fy11(sol["x"][0], sol["x"][1], sol["x"][2], an2))
print("12 lateral combined force [N]", f_fy12(sol["x"][0], sol["x"][1], sol["x"][3], an2))
print("21 lateral combined force [N]", f_fy21(sol["x"][0], sol["x"][1], sol["x"][3], an2))
print("22 lateral combined force [N]", f_fy22(sol["x"][0], sol["x"][1], sol["x"][3], an2))
print("-----")
print("11 longitudinal combined force [N]", f_fx11(sol["x"][0], sol["x"][1], sol["x"][2], an2))
print("12 longitudinal combined force [N]", f_fx12(sol["x"][0], sol["x"][1], sol["x"][3], an2))
print("21 longitudinal combined force [N]", f_fx21(sol["x"][0], sol["x"][1], sol["x"][3], an2))
print("22 longitudinal combined force [N]", f_fx22(sol["x"][0], sol["x"][1], sol["x"][3], an2))
print("-----")
print("Yaw rate is [1/s]", f_yaw_rate(sol["x"][1], an2))
print("-----")
print("11 vertical load ", f_fz11(sol["x"][0], sol["x"][1], an2))
print("12 vertical load ", f_fz12(sol["x"][0], sol["x"][1], an2))
print("21 vertical load ", f_fz21(sol["x"][0], sol["x"][1], an2))
print("22 vertical load ", f_fz22(sol["x"][0], sol["x"][1], an2))
print("-----")
print("11 load coefficent ", f_load_11(sol["x"][0], sol["x"][1], an2))
print("12 load coefficent ", f_load_12(sol["x"][0], sol["x"][1], an2))
print("21 load coefficent ", f_load_21(sol["x"][0], sol["x"][1], an2))
print("22 load coefficent ", f_load_22(sol["x"][0], sol["x"][1], an2))
