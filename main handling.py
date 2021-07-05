from scipy.optimize import fsolve, root
from math import *
import numpy as np
import matplotlib.pyplot as plt

track = 1.2  # [m] (same front and rear)
lf = 0.9  # front wheel base
lr = 0.7  # rear wheel base
m = 250  # vehicle mass
h = 0.3  # cog height

q1 = 0.04  # front roll center height [m]
q2 = 0.1  # rear roll center height [m]
qb = (lf * q1 + lr * q2) / (lf + lr)

k_susp_f = 25 * 1000  # Nm/rad (just spring and ARB, rigid tyres)
k_susp_r = 25 * 1000  # Nm/rad
# k_chassis = 100 * 1000  # Nm/rad

p_tire = 85 * 1000  # N/m (tyre vertical stiffness)
k_tyre_f = (p_tire * track ** 2) / 2  # Nm/rad (just tyres, rigid susp and ARB)
k_tyre_r = (p_tire * track ** 2) / 2

print("------")
print("tyre roll stiffness =", [k_tyre_f, k_tyre_r])

k_f = (k_susp_f * k_tyre_f) / (k_susp_f + k_tyre_f)  # Nm/rad (tyres and suspension)
k_r = (k_susp_r * k_tyre_r) / (k_susp_r + k_tyre_r)

print("susp axle stiffness =", [k_f, k_r])
k_roll = k_f + k_r

print("roll stiffness = ", [k_roll])

# pajecka coefficients
B = 12.27
C = 1.48
D = -1100
E = 0.07

# coefficients for fz dependency in the tyre model
load_A = 1.5
load_B = 900


def equation(vars, data):
    vy, yaw_rate, Y1, Y2 = vars
    delta = data[0]  # [rad]
    vx = data[1]  # [m/s]

    # imposing longitudinal slip = 0, we can retrieve the rolling speed
    wr_11 = ((vx - yaw_rate * track * 0.5) * np.cos(delta) + (vy + yaw_rate * lf * 0.5) * np.sin(delta))
    wr_12 = ((vx + yaw_rate * track * 0.5) * np.cos(delta) + (vy + yaw_rate * lf * 0.5) * np.sin(delta))
    wr_21 = (vx - yaw_rate * track * 0.5) * np.cos(delta)
    wr_22 = (vx + yaw_rate * track * 0.5) * np.cos(delta)

    slip_y_11 = ((vy + yaw_rate * lf) * np.cos(delta) - (vx - yaw_rate * track * 0.5) * np.sin(delta)) / wr_11
    slip_y_12 = ((vy + yaw_rate * lf) * np.cos(delta) - (vx + yaw_rate * track * 0.5) * np.sin(delta)) / wr_12
    slip_y_21 = (vy - yaw_rate * lr) / wr_21
    slip_y_22 = (vy - yaw_rate * lr) / wr_22

    Y = Y1 + Y2

    # Lateral load transfers
    Delta_Z_1 = (1 / track) * ((k_f / k_roll) * Y * (h - qb) + Y1 * q1 + (k_f * k_r / k_roll) * (Y2 * q2 / k_tyre_r - Y1 * q1 / k_tyre_f))
    Delta_Z_2 = (1 / track) * ((k_r / k_roll) * Y * (h - qb) + Y2 * q2 + (k_f * k_r / k_roll) * (Y1 * q1 / k_tyre_f - Y2 * q2 / k_tyre_r))

    fz_11 = 0.5 * (m * 9.81 * lr / (lr + lf)) - Delta_Z_1
    fz_12 = 0.5 * (m * 9.81 * lr / (lr + lf)) + Delta_Z_1
    fz_21 = 0.5 * (m * 9.81 * lf / (lr + lf)) - Delta_Z_2
    fz_22 = 0.5 * (m * 9.81 * lf / (lr + lf)) + Delta_Z_2

    load_11 = load_A * np.arctan(fz_11 / load_B)
    load_12 = load_A * np.arctan(fz_12 / load_B)
    load_21 = load_A * np.arctan(fz_21 / load_B)
    load_22 = load_A * np.arctan(fz_22 / load_B)

    Fy11 = load_11 * D * np.sin(C * np.arctan(B * slip_y_11 - E * (B * slip_y_11 - np.arctan(B * slip_y_11))))
    Fy12 = load_12 * D * np.sin(C * np.arctan(B * slip_y_12 - E * (B * slip_y_12 - np.arctan(B * slip_y_12))))
    Fy21 = load_21 * D * np.sin(C * np.arctan(B * slip_y_21 - E * (B * slip_y_21 - np.arctan(B * slip_y_21))))
    Fy22 = load_22 * D * np.sin(C * np.arctan(B * slip_y_22 - E * (B * slip_y_22 - np.arctan(B * slip_y_22))))

    DeltaX_1 = (Fy11 * np.sin(delta) - Fy12 * np.sin(delta)) / 2

    eq1 = m * vx * yaw_rate - Y1 - Y2
    eq2 = Y1 * lf - Y2 * lr + DeltaX_1 * track
    eq3 = Y1 - Fy11 * np.cos(delta) - Fy12 * np.cos(delta)
    eq4 = Y2 - Fy21 - Fy22

    return [eq1, eq2, eq3, eq4]


def results(vy, yaw_rate, Y1, Y2, data, plot_graph=True):
    delta = data[0]  # [rad]
    vx = data[1]  # [m/s]

    wr_11 = ((vx - yaw_rate * track * 0.5) * cos(delta) + (vy + yaw_rate * lf * 0.5) * sin(delta))
    wr_12 = ((vx + yaw_rate * track * 0.5) * cos(delta) + (vy + yaw_rate * lf * 0.5) * sin(delta))
    wr_21 = (vx - yaw_rate * track * 0.5) * cos(delta)
    wr_22 = (vx + yaw_rate * track * 0.5) * cos(delta)

    slip_y_11 = ((vy + yaw_rate * lf) * cos(delta) - (vx - yaw_rate * track * 0.5) * sin(delta)) / wr_11
    slip_y_12 = ((vy + yaw_rate * lf) * cos(delta) - (vx + yaw_rate * track * 0.5) * sin(delta)) / wr_12
    slip_y_21 = (vy - yaw_rate * lr) / wr_21
    slip_y_22 = (vy - yaw_rate * lr) / wr_22

    Y = Y1 + Y2
    Delta_Z_1 = (1 / track) * ((k_f / k_roll) * Y * (h - qb) + Y1 * q1 + (k_f * k_r / k_roll) * (Y2 * q2 / k_tyre_r - Y1 * q1 / k_tyre_f))
    Delta_Z_2 = (1 / track) * ((k_r / k_roll) * Y * (h - qb) + Y2 * q2 + (k_f * k_r / k_roll) * (Y1 * q1 / k_tyre_f - Y2 * q2 / k_tyre_r))

    fz_11 = 0.5 * (m * 9.81 * lr / (lr + lf)) - Delta_Z_1
    fz_12 = 0.5 * (m * 9.81 * lr / (lr + lf)) + Delta_Z_1
    fz_21 = 0.5 * (m * 9.81 * lf / (lr + lf)) - Delta_Z_2
    fz_22 = 0.5 * (m * 9.81 * lf / (lr + lf)) + Delta_Z_2

    load_11 = load_A * atan(fz_11 / load_B)
    load_12 = load_A * atan(fz_12 / load_B)
    load_21 = load_A * atan(fz_21 / load_B)
    load_22 = load_A * atan(fz_22 / load_B)

    Fy11 = load_11 * D * sin(C * atan(B * slip_y_11 - E * (B * slip_y_11 - atan(B * slip_y_11))))
    Fy12 = load_12 * D * sin(C * atan(B * slip_y_12 - E * (B * slip_y_12 - atan(B * slip_y_12))))
    Fy21 = load_21 * D * sin(C * atan(B * slip_y_21 - E * (B * slip_y_21 - atan(B * slip_y_21))))
    Fy22 = load_22 * D * sin(C * atan(B * slip_y_22 - E * (B * slip_y_22 - atan(B * slip_y_22))))

    beta = vy / vx
    ax = - vy * yaw_rate
    ay = vx * yaw_rate

    vel = sqrt(vx ** 2 + vy ** 2)
    an = -ax * sin(beta) + ay * cos(beta)
    at = ax * cos(beta) + ay * sin(beta)
    rho = yaw_rate / vx
    R = 1 / rho

    DeltaX_1 = (Fy11 * np.sin(delta) - Fy12 * np.sin(delta)) / 2

    # Perform a  series of checks
    check_eq1 = m * vx * yaw_rate - Y1 - Y2
    check_eq2 = Y1 * lf - Y2 * lr + DeltaX_1 * track
    check_Y1 = abs((Y1 - Fy11 * cos(delta) - Fy12 * cos(delta)) / Y1)
    check_Y2 = abs((Y2 - Fy21 - Fy22) / Y2)

    # static vertical laod
    Z0_f = 0.5 * (m * 9.81 * lr / (lr + lf))
    Z0_r = 0.5 * (m * 9.81 * lf / (lr + lf))

    results_dict = {
        "sol" : [vy, yaw_rate, Y1, Y2],
        "wr": [wr_11, wr_12, wr_21, wr_22],
        "slip_y": [slip_y_11, slip_y_12, slip_y_21, slip_y_22],
        "Delta_Z": [Delta_Z_1, Delta_Z_2],
        "fz": [fz_11, fz_12, fz_21, fz_22],
        "fz_static": [Z0_f, Z0_r],
        "lateral_load_transfer": [Delta_Z_1, Delta_Z_2],
        "load_coeff": [load_11, load_12, load_21, load_22],
        "fy": [Fy11, Fy12, Fy21, Fy22],
        "beta": beta,
        "axy": [ax, ay],
        "atn": [at, an],
        "speed": vel,
        "check_Y1": check_Y1,
        "check_Y2": check_Y2,
        "check_eq1": check_eq1,
        "check_eq2": check_eq2,
        "R": R,
        "rho": rho
    }

    if plot_graph:
        slip_plot = np.linspace(-0.4, 0.4, 1000)
        load_f_static = load_A * atan(Z0_f / load_B)
        load_r_static = load_A * atan(Z0_r / load_B)
        Fy_f_plot = load_f_static * D * np.sin(
            C * np.arctan(B * slip_plot - E * (B * slip_plot - np.arctan(B * slip_plot))))
        Fy_r_plot = load_r_static * D * np.sin(
            C * np.arctan(B * slip_plot - E * (B * slip_plot - np.arctan(B * slip_plot))))

        Fy_11_plot = load_11 * D * np.sin(C * np.arctan(B * slip_plot - E * (B * slip_plot - np.arctan(B * slip_plot))))
        Fy_12_plot = load_12 * D * np.sin(C * np.arctan(B * slip_plot - E * (B * slip_plot - np.arctan(B * slip_plot))))
        Fy_21_plot = load_21 * D * np.sin(C * np.arctan(B * slip_plot - E * (B * slip_plot - np.arctan(B * slip_plot))))
        Fy_22_plot = load_22 * D * np.sin(C * np.arctan(B * slip_plot - E * (B * slip_plot - np.arctan(B * slip_plot))))

        fig, axs = plt.subplots(2, 2, sharey=True)

        axs[0, 0].grid(True)
        axs[0, 0].plot(slip_plot, Fy_f_plot, "b--", label="Static")
        axs[0, 0].plot(slip_plot, Fy_11_plot, 'b-', label="Solved")
        axs[0, 0].plot(slip_y_11, Fy11, "cx")
        axs[0, 0].set_title('Fy 11')
        axs[0, 0].legend()
        # axs[0, 1].sharey(axs[0, 0])
        axs[0, 1].grid(True)
        axs[0, 1].plot(slip_plot, Fy_f_plot, "r--", label="Static")
        axs[0, 1].plot(slip_plot, Fy_12_plot, 'r-', label="Solved")
        axs[0, 1].plot(slip_y_12, Fy12, "cx")
        axs[0, 1].set_title('Fy 12')
        axs[0, 1].legend()
        # axs[1, 1].sharey(axs[1, 0])
        axs[1, 0].grid(True)
        axs[1, 0].plot(slip_plot, Fy_r_plot, "g--", label="Static")
        axs[1, 0].plot(slip_plot, Fy_21_plot, 'g-', label="Solved")
        axs[1, 0].plot(slip_y_21, Fy21, "cx")
        axs[1, 0].set_title('Fy 21')
        axs[1, 0].legend()
        axs[1, 1].grid(True)
        axs[1, 1].plot(slip_plot, Fy_r_plot, "k--", label="Static")
        axs[1, 1].plot(slip_plot, Fy_22_plot, 'k-', label="Solved")
        axs[1, 1].plot(slip_y_22, Fy22, "cx")
        axs[1, 1].set_title('Fy 22')
        axs[1, 1].legend()

        for ax in axs.flat:
            ax.set(xlabel='x-label', ylabel='y-label')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
        fig.show()

    return results_dict


def print_results(r_dict):
    print("------")
    print("v, r, y1, y2 =", r_dict["sol"])
    print("------")
    print("check on eq 1 (should be zero)", r_dict["check_eq1"])
    print("check on eq 2 (should be zero)", r_dict["check_eq2"])
    print("check on Y1 (eq3 normalized by Y1, should be zero)", r_dict["check_Y1"])
    print("check on Y2 (eq4 normalized by Y2, should be zero)", r_dict["check_Y2"])
    print("------")
    print("load_coeff", r_dict["load_coeff"])
    print("vertical load", r_dict["fz"])
    print("static vertical load", r_dict["fz_static"])
    print("lateral load transfer", r_dict["lateral_load_transfer"])
    print("------")
    print("lateral tire slip", r_dict["slip_y"])
    print("lateral tire force", r_dict["fy"])
    print("------")
    print("acceleration in xy", r_dict["axy"])
    print("acceleration in tn", r_dict["atn"])
    print("------")
    print("speed", r_dict["speed"])
    print("vehicle side slip angel", r_dict["beta"])


# inputs = [0.01, 12]  # [steering angle delta, vx]
# x0 = np.array([0, 0, 0, 0])
# [v, r, y1, y2] = fsolve(equation, x0, args=inputs)
# result_dict = results(v, r, y1, y2, inputs)
# print_results(result_dict)

velocity = [2, 5, 7, 8, 9, 10, 11, 12]
fig2, ax2 = plt.subplots()
for vel in velocity:
    x0 = np.array([0, 0, 0, 0])
    go_ahead = True
    steering_angle = 0.01
    steering_step = 0.01
    velocity = 10
    inputs = [steering_angle, vel]
    curvature = []
    steering = []

    while go_ahead:
        [v, r, y1, y2] = fsolve(equation, x0, args=inputs)
        x0 = np.array([v, r, y1, y2])
        inputs = [inputs[0] + steering_step, vel]
        if inputs[0] > 0.53:
            go_ahead = False
        result_dict = results(v, r, y1, y2, inputs, False)
        curvature.append(result_dict["rho"])
        steering.append(inputs[0])

    result_dict = results(v, r, y1, y2, inputs, False)
    print_results(result_dict)

    ax2.set_title("Road car rho vs delta")
    ax2.plot(steering, curvature)
plt.show()

