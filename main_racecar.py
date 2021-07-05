from scipy.optimize import fsolve, root
from math import *
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

track = 1.2  # [m] (same front and rear)
lf = 0.9  # front wheel base
lr = 0.7  # rear wheel base
m = 250  # vehicle mass
h = 0.3  # cog height

q1 = 0.04  # front roll center height [m]
q2 = 0.1  # rear roll center height [m]
qb = (lf * q1 + lr * q2) / (lf + lr)

aero_bal = 0.5
ksi_1 = 0.5 * 1.225 * 1.114 * 1.76 * aero_bal
ksi_2 = 0.5 * 1.225 * 1.114 * 1.76 * (1 - aero_bal)

k_susp_f = 25 * 1000  # Nm/rad (just spring and ARB, rigid tyres)
k_susp_r = 35 * 1000  # Nm/rad
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

# coefficients for combined tyre force
B_comb = 15
C_comb = 1.05
D_comb = 1

# rolling radius rear tyres
r2 = 0.25


def equation(vars, data):
    vx, vy, yaw_rate, Y1, Y2, delta_w = vars
    delta = data[0]  # [rad]
    w_h = data[1]  # [rad/s] (differential housing speed)

    w_21 = w_h - delta_w
    w_22 = w_h + delta_w

    # imposing longitudinal slip = 0, we can retrieve the rolling speed for the front (non driving) wheels
    wr_11 = ((vx - yaw_rate * track * 0.5) * np.cos(delta) + (vy + yaw_rate * lf * 0.5) * np.sin(delta))
    wr_12 = ((vx + yaw_rate * track * 0.5) * np.cos(delta) + (vy + yaw_rate * lf * 0.5) * np.sin(delta))
    slip_x_21 = (vx - yaw_rate * track * 0.5 - w_21 * r2) / (w_21 * r2)
    slip_x_22 = (vx + yaw_rate * track * 0.5 - w_22 * r2) / (w_22 * r2)

    slip_y_11 = ((vy + yaw_rate * lf) * np.cos(delta) - (vx - yaw_rate * track * 0.5) * np.sin(delta)) / wr_11
    slip_y_12 = ((vy + yaw_rate * lf) * np.cos(delta) - (vx + yaw_rate * track * 0.5) * np.sin(delta)) / wr_12
    slip_y_21 = (vy - yaw_rate * lr) / (w_21 * r2)
    slip_y_22 = (vy - yaw_rate * lr) / (w_22 * r2)

    Y = Y1 + Y2

    # Lateral load transfers
    Delta_Z_1 = (1 / track) * ((k_f / k_roll) * Y * (h - qb) + Y1 * q1 + (k_f * k_r / k_roll) * (
            Y2 * q2 / k_tyre_r - Y1 * q1 / k_tyre_f))
    Delta_Z_2 = (1 / track) * ((k_r / k_roll) * Y * (h - qb) + Y2 * q2 + (k_f * k_r / k_roll) * (
            Y1 * q1 / k_tyre_f - Y2 * q2 / k_tyre_r))

    fz_11 = 0.5 * (m * 9.81 * lr / (lr + lf) + ksi_1 * vx ** 2 - m * (- vy) * yaw_rate * h / (lf + lr)) - Delta_Z_1
    fz_12 = 0.5 * (m * 9.81 * lr / (lr + lf) + ksi_1 * vx ** 2 - m * (- vy) * yaw_rate * h / (lf + lr)) + Delta_Z_1
    fz_21 = 0.5 * (m * 9.81 * lf / (lr + lf) + ksi_2 * vx ** 2 + m * (- vy) * yaw_rate * h / (lf + lr)) - Delta_Z_2
    fz_22 = 0.5 * (m * 9.81 * lf / (lr + lf) + ksi_2 * vx ** 2 + m * (- vy) * yaw_rate * h / (lf + lr)) + Delta_Z_2

    load_11 = load_A * np.arctan(fz_11 / load_B)
    load_12 = load_A * np.arctan(fz_12 / load_B)
    load_21 = load_A * np.arctan(fz_21 / load_B)
    load_22 = load_A * np.arctan(fz_22 / load_B)

    G_21_y = D_comb * np.cos(C_comb * np.arctan(B_comb * slip_x_21))
    G_22_y = D_comb * np.cos(C_comb * np.arctan(B_comb * slip_x_22))

    G_21_x = D_comb * np.cos(C_comb * np.arctan(B_comb * slip_y_21))
    G_22_x = D_comb * np.cos(C_comb * np.arctan(B_comb * slip_y_22))

    Fx21 = G_21_x * load_21 * D * np.sin(C * np.arctan(B * slip_x_21 - E * (B * slip_x_21 - np.arctan(B * slip_x_21))))
    Fx22 = G_22_x * load_22 * D * np.sin(C * np.arctan(B * slip_x_22 - E * (B * slip_x_22 - np.arctan(B * slip_x_22))))
    # Fx22 = Fx21

    Fy11 = load_11 * D * np.sin(C * np.arctan(B * slip_y_11 - E * (B * slip_y_11 - np.arctan(B * slip_y_11))))
    Fy12 = load_12 * D * np.sin(C * np.arctan(B * slip_y_12 - E * (B * slip_y_12 - np.arctan(B * slip_y_12))))
    Fy21 = G_21_y * load_21 * D * np.sin(C * np.arctan(B * slip_y_21 - E * (B * slip_y_21 - np.arctan(B * slip_y_21))))
    Fy22 = G_22_y * load_22 * D * np.sin(C * np.arctan(B * slip_y_22 - E * (B * slip_y_22 - np.arctan(B * slip_y_22))))

    DeltaX_1 = (Fy11 * np.sin(delta) - Fy12 * np.sin(delta)) / 2

    X1 = - Fy11 * np.sin(delta) - Fy12 * np.sin(delta)
    X2 = Fx21 + Fx22
    Xa = 0.5 * 1.225 * 1.114 * 1.06 * vx ** 2

    eq1 = - m * vy * yaw_rate - X1 - X2 + Xa
    eq2 = m * vx * yaw_rate - Y1 - Y2
    eq3 = Y1 * lf - Y2 * lr + DeltaX_1 * track
    eq4 = Y1 - Fy11 * np.cos(delta) - Fy12 * np.cos(delta)
    eq5 = Y2 - Fy21 - Fy22
    eq6 = Fx21 - Fx22
    return [eq1, eq2, eq3, eq4, eq5, eq6]


def x0_model(vars, data):
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
    Delta_Z_1 = (1 / track) * ((k_f / k_roll) * Y * (h - qb) + Y1 * q1 + (k_f * k_r / k_roll) * (
                Y2 * q2 / k_tyre_r - Y1 * q1 / k_tyre_f))
    Delta_Z_2 = (1 / track) * ((k_r / k_roll) * Y * (h - qb) + Y2 * q2 + (k_f * k_r / k_roll) * (
                Y1 * q1 / k_tyre_f - Y2 * q2 / k_tyre_r))

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


def x0_racecar(data_in):  # input is steering delta and omega h (differential housing)
    data0 = [0, 0]
    data0[0] = data_in[0]
    data0[1] = data_in[1] * r2  # conversion from omega h to vx needed by x0_model
    x0_rc = np.array([0, 0, 0, 0])
    [vy0, r0, y10, y20] = fsolve(x0_model, x0_rc, args=data0)
    R = data0[1] / r0
    w022 = r0 * (R + track / 2) / r2
    w021 = r0 * (R - track / 2) / r2
    delta_w0 = (w022 - w021) / 2
    return np.array([data0[1], vy0, r0, y10, y20, 0.5 * delta_w0])


def results(vx, vy, yaw_rate, Y1, Y2, delta_w, data, plot_graph=True):
    delta = data[0]  # [rad]
    w_h = data[1]  # [rad/s]

    w_21 = w_h - delta_w
    w_22 = w_h + delta_w

    # imposing longitudinal slip = 0, we can retrieve the rolling speed for the front (non driving) wheels
    wr_11 = ((vx - yaw_rate * track * 0.5) * np.cos(delta) + (vy + yaw_rate * lf * 0.5) * np.sin(delta))
    wr_12 = ((vx + yaw_rate * track * 0.5) * np.cos(delta) + (vy + yaw_rate * lf * 0.5) * np.sin(delta))
    slip_x_21 = (vx - yaw_rate * track * 0.5 - w_21 * r2) / (w_21 * r2)
    slip_x_22 = (vx + yaw_rate * track * 0.5 - w_22 * r2) / (w_22 * r2)

    slip_y_11 = ((vy + yaw_rate * lf) * np.cos(delta) - (vx - yaw_rate * track * 0.5) * np.sin(delta)) / wr_11
    slip_y_12 = ((vy + yaw_rate * lf) * np.cos(delta) - (vx + yaw_rate * track * 0.5) * np.sin(delta)) / wr_12
    slip_y_21 = (vy - yaw_rate * lr) / (w_21 * r2)
    slip_y_22 = (vy - yaw_rate * lr) / (w_22 * r2)

    Y = Y1 + Y2

    # Lateral load transfers
    Delta_Z_1 = (1 / track) * ((k_f / k_roll) * Y * (h - qb) + Y1 * q1 + (k_f * k_r / k_roll) * (
            Y2 * q2 / k_tyre_r - Y1 * q1 / k_tyre_f))
    Delta_Z_2 = (1 / track) * ((k_r / k_roll) * Y * (h - qb) + Y2 * q2 + (k_f * k_r / k_roll) * (
            Y1 * q1 / k_tyre_f - Y2 * q2 / k_tyre_r))

    fz_11 = 0.5 * (m * 9.81 * lr / (lr + lf) + ksi_1 * vx ** 2 - m * (- vy) * yaw_rate * h / (lf + lr)) - Delta_Z_1
    fz_12 = 0.5 * (m * 9.81 * lr / (lr + lf) + ksi_1 * vx ** 2 - m * (- vy) * yaw_rate * h / (lf + lr)) + Delta_Z_1
    fz_21 = 0.5 * (m * 9.81 * lf / (lr + lf) + ksi_2 * vx ** 2 + m * (- vy) * yaw_rate * h / (lf + lr)) - Delta_Z_2
    fz_22 = 0.5 * (m * 9.81 * lf / (lr + lf) + ksi_2 * vx ** 2 + m * (- vy) * yaw_rate * h / (lf + lr)) + Delta_Z_2

    load_11 = load_A * np.arctan(fz_11 / load_B)
    load_12 = load_A * np.arctan(fz_12 / load_B)
    load_21 = load_A * np.arctan(fz_21 / load_B)
    load_22 = load_A * np.arctan(fz_22 / load_B)

    G_21_y = D_comb * np.cos(C_comb * np.arctan(B_comb * slip_x_21))
    G_22_y = D_comb * np.cos(C_comb * np.arctan(B_comb * slip_x_22))

    G_21_x = D_comb * np.cos(C_comb * np.arctan(B_comb * slip_y_21))
    G_22_x = D_comb * np.cos(C_comb * np.arctan(B_comb * slip_y_22))

    Fx21 = G_21_x * load_21 * D * np.sin(C * np.arctan(B * slip_x_21 - E * (B * slip_x_21 - np.arctan(B * slip_x_21))))
    Fx22 = G_22_x * load_22 * D * np.sin(C * np.arctan(B * slip_x_22 - E * (B * slip_x_22 - np.arctan(B * slip_x_22))))
    # Fx22 = Fx21

    Fy11 = load_11 * D * np.sin(C * np.arctan(B * slip_y_11 - E * (B * slip_y_11 - np.arctan(B * slip_y_11))))
    Fy12 = load_12 * D * np.sin(C * np.arctan(B * slip_y_12 - E * (B * slip_y_12 - np.arctan(B * slip_y_12))))
    Fy21 = G_21_y * load_21 * D * np.sin(C * np.arctan(B * slip_y_21 - E * (B * slip_y_21 - np.arctan(B * slip_y_21))))
    Fy22 = G_22_y * load_22 * D * np.sin(C * np.arctan(B * slip_y_22 - E * (B * slip_y_22 - np.arctan(B * slip_y_22))))

    DeltaX_1 = (Fy11 * np.sin(delta) - Fy12 * np.sin(delta)) / 2

    X1 = - Fy11 * np.sin(delta) - Fy12 * np.sin(delta)
    X2 = Fx21 + Fx22
    Xa = 0.5 * 1.225 * 1.114 * 1.06 * vx ** 2

    beta = vy / vx
    ax = - vy * yaw_rate
    ay = vx * yaw_rate

    vel = sqrt(vx ** 2 + vy ** 2)
    an = -ax * sin(beta) + ay * cos(beta)
    at = ax * cos(beta) + ay * sin(beta)
    rho = yaw_rate / vx
    R = 1 / rho

    # Perform a  series of checks
    check_eq1 = - m * vy * yaw_rate - X1 - X2 + Xa
    check_eq2 = m * vx * yaw_rate - Y1 - Y2
    check_eq3 = Y1 * lf - Y2 * lr + DeltaX_1 * track
    check_Y1 = abs((Y1 - Fy11 * cos(delta) - Fy12 * cos(delta)) / Y1)
    check_Y2 = abs((Y2 - Fy21 - Fy22) / Y2)
    checkX = abs((Fx21 - Fx22) / (Fx21 + Fx22))

    # static vertical laod
    Z0_f = 0.5 * (m * 9.81 * lr / (lr + lf))
    Z0_r = 0.5 * (m * 9.81 * lf / (lr + lf))

    results_dict = {
        "sol": [vx, vy, yaw_rate, Y1, Y2, delta_w],
        "wr": [wr_11, wr_12, w_21 * r2, w_22 * r2],
        "w": [w_21, w_22],
        "wh": [w_h],
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
        "check_eq3": check_eq3,
        "checkX": checkX,
        "R": R,
        "rho": rho,
        "fx": [Fx21, Fx22],
        "slip_x": [slip_x_21, slip_x_22],
        "Gx": [G_21_x, G_22_x],
        "Gy": [G_21_y, G_22_y],
        "downforce": [ksi_1 * vx ** 2, ksi_2 * vx ** 2],
        "drag": Xa
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
        Fy_21_plot = G_21_y * load_21 * D * np.sin(
            C * np.arctan(B * slip_plot - E * (B * slip_plot - np.arctan(B * slip_plot))))
        Fy_22_plot = G_22_y * load_22 * D * np.sin(
            C * np.arctan(B * slip_plot - E * (B * slip_plot - np.arctan(B * slip_plot))))

        fig, axs = plt.subplots(2, 2, sharex="all", sharey="all")

        axs[0, 0].grid(True)
        axs[0, 0].plot(slip_plot, Fy_f_plot, "b--", label="Static")
        axs[0, 0].plot(slip_plot, Fy_11_plot, 'b-', label="Solved")
        axs[0, 0].plot(slip_y_11, Fy11, "cx")
        axs[0, 0].set_title('Fy 11')
        axs[0, 0].legend()

        axs[0, 0].annotate("input: \nsteering angle [rad] " + str(delta) + "\ndiff house speed [rad/s] " + str(w_h),
                           xy=(-0.4, 0),
                           xytext=(-0.39, -1500))

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
        axs[1, 0].set_title('Fy 21 combined')
        axs[1, 0].legend()
        axs[1, 1].grid(True)
        axs[1, 1].plot(slip_plot, Fy_r_plot, "k--", label="Static")
        axs[1, 1].plot(slip_plot, Fy_22_plot, 'k-', label="Solved")
        axs[1, 1].plot(slip_y_22, Fy22, "cx")
        axs[1, 1].set_title('Fy 22 combined')
        axs[1, 1].legend()

        for ax in axs.flat:
            ax.set(xlabel='Theoretical lateral slip', ylabel='Lateral force Fy')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
        fig.show()

        fig2, axs2 = plt.subplots(1, 2, sharex="all", sharey="all")

        axs2[0].grid(True)
        axs2[0].plot([0, Fy21], [0, Fx21], "b-", label="Solved")
        # axs2[0].plot([0, slip_y_21 * 100], [0, slip_x_21 * 100], "r-", label="Solved")
        axs2[0].set_title('Fx 21 + Fy 21')
        axs2[0].legend()

        axs2[1].grid(True)
        axs2[1].plot([0, Fy22], [0, Fx22], "b-", label="Force Solved")
        # axs2[0].plot([0, slip_y_22 * 100], [0, slip_x_22 * 100], "r-", label="Slip Solved")
        axs2[1].set_title('Fx 22 + Fy 22')
        axs2[1].legend()

        for ax in axs2:
            xabs_max = abs(max(ax.get_xlim(), key=abs))
            yabs_max = abs(max(ax.get_ylim(), key=abs))
            ax.set_xlim(xmin=-xabs_max * 1.25, xmax=xabs_max * 1.25)
            ax.set_ylim(ymin=-yabs_max * 1.25, ymax=yabs_max * 1.25)
            ax.invert_xaxis()
            # ax.invert_yaxis()

        for ax in axs2.flat:
            ax.set(xlabel='Fx', ylabel='Fy')

        for ax in axs2.flat:
            ax.label_outer()

        fig2.show()

    return results_dict


def print_results(r_dict):
    print("------")
    print("u, v, r, y1, y2, delta_w =", r_dict["sol"])
    print("------")
    print("check on eq 1 (should be zero)", r_dict["check_eq1"])
    print("check on eq 2 (should be zero)", r_dict["check_eq2"])
    print("check on eq 3 (should be zero)", r_dict["check_eq3"])
    print("check on Y1 (eq3 normalized by Y1, should be zero)", r_dict["check_Y1"])
    print("check on Y2 (eq4 normalized by Y2, should be zero)", r_dict["check_Y2"])
    print("check on Fx21=Fx22 normalized by Fx21+Fx22 (should be zero)", r_dict["checkX"])
    print("------")
    print("load_coeff", r_dict["load_coeff"])
    print("vertical load", r_dict["fz"])
    print("static vertical load", r_dict["fz_static"])
    print("lateral load transfer", r_dict["lateral_load_transfer"])
    print("------")
    print("lateral tire slip", r_dict["slip_y"])
    print("lateral tire force", r_dict["fy"])
    print("------")
    print("slip x", r_dict["slip_x"])
    print("longitudinal tire force", r_dict["fx"])
    print("------")
    print("combined factor for fx", r_dict["Gx"])
    print("combined factor for fy", r_dict["Gy"])
    print("------")
    print("w_r", r_dict["wr"])
    print("w21, w22", r_dict["w"])
    print("w_h", result_dict["wh"])
    print("------")
    print("acceleration in xy", r_dict["axy"])
    print("acceleration in tn", r_dict["atn"])
    print("------")
    print("speed", r_dict["speed"])
    print("vehicle side slip angel", r_dict["beta"])
    print("------")
    print("downforce", r_dict["downforce"])
    print("drag", r_dict["drag"])


inputs = [0.3, 30]  # [steering angle delta, differential housing omega]
x0 = x0_racecar(inputs)
print("------")
print("x0 (vx, vy, r, y1, y2, delta_w)", x0)
[u, v, r, y1, y2, delta_omega] = fsolve(equation, x0, args=inputs)
result_dict = results(u, v, r, y1, y2, delta_omega, inputs)
print_results(result_dict)
print("------")
print([u, v, r, y1, y2, delta_omega])

# velocity = [8, 20, 28, 32, 36, 40, 44, 48]
# # velocity = [28, 32]
# fig2, ax2 = plt.subplots()
# tb = PrettyTable()
# for vel in velocity:
#     go_ahead = True
#     steering_angle = 0.01
#     steering_step = 0.01
#     inputs = [steering_angle, vel]
#     curvature = []
#     steering = []
#     x0 = x0_racecar(inputs)
#     header_line = [str(vel) + " [rad/s]", "Fy11 [N]", "Fy12 [N]", "Fy21 [N]", "Fy22 [N]", "Fx21 [N]", "Fx22 [N]"]
#     tb.add_row(header_line)
#     spacer = []
#     for cat in header_line:
#         spacer.append("--------")
#     tb.add_row(spacer)
#     while go_ahead:
#         [u, v, r, y1, y2, delta_omega] = fsolve(equation, x0, args=inputs)
#         x0 = np.array([u, v, r, y1, y2, delta_omega])
#         inputs = [inputs[0] + steering_step, vel]
#         if inputs[0] > 0.5:
#             go_ahead = False
#         result_dict = results(u, v, r, y1, y2, delta_omega, inputs, False)
#         curvature.append(result_dict["rho"])
#         fy_r = (result_dict["fy"])
#         fx_r = (result_dict["fx"])
#         steering.append(inputs[0])
#         row = [str(round(inputs[0], 3)) + " [rad]", str(round(fy_r[0], 3)), str(round(fy_r[1], 3)), str(round(fy_r[2], 3)), str(round(fy_r[3], 3)),
#                str(round(fx_r[0], 3)), str(round(fx_r[1], 3))]
#         tb.add_row(row)
#
#     tb.add_row(spacer)
#
#     result_dict = results(u, v, r, y1, y2, delta_omega, inputs, False)
#     # print_results(result_dict)
#     ax2.set_title("Race car rho vs delta")
#     ax2.plot(steering, curvature, label=str(inputs[1]))
#     ax2.set(xlabel='Steering angle [Rad]', ylabel='Curvature [1/m]')
#     ax2.legend(title='Differential \nhousing \nspeed [rad/s]')
#
# plt.show()
# print(tb)
