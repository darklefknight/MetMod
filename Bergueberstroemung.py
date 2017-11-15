import numpy as np
from numpy import *
from scipy.misc import derivative
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def Z_s(x):
    """
    Funktion die einen glockenförmigen 2D Berg definiert. (Orographie)
    Z_bx: Länge des Berges in m
    Z_bz: Höhe des Berges in m
    :param x: x-achsen Abschnitt in m
    :return: Höhe in Abhängigkeit von x
    """
    Z = L_zb * (L_xb * L_xb) / ((L_xb * L_xb) + (x * x))
    return Z


def f_2(x):
    return -1 * (x / L_xb) * Z_s(x)


def Verschiebung(x, z):
    """
    Gleichung zur Berechnung der Verschiebung der Stromlinien.
    Diese Gleichung gilt es zu lösen.
    Abhängigkeiten: N,f_2,Z_s
    :param x: x-achsen Abschnitt in m
    :param z: x-achsen Abschnitt in m
    :return:
    """
    Klammer = (N() / U) * (z - Z_s(x))
    term1 = Z_s(x) * np.cos(Klammer)
    term2 = f_2(x) * np.sin(Klammer)
    result = term1 + term2
    return result


def N():
    """
    Konstante
    :return: Konstante
    """
    radikant = (g / Theta_0) * (Schichtung)
    result = np.sqrt(radikant)
    return result


# =========================
# Analytische Lösungen:
# =========================
def Theta(x, z):
    """
    Gleichung zur analytischen Lösung des Temperaturfeldes
    :param x: x-achsen Abschnitt in m
    :param z: x-achsen Abschnitt in m
    :return: Temperatur in K an der Stelle x,z
    """

    result = Theta_0 * (1 + (N() * N() * (z - Verschiebung(x, z)) / g))
    return result


def u(x, z):
    """
    Analytische Lösung des Horizontalwindes.
    Hierzu wird die Ableitungsfunktion von scipy benutzt.
    :param x: x-achsen Abschnitt in m
    :param z: x-achsen Abschnitt in m
    :return: Horizontalwind in m/s an Stelle x,z
    """
    der = derivative(Verschiebung(x, z), x0=z)
    result = U * (1 - der)
    return result


def w(x, z):
    """
    Analytische Lösung des Vertikalwindes.
    Hierzu wird die Ableitungsfunktion von scipy benutzt.
    :param x: x-achsen Abschnitt in m
    :param z: x-achsen Abschnitt in m
    :return: Vertikalwind in m/s an Stelle x,z
    """
    der = derivative(Verschiebung(x, z), x0=x)
    result = U * der
    return result


def w(x,z):
    dx = (- (2 * L_xb**3 * L_zb**2 * N() * x**2 * np.cos(N() / U * (z - Z_s(x)))) / (U * (L_xb**2 + x**2)**3)
          - (2 * L_xb**2 * L_zb *          x    * np.cos(N() / U * (z - Z_s(x)))) / (     L_xb**2 + x**2)**2
          - (2 * L_xb**4 * L_zb**2 * N() * x    * np.sin(N() / U * (z - Z_s(x)))) / (U * (L_xb**2 + x**2)**3)
          + (2 * L_xb    * L_zb          * x**2 * np.sin(N() / U * (z - Z_s(x)))) / (     L_xb**2 + x**2)**2
          - (    L_xb    * L_zb                 * np.sin(N() / U * (z - Z_s(x)))) / (     L_xb**2 + x**2))

    return U*dx


def u(x,z):
    dz = -Z_s(x) * np.sin( N()/U * (z-Z_s(x)) ) * N()/U - x/L_xb * Z_s(x) * np.cos( N()/U * (z-Z_s(x)) ) * N()/U
    return U*(1-dz)

if __name__ == "__main__":

    # Festlegen der Eingangsgrößen:
    U = 10  # m/s
    Schichtung = 0.05  # K/m
    T_s = 290  # K
    RH = 0  # %
    g = 9.81
    Theta_0 = T_s

    L_xb = 3000  # m
    L_zb = 300  # m

    # Gitter:
    x_grid_start = -50000
    x_grid_length = 50000
    z_grid_length = 10000
    x_step = 1000
    z_step = 100
    # Starte Berechnung
    verschiebungen = np.zeros([int((x_grid_length - x_grid_start) / x_step) + 1, int(z_grid_length / z_step)]).astype(
        float)
    Theta_feld = np.zeros([int((x_grid_length - x_grid_start) / x_step) + 1, int(z_grid_length / z_step)]).astype(float)
    w_feld = np.zeros([int((x_grid_length - x_grid_start) / x_step) + 1, int(z_grid_length / z_step)]).astype(float)
    u_feld = np.zeros([int((x_grid_length - x_grid_start) / x_step) + 1, int(z_grid_length / z_step)]).astype(float)

    Berg = np.zeros(int((x_grid_length - x_grid_start) / x_step) + 1).astype(float)

    x_counter = 0
    for x in range(x_grid_start, x_grid_length, x_step):

        Berg[x_counter] = Z_s(x)

        z_counter = 0

        for z in range(0, z_grid_length, z_step):
            Theta_feld[x_counter, z_counter] = Theta(x, z)
            u_feld[x_counter, z_counter] = u(x, z)
            w_feld[x_counter,z_counter] = w(x,z)
            verschiebungen[x_counter, z_counter] = Verschiebung(x, z)
            z_counter += 1
        x_counter += 1

    # print(np.shape(Berg))
    # print(np.shape(verschiebungen))
    # print(np.shape(np.arange(x_grid_start,x_grid_length,x_step)))

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(411)

    # plotte Stromlinienverschiebungen:
    sl_levels = np.arange(-300, 300, 10)
    cs = ax1.contourf(np.arange(x_grid_start, x_grid_length + 1, x_step), np.arange(0, z_grid_length, z_step),
                      verschiebungen.transpose(), cmap="bwr", levels=sl_levels)
    cb = fig.colorbar(cs, label="stremline displacement m")
    ax1.set_xlabel("width [m]")
    ax1.set_ylabel("height [m]")

    # plotte Theta-Feld
    ax2 = fig.add_subplot(412)
    levels = np.arange(280, np.nanmax(Theta_feld), 1)
    theta_plot = ax2.contourf(np.arange(x_grid_start, x_grid_length + 1, x_step), np.arange(0, z_grid_length, z_step),
                              Theta_feld.transpose(), cmap="jet", levels=levels)
    cb2 = fig.colorbar(theta_plot, label="Potential Temperature [K]")
    ax2.set_xlabel("width [m]")
    ax2.set_ylabel("height [m]")

    # plotte w-feld
    ax3 = fig.add_subplot(413)
    w_levels = np.arange(np.nanmin(w_feld), np.nanmax((w_feld)), 0.01)
    w_plot = ax3.contourf(np.arange(x_grid_start, x_grid_length + 1, x_step), np.arange(0, z_grid_length, z_step),
                              w_feld.transpose(), cmap="bwr",levels=w_levels)
    cb3 = fig.colorbar(w_plot, label="Vertikalwind [m/s]")
    ax3.set_xlabel("width [m]")
    ax3.set_ylabel("height [m]")

    # plotte u-feld
    ax4 = fig.add_subplot(414)
    u_levels = np.arange(np.nanmin(u_feld), np.nanmax(u_feld), 0.1)
    u_plot = ax4.contourf(np.arange(x_grid_start, x_grid_length + 1, x_step), np.arange(0, z_grid_length, z_step),
                              u_feld.transpose(), cmap="jet",levels=u_levels)
    cb4 = fig.colorbar(u_plot, label="Horizontalwind [m/s]")
    ax4.set_xlabel("width [m]")
    ax4.set_ylabel("height [m]")

    # plotte Berg:
    ax1.fill_between(np.arange(x_grid_start, x_grid_length + 1, x_step), 0, Berg, color="black")
    ax2.fill_between(np.arange(x_grid_start, x_grid_length + 1, x_step), 0, Berg, color="black")
