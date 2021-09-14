import math
import time
from typing import List

from scipy.optimize import minimize, NonlinearConstraint
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial import Delaunay


import numpy as np
import random


#точки
def rand_cord_cent(n, xm, ym, zm):
    ct = []
    for i in range(n):
        # ct.append([1.0,1.0,2.0])
        ct.append([random.uniform(0,xm),random.uniform(0,ym),random.uniform(0,zm)])
    return ct


#радиусы
def rand_rad(n, min, max):
    rd = []
    for i in range(n):
        rd.append(random.uniform(min, max))
    return rd

def v(sphere_radiuses):
    sum = 0
    t = sphere_radiuses
    for i in t:
        sum += (4/3) * 3.14 * (i ** 3)
    return sum


radiuses = [0.15, 0.11, 0.07]

def func1(xx)->np.ndarray:
    vol = 0.2
    side = 1.0
    kol: np.ndarray = np.zeros(3)
    for i in range(3):
        kol[i] = (xx[i]*vol)/(4/3 * math.pi * radiuses[i] ** 3)
    return kol


xx: np.ndarray = np.array([0.0, 0.4, 0.6])
kol = func1(xx)
kol1 = np.int16(kol)
print(kol1)

p_a: float = 1.0
p_h: float = 0.2
sphere_count: int = np.sum(kol1)
print(sphere_count)
var_count: int = 3 * sphere_count + 1
sphere_centures: np.ndarray = np.array(rand_cord_cent(sphere_count, p_a, p_a, p_h))
sphere_radiuses: np.ndarray = np.zeros(sphere_count)
for i in range(kol1[0]):
    sphere_radiuses[i] = radiuses[0]
for i in range(kol1[0], kol1[0]+kol1[1]):
    sphere_radiuses[i] = radiuses[1] 
for i in range(kol1[0]+kol1[1], sphere_count):
    sphere_radiuses[i] = radiuses[2]    

constrain_count: int =\
    sphere_count * 6 + sphere_count * (sphere_count - 1) // 2

print('__________________________________')
print('sphere_radiuses', sphere_radiuses)

def constr(x):
    c: List[float] = [0.0] * constrain_count
    constr_numb: int = 0
    for i in range(sphere_count):
        c[constr_numb] = x[3 * i] - sphere_radiuses[i]
        constr_numb += 1
        c[constr_numb] = - x[3 * i] - sphere_radiuses[i] + p_a
        constr_numb += 1
        c[constr_numb] = x[3 * i + 1] - sphere_radiuses[i]
        constr_numb += 1
        c[constr_numb] = - x[3 * i + 1] - sphere_radiuses[i] + p_a
        constr_numb += 1
        c[constr_numb] = x[3 * i + 2] - sphere_radiuses[i]
        constr_numb += 1
        c[constr_numb] = - x[3 * i + 2] - sphere_radiuses[i] + x[var_count - 1]
        constr_numb += 1
        for j in range(i+1, sphere_count):
            c[constr_numb] = (x[3 * i] - x[3 * j]) * (x[3 * i] - x[3 * j]) +\
                (x[3 * i + 1] - x[3 * j + 1]) * (x[3 * i + 1] - x[3 * j + 1])\
                + (x[3 * i + 2] - x[3 * j + 2]) * (x[3 * i + 2] - x[3 * j + 2])\
                - (sphere_radiuses[i] + sphere_radiuses[j]) * (sphere_radiuses[i] + sphere_radiuses[j])
            constr_numb += 1
    return c


def jacob(x):
    jac: np.ndarray = np.zeros((constrain_count, var_count))
    constr_numb: int = 0
    for i in range(sphere_count):
        jac[constr_numb, 3 * i] = 1
        constr_numb += 1
        jac[constr_numb, 3 * i] = -1
        constr_numb += 1
        jac[constr_numb, 3 * i + 1] = 1
        constr_numb += 1
        jac[constr_numb, 3 * i + 1] = -1
        constr_numb += 1
        jac[constr_numb, 3 * i + 2] = 1
        constr_numb += 1
        jac[constr_numb, 3 * i + 1] = -1
        jac[constr_numb, var_count - 1] = 1
        constr_numb += 1
        for j in range(i+1, sphere_count):
            jac[constr_numb, 3 * i] = 2 * (x[3 * i] - x[3 * j])
            jac[constr_numb, 3 * j] = - 2 * (x[3 * i] - x[3 * j])
            jac[constr_numb, 3 * i + 1] = 2 * (x[3 * i + 1] - x[3 * j + 1])
            jac[constr_numb, 3 * j + 1] = - 2 * (x[3 * i + 1] - x[3 * j + 1])
            jac[constr_numb, 3 * i + 2] = 2 * (x[3 * i + 2] - x[3 * j + 2])
            jac[constr_numb, 3 * j + 2] = - 2 * (x[3 * i + 2] - x[3 * j + 2])
            constr_numb += 1
    return jac

def ff(x) -> float:
    return x[var_count - 1]


def grad(x) -> np.ndarray:
    g: np.ndarray = np.zeros(var_count)
    g[var_count - 1] = 1
    return g


def dr_sphere(xi: float, yi: float, zi: float, rad: float, ax, col: str):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = (2*rad)*(np.cos(u)*np.sin(v)) + xi
    y = (2*rad)*(np.sin(u)*np.sin(v)) + yi
    z = (2*rad) * np.cos(v) + zi
    ax.plot_wireframe(x, y, z, color="g")
    return


def get_cube():
    phi = np.arange(1, 10, 2)*np.pi/4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi)*np.sin(Theta)
    y = np.sin(Phi)*np.sin(Theta)
    z = np.cos(Theta)/np.sqrt(2)
    return x, y, z


def dr_parall(a: float, ax, b: float):
    x, y, z = get_cube()
    ax.plot_surface(x*a+a/2, y*a+a/2, z*b+b/2, edgecolors="r", alpha=0.2)


def problem_vizualization(
        sphere_centures: np.ndarray,
        sphere_radiuses: np.ndarray,
        p_a: float, p_h: float):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    dr_parall(p_a, ax, p_h)
    for i in range(sphere_count):
        dr_sphere(
            sphere_centures[i, 0], sphere_centures[i, 1],
            sphere_centures[i, 2], sphere_radiuses[i] / 2, ax, "g")

    ax.set_xlim(0, p_a)
    ax.set_ylim(0, p_a)
    ax.set_zlim(0, p_h)

    plt.show()


if __name__ == "__main__":
    problem_vizualization(sphere_centures, sphere_radiuses, p_a, p_h)
    low: List[float] = [0.0] * constrain_count
    upper: List[float] = [float("inf")] * constrain_count
    constr_numb: int = 0
    x: np.ndarray = np.zeros(var_count)
    for i in range(sphere_count):
        x[3 * i] = sphere_centures[i, 0]
        x[3 * i + 1] = sphere_centures[i, 1]
        x[3 * i + 2] = sphere_centures[i, 2]
    x[var_count - 1] = p_h
    c = NonlinearConstraint(constr, low, upper)
    start_time: float = time.monotonic()
    res = minimize(ff, x, method='COBYLA', jac=grad, constraints=c)
    end_time: float = time.monotonic()
    print('time', end_time - start_time)

    print('res.x', res.x)
    for i in range(sphere_count):
        sphere_centures[i, 0] = res.x[3 * i]
        sphere_centures[i, 1] = res.x[3 * i + 1]
        sphere_centures[i, 2] = res.x[3 * i + 2]
    p_h = res.x[var_count - 1]
    print('///////////////////////////////////////')
    print(res.x[var_count - 1])
    vv = (res.x[var_count - 1]) * p_a * p_a
    tt = v(sphere_radiuses)
    vvv = (tt/vv) * 100
    print(vvv)
    problem_vizualization(sphere_centures, sphere_radiuses, p_a, p_h)
