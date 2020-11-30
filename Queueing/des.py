import numpy as np

if __name__ == "__main__":
    c = 4
    rho = 0.95
    mu = 1

    PI = (c * rho)**c / np.math.factorial(c)

    summa = 0
    for n in range(0, c):
        summa += (c * rho)**n / np.math.factorial(n)

    summa = (1 - rho) * summa + (c * rho)**c / np.math.factorial(c)
    summa = summa**-1

    PI = PI * summa
    print("c:", c)
    print("delay prob:", PI)

    EW = PI * (1 / (1 - rho)) * 1 / (c*mu)

    print("E(W) =", EW)

    print("deel 2")

    a = 4.5
    b = 22.5

    c = (0.75 * a) + (0.25 * b)
    print("c:", c)



    