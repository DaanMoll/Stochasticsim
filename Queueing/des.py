import numpy as np

if __name__ == "__main__":
    rho = 0.90
    m = 2
    p0 = 0
    mu = 1

    for k in range(m):
        p0 += (m*rho)**k / np.math.factorial(k) + (m*rho)**m / np.math.factorial(m) * 1/(1-rho)
 
    p0 = 1/p0
    print(p0)
    Mn = m * rho + rho * (((m * rho)**m) / np.math.factorial(m)) * p0/(1-rho)**2
    # print(Mn)

    M1 = rho / (1-rho)
    # print(M1)

    El = Mn
    # lamda 1 is 2x zo laag als lambda van M2
    Es = El / 2 

    print(p0*(1/1-rho) * 0.5)
    print(Es)

    