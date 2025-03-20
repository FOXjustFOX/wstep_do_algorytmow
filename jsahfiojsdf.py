import random
import numpy as np


def gowno():
    L = 3
    M = 2

    def macierzePQ():  # tworzenie macierzy
        P = [[7, 0],[7, 2],[6, 8]] # losowanie liczb
        Q = [[10, 7],[9, 3],[3, 6]]
        print("Macierz P:")
        for row in P:
            print(row)
        print("\nMacierz Q:")
        for row in Q:
            print(row)
        return P, Q

    def odl_syme(P, Q):  # D(P,Q)=  suma (i=1,L) * suma (i=1,M) * | p(i,j) - q(i,j) |
        d = 0
        for i in range(len(P)):
            for j in range(len(P[0])):
                d += abs(P[i][j] - Q[i][j])  # wartość bezwzgledna
        return d

    P, Q = macierzePQ()  # generowanie macierzy
    d = odl_syme(P, Q)  # obliczanie odległości symetrycznej
    print(d)


def moje():
    
    P = np.array([[7, 0],[7, 2],[6, 8]])
    Q = np.array([[10, 7],[9, 3],[3, 6]])

    suma = np.sum(np.abs(np.sum(P - Q, axis=1)))
    print(f"Odległość symetryczna między P a Q: {suma:.2f}")

gowno()
moje()