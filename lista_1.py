import numpy as np
import matplotlib.pyplot as plt

def zad_1(m, n):
    # Losujemy macierz ocen; zaokrąglamy do jednego miejsca po przecinku
    lista_ocen = [2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    
    oceny = np.random.choice(lista_ocen, (m, n))
    print("Macierz ocen (studenci x przedmioty):")
    print(oceny)
    
    # a) liczba studentów którzy nie zaliczyli wszystkich przedmiotów (<3.0)
    prog = oceny < 3.0
    l_studentow = np.sum(prog, axis=1)
    lie_zaliczyli = np.sum(l_studentow >= n)
    print(f"\nliczba studentów którzy nie zaliczyli wszystkich przedmiotów : {lie_zaliczyli}")
    
    # b) srednie ocen oraz max i min
    srednie = np.mean(oceny, axis=1)
    
    min_index = np.argmin(srednie)
    max_index = np.argmax(srednie)
    print(f"\nStudent z najnizsza srednia (indeks {min_index}): oceny {oceny[min_index]}, \nsrednia = {srednie[min_index]:.2f}, \nStudent z najwyzsza srednia (indeks {max_index}): oceny {oceny[max_index]}, \nsrednia = {srednie[max_index]:.2f}")
    
    
    # c) Student z największą liczbą ocen najwyższych
    max_ocena = (np.max(oceny))
    
    l_max_u_stud = [np.where(row == max_ocena)[0].shape[0] for row in oceny]
    max_count = np.max(l_max_u_stud)
    studenci_z_max = np.where(np.array(l_max_u_stud) == max_count)[0]
    print(f"\nStudent(ci) z największą liczbą ocen równych najwyższej ocenie (liczba ocen: {max_count}): indeksy {studenci_z_max + 1}") # +1 bo indeksy od 0
    
    # d) Histogramy ocen dla poszczególnych przedmiotów
    print("\nHistogramy ocen dla poszczególnych przedmiotów:")
    for j in range(n):
        # Define fixed bins for grades (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5)
        bins = np.arange(2.0, 6.0, 0.5)
        hist, _ = np.histogram(oceny[:, j], bins=bins) # ile poszczegolnych ocen, w przedziale
            
        print(f"\nPrzedmiot {j+1}:")
        print(f"{'Ocena'} | {'L studentów'} | {'Histogram'}")
        print("-" * 60)
        
        for i in range(len(hist)): # drukuje tez oceny ktore nie wystepuja
            grade = bins[i]
            count = hist[i]
            bar = "#" * count  # Create ASCII bar
            print(f"{grade} | {count} | {bar}")
            
            
    srednie_ponad_4_5 = np.where(srednie >= 4.5)[0]
    print(f"\nLista studentów ze średnią >= 4.5: {srednie_ponad_4_5 + 1}") # +1 bo indeksy od 0
    
# zad_1(7, 7)

# -----------------------------
# Zadanie 2: Wyznaczanie odległości symetrycznej między dwoma macierzami
# -----------------------------

def zad_2(L,M):
    """
    Oblicza odległość symetryczną między dwiema macierzami P i Q:
      δ_RS(P, Q) = Σ (dla i=1 do L) | Σ (dla j=1 do M) (p[i,j] - q[i,j]) |
    """
    P = np.random.uniform(0, 10, (L, M))  
    Q = np.random.uniform(0, 10, (L, M))
    
    suma = np.sum(np.abs(np.sum(P - Q, axis=1)))
    
    # to samo co wyzej ale np to lliczy bez petli
    
    # suma = 0
    # for i in range(P.shape[0]):
    #     suma += np.sum(np.abs(P[i] - Q[i]))
    
    print(f"Odległość symetryczna między P a Q: {suma:.2f}")
    
zad_2(3, 3)