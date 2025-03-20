import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(48) # dla powtarzalnosci wynikow

def zad_1(m, n):
    # Losujemy macierz ocen; zaokrąglamy do jednego miejsca po przecinku
    lista_ocen = [2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    
    oceny = np.random.choice(lista_ocen, (m, n))
    print("Macierz ocen (studenci x przedmioty):")
    print(oceny)
    
    # a) liczba studentów którzy nie zaliczyli wszystkich przedmiotów (<3.0)
    prog = oceny < 3.0
    l_studentow = np.sum(prog, axis=1)
    nie_zaliczyli = np.sum(l_studentow >= n)
    print(f"\nliczba studentów którzy nie zaliczyli wszystkich przedmiotów : {nie_zaliczyli}")
    
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
      
    For each row i, calculate the sum of differences (p[i,j] - q[i,j]) across all columns j
    Take the absolute value of that row sum
    Sum these absolute row sums across all rows i
    """
    
    P = np.random.uniform(0, 10, (L, M))  
    Q = np.random.uniform(0, 10, (L, M))
    
    suma = np.sum(np.abs(np.sum(P - Q, axis=1)))
    
    # to samo co wyzej ale np to lliczy bez petli
    
    # suma = 0
    # for i in range(P.shape[0]):
    #     suma += np.sum(np.abs(P[i] - Q[i]))
    
    print(f"Odległość symetryczna między P a Q: {suma:.2f}")
    
    # print(np.sum(P, axis=1))
    # print(np.sum(Q, axis=1))
    # print(np.sum(P - Q, axis=1))
    # print(np.sum(np.sum(P - Q, axis=1)))
    
# zad_2(3, 5)

# -----------------------------

# Zadanie 3: Postać schodkowa zredukowana (Gauss-Jordan)
# -----------------------------

def gauss(mx):
    A = mx.astype(float).copy()
    rows, cols = A.shape
    
    r = 0 # bieżący indeks wiersza, zaczynay od pierwszego
    
    for c in range(cols):
        # Znajdź pivot w kolumnie c (od wiersza r w dół)
        pivot = None
        for i in range(r, rows):
            if A[i, c] != 0:
                pivot = i
                break
        if pivot is None:
            continue    
        # Zamień bieżący wiersz z wierszem z pivotem
        A[[r, pivot]] = A[[pivot, r]]
        # Normalizuj wiersz z pivotem
        A[r] = A[r] / A[r, c]
        
        for i in range(rows):
            if i != r:
                A[i] = A[i] - A[i, c] * A[r]
        
        r += 1
        if r == rows:
            break
    return A

def zad_3(n):
    
    matrix = np.random.randint(-10, 10, (n, n+1)).astype(int)
    print("Macierz początkowa (układ równań):")
    print(matrix)
    rref_matrix = gauss(matrix)
    print("\nMacierz w postaci schodkowej zredukowanej:")
    print(rref_matrix)
    
# zad_3(3)

def zad_4():
    """
    Dwie macierze:
      - Paragony: kolumny: [numer klienta, numer towaru, ilość (sztuki lub waga)]
      - Produkty: kolumny: [numer towaru, cena jednostkowa (lub za kg), typ sprzedaży]
        (typ: 0 - sprzedawany na sztuki, 1 - sprzedawany na wagę)
    
    Sprawdzane:
      - Czy numery towarów z paragonów istnieją w macierzy produktów.
      - Dla towarów sprzedawanych na sztuki sprawdzamy, czy ilość jest liczbą całkowitą.
    
    Obliczenia:
      - Łączny koszt paragonów dla każdego klienta.
    """
    # Przykładowa macierz paragonów: [klient, produkt, ilość]
    receipts = np.array([
        [1, 101, 2],
        [1, 102, 3.5],
        [2, 101, 1],
        [2, 103, 2],
        [3, 104, 5]
    ], dtype=float)
    
    # Przykładowa macierz produktów: [produkt, cena, typ sprzedaży (0: sztuki, 1: waga)]
    products = np.array([
        [101, 10.0, 0],
        [102, 20.0, 1],
        [103, 15.0, 0],
        [104, 8.0, 0]
    ], dtype=float)
    
    print("Paragony:")
    print(receipts)
    print("\nProdukty:")
    print(products)
    
    product_ids = products[:, 0]
    for receipt in receipts:
        cust, prod, qty = receipt
        if prod not in product_ids:
            print(f"Błąd: produkt {prod} nie istnieje w macierzy produktów.")
        else:
            # Pobieramy informacje o produkcie
            product_info = products[products[:, 0] == prod]
            type_sale = product_info[0, 2]
            # Jeśli produkt sprzedawany na sztuki, ilość powinna być całkowita
            if type_sale == 0 and not np.isclose(qty, round(qty)):
                print(f"Błąd: produkt {prod} sprzedawany na sztuki, a ilość {qty} nie jest liczbą całkowitą.")
    
    
    # Obliczenie łącznego kosztu paragonów dla każdego klienta
    
    total_costs = {}
    for receipt in receipts:
        cust, prod, qty = receipt
        product_info = products[products[:, 0] == prod]
        price = product_info[0, 1]
        cost = float(qty * price)
        total_costs[int(cust)] = total_costs.get(int(cust), 0) + cost
        
    print("\nŁączny koszt paragonów dla każdego klienta:")
    for i in total_costs:
        print(f"Klient {i}: {total_costs[i]:.2f}")

    print(np.max(receipts))
 
# zad_4()