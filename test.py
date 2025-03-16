import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Zadanie 1: Analiza ocen studentów
# -----------------------------
def task1_student_grades(m, n):
    """
    Losuje macierz ocen o rozmiarze m x n z przedziału 2.0-5.5.
    Wiersze reprezentują studentów, kolumny – przedmioty.
    
    Obliczenia:
      - Liczba studentów, którzy nie zaliczyli (ocena < 3.0) wszystkich przedmiotów.
      - Student z najniższą oraz najwyższą średnią.
      - Student (lub studenci) z największą liczbą ocen równych jego najwyższej ocenie.
      - Dla każdego przedmiotu histogram ocen (używając numpy.histogram oraz wykresu).
      - Lista studentów ze średnią >= 4.5.
    """
    # Losujemy macierz ocen; zaokrąglamy do jednego miejsca po przecinku
    grades = np.random.uniform(2.0, 5.5, (m, n))
    grades = np.round(grades, 1)
    print("Macierz ocen (studenci x przedmioty):")
    print(grades)
    
    # a) Liczba studentów, którzy nie zaliczyli wszystkich przedmiotów (próg zaliczenia: 3.0)
    fail_mask = grades < 3.0
    fail_counts = np.sum(fail_mask, axis=1)
    # Jeśli student nie zdał danego przedmiotu, to jego ocena < 3.0; 
    # sprawdzamy, czy liczba przedmiotów niezaliczonych jest równa liczbie przedmiotów (czyli wszystkich)
    count_fail_all = np.sum(fail_counts >= n)
    print(f"\nLiczba studentów, którzy nie zaliczyli WSZYSTKICH przedmiotów (ocena < 3.0): {count_fail_all}")
    
    # b) Średnie ocen dla studentów oraz wyznaczenie najniższej i najwyższej średniej
    averages = np.mean(grades, axis=1)
    lowest_avg_index = np.argmin(averages)
    highest_avg_index = np.argmax(averages)
    print(f"\nStudent z najniższą średnią (indeks {lowest_avg_index}): oceny {grades[lowest_avg_index]}, średnia = {averages[lowest_avg_index]:.2f}")
    print(f"Student z najwyższą średnią (indeks {highest_avg_index}): oceny {grades[highest_avg_index]}, średnia = {averages[highest_avg_index]:.2f}")
    
    # c) Student z największą liczbą ocen najwyższych
    # Dla każdego studenta najpierw wyznaczamy jego maksymalną ocenę, a następnie liczbę wystąpień tej oceny.
    count_max_grades = [np.sum(row == np.max(row)) for row in grades]
    max_count = np.max(count_max_grades)
    students_with_max = np.where(np.array(count_max_grades) == max_count)[0]
    print(f"\nStudent(y) z największą liczbą ocen równych swojej najwyższej ocenie (liczba ocen: {max_count}): indeksy {students_with_max}")
    
    # d) Histogramy ocen dla poszczególnych przedmiotów
    print("\nHistogramy ocen dla poszczególnych przedmiotów:")
    for j in range(n):
        hist, bin_edges = np.histogram(grades[:, j], bins='auto')
        print(f"Przedmiot {j}: histogram: {hist}, krawędzie: {bin_edges}")
        plt.figure()
        plt.hist(grades[:, j], bins='auto', edgecolor='black')
        plt.title(f'Histogram ocen dla przedmiotu {j}')
        plt.xlabel('Ocena')
        plt.ylabel('Liczba studentów')
        plt.show()
    
    # e) Lista studentów ze średnią nie niższą niż 4.5
    students_avg_ge_45 = np.where(averages >= 4.5)[0]
    print(f"\nLista studentów (indeksy) ze średnią >= 4.5: {students_avg_ge_45}")
    return grades

# -----------------------------
# Zadanie 2: Odległość symetryczna dwóch macierzy
# -----------------------------
def task2_symmetric_distance(L, M):
    """
    Losuje dwie macierze P i Q o wymiarach L x M.
    Oblicza odległość symetryczną zdefiniowaną jako:
      δ_RS(P, Q) = Σ (dla i=1 do L) | Σ (dla j=1 do M) (p[i,j] - q[i,j]) |
    """
    P = np.random.uniform(0, 10, (L, M))
    Q = np.random.uniform(0, 10, (L, M))
    print("Macierz P:")
    print(P)
    print("\nMacierz Q:")
    print(Q)
    
    # Obliczamy różnicę w wierszach, sumujemy różnice dla każdej kolumny, bierzemy wartość bezwzględną,
    # a następnie sumujemy te wartości dla wszystkich wierszy.
    delta_RS = np.sum(np.abs(np.sum(P - Q, axis=1)))
    print(f"\nOdległość symetryczna między macierzami: {delta_RS:.2f}")
    return delta_RS

# -----------------------------
# Zadanie 3: Postać schodkowa zredukowana (Gauss-Jordan)
# -----------------------------
def gauss_jordan(matrix):
    """
    Funkcja wykonuje eliminację Gaussa z wyborem elementu głównego
    i sprowadza macierz do postaci schodkowej zredukowanej (RREF).
    """
    A = matrix.astype(float).copy()
    rows, cols = A.shape
    r = 0  # bieżący indeks wiersza
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
        # Zeruj elementy w kolumnie c w pozostałych wierszach
        for i in range(rows):
            if i != r:
                A[i] = A[i] - A[i, c] * A[r]
        r += 1
        if r == rows:
            break
    return A

def task3_reduced_row_echelon(n):
    """
    Losuje macierz rozszerzoną o rozmiarze n x (n+1) (dla n >= 2)
    i sprowadza ją do postaci schodkowej zredukowanej.
    """
    matrix = np.random.randint(-10, 10, (n, n+1)).astype(float)
    print("Macierz początkowa (układ równań):")
    print(matrix)
    rref_matrix = gauss_jordan(matrix)
    print("\nMacierz w postaci schodkowej zredukowanej:")
    print(rref_matrix)
    return rref_matrix

# -----------------------------
# Zadanie 4: Analiza paragonów w sklepie
# -----------------------------
def task4_receipts():
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
    
    # Sprawdzenie błędów:
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
    
    # Obliczenie łącznego kosztu paragonów dla poszczególnych klientów.
    # Tworzymy słownik mapujący numer produktu na (cenę, typ)
    product_dict = {row[0]: (row[1], row[2]) for row in products}
    total_cost = {}
    for receipt in receipts:
        cust, prod, qty = receipt
        price, type_sale = product_dict[prod]
        cost = qty * price
        total_cost[cust] = total_cost.get(cust, 0) + cost
    
    print("\nŁączny koszt paragonów dla poszczególnych klientów:")
    for cust, cost in total_cost.items():
        print(f"Klient {int(cust)}: {cost:.2f}")
    
    return receipts, products, total_cost

# -----------------------------
# Funkcja główna: wywołanie poszczególnych zadań
# -----------------------------
def main():
    np.random.seed(42)  # Ustawienie ziarna dla powtarzalności wyników
    
    print("==== Zadanie 1: Analiza ocen studentów ====")
    task1_student_grades(m=10, n=5)  # przykładowo 10 studentów, 5 przedmiotów
    
    print("\n==== Zadanie 2: Odległość symetryczna macierzy ====")
    task2_symmetric_distance(L=4, M=6)  # przykładowe wymiary L=4, M=6
    
    print("\n==== Zadanie 3: Eliminacja Gaussa (postać schodkowa zredukowana) ====")
    task3_reduced_row_echelon(n=3)  # przykładowo n=3
    
    print("\n==== Zadanie 4: Analiza paragonów w sklepie ====")
    task4_receipts()

if __name__ == "__main__":
    main()