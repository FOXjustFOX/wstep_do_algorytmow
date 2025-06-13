import collections
import unicodedata

# =============================================================================
# --- ZADANIE 1: ODLEGŁOŚĆ HAMMINGA I JEJ MODYFIKACJE ---
# =============================================================================

# --- 1a) Odległość Hamminga ---


def hamming_distance(s1: str, s2: str) -> int:
    """
    Oblicza odległość Hamminga pomiędzy dwoma ciągami znaków.
    Zwraca liczbę miejsc, na których ciągi się różnią.
    Dla ciągów o różnej długości, oblicza odległość dla wspólnej części
    i dodaje różnicę długości.
    """
    if len(s1) != len(s2):
        # Dla ciągów o różnej długości: różnica długości + Hamming dla wspólnej części
        min_length = min(len(s1), len(s2))
        length_diff = abs(len(s1) - len(s2))
        common_hamming = sum(
            c1 != c2 for c1, c2 in zip(s1[:min_length], s2[:min_length])
        )
        return length_diff + common_hamming
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


# --- 1b) Zmodyfikowana odległość Hamminga (klawiatura) ---


def modified_hamming_distance(s1: str, s2: str) -> int:
    """
    Oblicza zmodyfikowaną odległość Hamminga.
    - 0, jeśli znaki są te same.
    - 1, jeśli znaki sąsiadują na klawiaturze QWERTY.
    - 2, jeśli znaki nie sąsiadują.
    Dodatkowo, polskie znaki diakrytyczne są traktowane jako "sąsiedzi"
    swoich łacińskich odpowiedników (symulacja Alt + litera).
    """
    if len(s1) != len(s2):
        raise ValueError("Ciągi muszą mieć tę samą długość")

    # Uproszczona mapa sąsiedztwa na klawiaturze QWERTY
    adjacency = {
        "q": "wa",
        "w": "qase",
        "e": "wsdr",
        "r": "edft",
        "t": "rfgy",
        "y": "tghu",
        "u": "yhji",
        "i": "ujko",
        "o": "iklp",
        "p": "ol",
        "a": "qwsz",
        "s": "awedxz",
        "d": "serfcx",
        "f": "drtgvc",
        "g": "ftyhbv",
        "h": "gyujnb",
        "j": "huikmn",
        "k": "jiolm",
        "l": "kop",
        "z": "asx",
        "x": "zsdc",
        "c": "xdfv",
        "v": "cfgb",
        "b": "vghn",
        "n": "bhjm",
        "m": "njk",
    }

    # Mapa polskich znaków do ich bazowych odpowiedników
    diacritics_map = {
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ź": "z",
        "ż": "z",
    }

    distance = 0
    s1, s2 = s1.lower(), s2.lower()

    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            continue

        # Sprawdzenie "sąsiedztwa" diakrytycznego (np. a i ą)
        base_c1 = diacritics_map.get(c1, c1)
        base_c2 = diacritics_map.get(c2, c2)
        if (c1 != c2) and (base_c1 == base_c2):
            distance += 1
            continue

        # Sprawdzenie sąsiedztwa na klawiaturze
        if adjacency.get(c1) and c2 in adjacency[c1]:
            distance += 1
        else:
            distance += 2

    return distance


# --- 1c) Słownik i sprawdzanie podobieństwa ---


def spell_checker(word: str, dictionary: list[str]) -> str | list[str]:
    """
    Sprawdza, czy słowo jest w słowniku.
    - Jeśli tak, zwraca "OK".
    - Jeśli nie, zwraca do 3 najbardziej podobnych słów o tej samej długości
      na podstawie odległości Hamminga.
    """
    word = word.lower()
    dictionary_set = {d.lower() for d in dictionary}

    if word in dictionary_set:
        return "OK"
    else:
        # Filtruj słownik, aby znaleźć kandydatów o tej samej długości
        same_length_words = [d for d in dictionary if len(d) == len(word)]

        if not same_length_words:
            return "Nie znaleziono w słowniku słów o tej samej długości."

        # Oblicz odległości i posortujiougc
        distances = [(d, hamming_distance(word, d)) for d in same_length_words]
        distances.sort(key=lambda item: item[1])

        # Zwróć do 3 najlepszych propozycji
        return [item[0] for item in distances[:3]]


# =============================================================================
# --- ZADANIE 2: CZĘSTOŚĆ ZNAKÓW ---
# =============================================================================

# --- 2a) Tablice częstości ---
# Dane na podstawie strony z zadania (https://en.wikipedia.org/wiki/Letter_frequency)
# Sumujemy częstości liter z ogonkami i bez nich.
# Częstości są w %, dla uproszczenia pozostawione jako takie.

FREQ_TABLES = {
    "english": {
        "a": 8.2,
        "b": 1.5,
        "c": 2.8,
        "d": 4.3,
        "e": 12.7,
        "f": 2.2,
        "g": 2.0,
        "h": 6.1,
        "i": 7.0,
        "j": 0.15,
        "k": 0.77,
        "l": 4.0,
        "m": 2.4,
        "n": 6.7,
        "o": 7.5,
        "p": 1.9,
        "q": 0.095,
        "r": 6.0,
        "s": 6.3,
        "t": 9.1,
        "u": 2.8,
        "v": 0.98,
        "w": 2.4,
        "x": 0.15,
        "y": 2.0,
        "z": 0.074,
    },
    "german": {
        "a": 6.51,
        "b": 1.89,
        "c": 3.06,
        "d": 5.08,
        "e": 17.40,
        "f": 1.66,
        "g": 3.01,
        "h": 4.76,
        "i": 7.55,
        "j": 0.27,
        "k": 1.21,
        "l": 3.44,
        "m": 2.53,
        "n": 9.78,
        "o": 2.51,
        "p": 0.79,
        "q": 0.02,
        "r": 7.00,
        "s": 7.27,
        "t": 6.15,
        "u": 4.35,
        "v": 0.67,
        "w": 1.89,
        "x": 0.03,
        "y": 0.04,
        "z": 1.13,
    },
    "polish": {  # Połączone 'a' z 'ą', 'c' z 'ć' itd.
        "a": 8.91 + 0.99,
        "b": 1.47,
        "c": 3.96 + 0.4,
        "d": 3.25,
        "e": 7.66 + 1.11,
        "f": 0.3,
        "g": 1.42,
        "h": 1.08,
        "i": 8.21,
        "j": 2.28,
        "k": 3.51,
        "l": 2.1 + 1.82,
        "m": 2.8,
        "n": 5.72 + 0.2,
        "o": 7.75 + 0.9,
        "p": 3.13,
        "r": 4.69,
        "s": 4.32 + 0.66,
        "t": 3.98,
        "u": 2.5,
        "w": 4.65,
        "y": 3.76,
        "z": 5.64 + 0.06 + 0.83,
    },
}


def normalize_text(text: str) -> str:
    """Usuwa znaki diakrytyczne i zwraca tekst z małymi literami."""
    # Usuwa znaki diakrytyczne, np. 'ą' -> 'a'
    nfkd_form = unicodedata.normalize("NFKD", text.lower())
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


# --- 2b) Identyfikacja języka ---


def get_char_frequencies(text: str) -> dict[str, float]:
    """Oblicza częstość występowania liter w podanym tekście."""
    # Normalizuj tekst (małe litery, bez 'ogonków')
    normalized = normalize_text(text)
    # Zliczaj tylko litery alfabetu
    letters_only = [c for c in normalized if "a" <= c <= "z"]

    if not letters_only:
        return {}

    counts = collections.Counter(letters_only)
    total_letters = len(letters_only)

    # Zwróć częstości w %
    return {char: (count / total_letters) * 100 for char, count in counts.items()}


def detect_language(text: str, freq_tables: dict) -> str:
    """
    Wykrywa język tekstu przez porównanie częstości jego liter
    z predefiniowanymi tabelami częstości.
    Używa sumy kwadratów różnic jako miary odległości.
    """
    text_freq = get_char_frequencies(text)
    if not text_freq:
        return "Nie można przeanalizować tekstu (brak liter)."

    all_chars = set(text_freq.keys())
    for lang_table in freq_tables.values():
        all_chars.update(lang_table.keys())

    scores = {}
    for lang, lang_freq in freq_tables.items():
        # Suma kwadratów różnic
        error = sum(
            (text_freq.get(char, 0) - lang_freq.get(char, 0)) ** 2 for char in all_chars
        )
        scores[lang] = error

    # Zwróć język z najmniejszym błędem
    return min(scores, key=scores.get)


# --- 2c) Skrócone tablice (samogłoski/spółgłoski) ---

VOWELS = "aeiou"


def get_vowel_consonant_freq(text: str) -> dict[str, float]:
    """Oblicza częstość samogłosek i spółgłosek w tekście."""
    normalized = normalize_text(text)
    letters_only = [c for c in normalized if "a" <= c <= "z"]

    if not letters_only:
        return {"vowels": 0, "consonants": 0}

    vowel_count = sum(1 for char in letters_only if char in VOWELS)
    total_letters = len(letters_only)

    vowel_freq = (vowel_count / total_letters) * 100
    consonant_freq = 100 - vowel_freq

    return {"vowels": vowel_freq, "consonants": consonant_freq}


def get_vowel_consonant_tables(freq_tables: dict) -> dict:
    """Tworzy skrócone tablice częstości samogłosek/spółgłosek."""
    vc_tables = {}
    for lang, table in freq_tables.items():
        vowel_total = sum(freq for char, freq in table.items() if char in VOWELS)
        consonant_total = sum(
            freq for char, freq in table.items() if char not in VOWELS
        )
        # Normalizacja do 100%
        total = vowel_total + consonant_total
        vc_tables[lang] = {
            "vowels": (vowel_total / total) * 100,
            "consonants": (consonant_total / total) * 100,
        }
    return vc_tables


def detect_language_vowel_consonant(text: str, vc_freq_tables: dict) -> str:
    """Wykrywa język na podstawie częstości samogłosek i spółgłosek."""
    text_vc_freq = get_vowel_consonant_freq(text)

    scores = {}
    for lang, lang_vc_freq in vc_freq_tables.items():
        error = (text_vc_freq["vowels"] - lang_vc_freq["vowels"]) ** 2 + (
            text_vc_freq["consonants"] - lang_vc_freq["consonants"]
        ) ** 2
        scores[lang] = error

    return min(scores, key=scores.get)


# =============================================================================
# --- ZADANIE 3: ODLEGŁOŚCI EDYCYJNE ---
# =============================================================================

# --- 3a) Najdłuższy wspólny podciąg (bez przerw) ---


def longest_common_substring(s1: str, s2: str) -> tuple[str, int]:
    """
    Znajduje najdłuższy wspólny podciąg (ciągły) dla dwóch ciągów znaków.
    Zwraca ten podciąg oraz jego długość.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    end_pos = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0

    substring = s1[end_pos - max_len : end_pos]
    return substring, max_len


# --- 3b) Najdłuższy wspólny podciąg (z przerwami) ---


def longest_common_subsequence(s1: str, s2: str) -> tuple[str, int]:
    """
    Znajduje najdłuższy wspólny podciąg (nieciągły) dla dwóch ciągów znaków.
    Zwraca ten podciąg (jeden z możliwych) oraz jego długość.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
    # print(f"Macierz DP:\n{dp}")  # Debug: Wyświetlenie macierzy DP

    # Odtwarzanie podciągu
    subsequence = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            subsequence.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    result_str = "".join(reversed(subsequence))
    return result_str, dp[m][n]


# --- 3d*) Odległość Levenshteina ---


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Oblicza odległość Levenshteina między dwoma ciągami.
    Jest to minimalna liczba operacji (wstawienie, usunięcie, zamiana)
    potrzebnych do przekształcenia jednego ciągu w drugi.
    """
    m, n = len(s1), len(s2)

    # Utwórz macierz DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Inicjalizacja macierzy
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Wypełnianie macierzy
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Usunięcie
                dp[i][j - 1] + 1,  # Wstawienie
                dp[i - 1][j - 1] + cost,  # Zamiana
            )

    return dp[m][n]


# =============================================================================
# --- DEMO ---
# =============================================================================
if __name__ == "__main__":
    print("=" * 40)
    print("--- ZADANIE 1: ODLEGŁOŚĆ HAMMINGA ---")
    print("=" * 40)

    print("\n--- 1a) Odległość Hamminga ---")
    s1, s2 = "karolin", "kathrin"
    print(f"Odległość Hamminga między '{s1}' a '{s2}': {hamming_distance(s1, s2)}")
    s1, s2 = "płaszczka", "płaszczkaa"
    print(f"Odległość Hamminga między '{s1}' a '{s2}': {hamming_distance(s1, s2)}")

    print("\n--- 1b) Zmodyfikowana odległość Hamminga ---")
    s1, s2 = "mama", "nawa"
    print(
        f"Zmodyfikowana odległość między '{s1}' a '{s2}': {modified_hamming_distance(s1, s2)}"
    )
    s1, s2 = "kot", "kąt"
    print(
        f"Zmodyfikowana odległość między '{s1}' a '{s2}' (ą): {modified_hamming_distance(s1, s2)}"
    )

    print("\n--- 1c) Słownik ---")
    my_dictionary = [
        "kot",
        "pies",
        "dom",
        "stół",
        "okno",
        "drzwi",
        "lampa",
        "kocur",
        "biurko",
        "lato",
        "zima",
        "wiosna",
        "jesień",
        "płaszcz",
        "kwiat",
    ]
    word1 = "kocur"
    word2 = "kocurr"  # nie ma w słowniku
    word3 = "koter"  # literówka w "kocur"
    print(f"Sprawdzam słowo '{word1}': {spell_checker(word1, my_dictionary)}")
    print(f"Sprawdzam słowo '{word2}': {spell_checker(word2, my_dictionary)}")
    print(f"Sprawdzam słowo '{word3}': {spell_checker(word3, my_dictionary)}")

    print("\n" + "=" * 40)
    print("--- ZADANIE 2: CZĘSTOŚĆ ZNAKÓW ---")
    print("=" * 40)

    text_pl = (
        "W Szczebrzeszynie chrząszcz brzmi w trzcinie i Szczebrzeszyn z tego słynie."
    )
    text_en = "The quick brown fox jumps over the lazy dog. im an it student on wroclaw univercity of science and technology."
    text_de = "Die Ostfriesischen Inseln liegen vor der Küste von Niedersachsen "

    print("\n--- 2b) Detekcja języka (pełne tablice) ---")
    print(
        f"Tekst: '{text_pl}' -> Wykryty język: {detect_language(text_pl, FREQ_TABLES)}"
    )
    print(
        f"Tekst: '{text_en[:20]}...' -> Wykryty język: {detect_language(text_en, FREQ_TABLES)}"
    )
    print(
        f"Tekst: '{text_de[:20]}...' -> Wykryty język: {detect_language(text_de, FREQ_TABLES)}"
    )

    print("\n--- 2c) Detekcja języka (samogłoski/spółgłoski) ---")
    vc_tables = get_vowel_consonant_tables(FREQ_TABLES)
    print("Utworzono skrócone tablice częstości.")
    print(
        f"Tekst: '{text_pl}' -> Wykryty język: {detect_language_vowel_consonant(text_pl, vc_tables)}"
    )
    print(
        f"Tekst: '{text_en[:20]}...' -> Wykryty język: {detect_language_vowel_consonant(text_en, vc_tables)}"
    )
    print(
        f"Tekst: '{text_de[:20]}...' -> Wykryty język: {detect_language_vowel_consonant(text_de, vc_tables)}"
    )

    print("\n" + "=" * 40)
    print("--- ZADANIE 3: ODLEGŁOŚCI EDYCYJNE ---")
    print("=" * 40)

    print("\n--- 3a) Najdłuższy wspólny podciąg (bez przerw) ---")
    s1, s2 = "konwalia", "zawalina"
    substring, length = longest_common_substring(s1, s2)
    print(
        f"Dla '{s1}' i '{s2}', najdłuższy wspólny podciąg to '{substring}' (długość: {length})"
    )

    print("\n--- 3b) Najdłuższy wspólny podciąg (z przerwami) ---")
    s1, s2 = "ApqBCrDsEF", "tABuCvDEwxFyz"
    subsequence, length = longest_common_subsequence(
        s1.lower(), s2.lower()
    )  # ignorujemy wielkość liter
    print(
        f"Dla '{s1}' i '{s2}', najdłuższy wspólny podciąg to '{subsequence.upper()}' (długość: {length})"
    )

    print("\n--- 3c) Złożoność obliczeniowa i Drzewa Trie (komentarz) ---")
    print(
        "Algorytmy znajdowania najdłuższego wspólnego podciągu (Substring i Subsequence) oraz odległości Levenshteina,"
    )
    print(
        "zaimplementowane przy użyciu programowania dynamicznego, mają złożoność czasową O(m*n), gdzie m i n to długości"
    )
    print(
        "porównywanych ciągów. Jest to efektywne dla słów, ale może być zbyt wolne dla bardzo długich tekstów."
    )
    print(
        "\nDrzewa Trie (prefiksowe) to struktura danych idealna do przechowywania słowników. Pozwalają na bardzo szybkie"
    )
    print(
        "sprawdzanie, czy słowo (lub jego prefiks) istnieje w zbiorze. Użycie Trie do znajdowania sugestii literówek"
    )
    print(
        "(np. w połączeniu z algorytmem odległości edycyjnej) jest znacznie bardziej wydajne niż iterowanie po całej liście słów."
    )
    print("Link z zadania: https://pl.wikipedia.org/wiki/Drzewo_trie")

    print("\n--- 3d*) Odległość Levenshteina ---")
    s1, s2 = "kot", "sok"
    print(
        f"Odległość Levenshteina między '{s1}' a '{s2}': {levenshtein_distance(s1, s2)}"
    )
    s1, s2 = "płaszczka", "płaszczkaa"
    print(
        f"Odległość Levenshteina między '{s1}' a '{s2}': {levenshtein_distance(s1, s2)}"
    )
    s1, s2 = "algorytm", "logarytm"
    print(
        f"Odległość Levenshteina między '{s1}' a '{s2}': {levenshtein_distance(s1, s2)}"
    )
