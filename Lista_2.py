import time
import random
import matplotlib.pyplot as plt
import numpy as np


def bubble_sort_standard(lst):
    """Standard bubble"""
    n = len(lst)
    for i in range(n):
        for j in range(0, n - i - 1):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]


def bubble_sort_early_exit(lst):
    """Break if no swaps occurred in a complete pass"""
    n = len(lst)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                swapped = True
        if not swapped:
            break


def bubble_sort_NAIVE(lst):
    """we go through the whole list for each element"""
    n = len(lst)
    for i in range(n):
        for j in range(0, n - 1):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]


def insertion_sort(lst):
    """Simple insertion"""
    for i in range(1, len(lst)):
        key = lst[i]
        j = i - 1
        while j >= 0 and lst[j] > key:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = key


def selection_sort(lst):
    """Simple selection"""
    n = len(lst)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if lst[j] < lst[min_index]:
                min_index = j
        lst[i], lst[min_index] = lst[min_index], lst[i]


list_sizes = [10, 20, 50, 100, 200, 500, 1000]
num_trials = 10

# -----------------------------
# Bubble Sort Modifications
# -----------------------------
bubble_versions = {
    "Bubble Standard": bubble_sort_standard,
    "Bubble Early Exit": bubble_sort_early_exit,
    "Bubble Naive": bubble_sort_NAIVE,
}

# {algorithm_name: {list_size: [times]}}
results_bubble = {name: {size: [] for size in list_sizes} for name in bubble_versions}

for size in list_sizes:
    for _ in range(num_trials):
        # generate a random list of integers
        base_list = [random.randint(0, 10000) for _ in range(size)]
        for name, sort_func in bubble_versions.items():
            lst_copy = (
                base_list.copy()
            )  # copy for each algorithm to work on the same list
            start = time.perf_counter()
            sort_func(lst_copy)
            end = time.perf_counter()
            elapsed = (end - start) * 1e9  # convert seconds to nanoseconds
            results_bubble[name][size].append(elapsed)

# Compute average and maximum times for each bubble sort modification
bubble_avg = {name: [] for name in bubble_versions}
bubble_max = {name: [] for name in bubble_versions}
for name in bubble_versions:
    for size in list_sizes:
        times = results_bubble[name][size]
        bubble_avg[name].append(sum(times) / len(times))
        bubble_max[name].append(max(times))

# -----------------------------
# sorting alg comparison
# -----------------------------
algorithms = {
    "Bubble Standard": bubble_sort_standard,
    "Insertion Sort": insertion_sort,
    "Selection Sort": selection_sort,
    "Python Sort": lambda lst: lst.sort(),  # built-in sort
}

results_algo = {name: {size: [] for size in list_sizes} for name in algorithms}

for size in list_sizes:
    for _ in range(num_trials):
        base_list = [random.randint(0, 10000) for _ in range(size)]
        for name, sort_func in algorithms.items():
            lst_copy = base_list.copy()
            start = time.perf_counter()
            sort_func(lst_copy)
            end = time.perf_counter()
            elapsed = (end - start) * 1e9  # nanoseconds
            results_algo[name][size].append(elapsed)

# Compute average and maximum times for each algorithm
algo_avg = {name: [] for name in algorithms}
algo_max = {name: [] for name in algorithms}
for name in algorithms:
    for size in list_sizes:
        times = results_algo[name][size]
        algo_avg[name].append(sum(times) / len(times))
        algo_max[name].append(max(times))

# Plots

# Figure 1: Bubble Sort Modifications (Average and Maximum times)
x = np.arange(len(list_sizes))
width = 0.25  # bar width for grouping

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot Average times for bubble modifications
ax1.bar(x - width, bubble_avg["Bubble Standard"], width, label="Bubble Standard")
ax1.bar(x, bubble_avg["Bubble Early Exit"], width, label="Bubble Early Exit")
ax1.bar(x + width, bubble_avg["Bubble Naive"], width, label="Bubble Naive")
ax1.set_xlabel("Rozmiar ciągu (n)")
ax1.set_ylabel("Czas wykonania (ns)")
ax1.set_title("Średni czas działania modyfikacji sortowania bąbelkowego")
ax1.set_xticks(x)
ax1.set_xticklabels(list_sizes)
ax1.legend()
ax1.set_yscale("log")
ax1.grid(True)

# Plot Maximum times for bubble modifications
ax2.bar(x - width, bubble_max["Bubble Standard"], width, label="Bubble Standard")
ax2.bar(x, bubble_max["Bubble Early Exit"], width, label="Bubble Early Exit")
ax2.bar(x + width, bubble_max["Bubble Naive"], width, label="Bubble Naive")
ax2.set_xlabel("Rozmiar ciągu (n)")
ax2.set_ylabel("Czas wykonania (ns)")
ax2.set_title("Maksymalny czas działania modyfikacji sortowania bąbelkowego")
ax2.set_xticks(x)
ax2.set_xticklabels(list_sizes)
ax2.legend()
ax2.grid(True)
ax2.set_yscale("log")


# plot 2
x2 = np.arange(len(list_sizes))
width2 = 0.20
colors = {
    "Bubble Standard": "blue",
    "Insertion Sort": "green",
    "Selection Sort": "red",
    "Python Sort": "purple",
}

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))

# Plot Average times various algorithms
for idx, (name, color) in enumerate(colors.items()):
    ax3.bar(x2 + (idx - 1.5) * width2, algo_avg[name], width2, label=name, color=color)
ax3.set_xlabel("Rozmiar ciągu (n)")
ax3.set_ylabel("Czas wykonania (ns)")
ax3.set_title("Średni czas działania algorytmów sortowania")
ax3.set_xticks(x2)
ax3.set_xticklabels(list_sizes)
ax3.legend()
ax3.grid(True)
ax3.set_yscale("log")
# Plot Maximum times various algorithms
for idx, (name, color) in enumerate(colors.items()):
    ax4.bar(x2 + (idx - 1.5) * width2, algo_max[name], width2, label=name, color=color)
ax4.set_xlabel("Rozmiar ciągu (n)")
ax4.set_ylabel("Czas wykonania (ns)")
ax4.set_title("Maksymalny czas działania algorytmów sortowania")
ax4.set_xticks(x2)
ax4.set_xticklabels(list_sizes)
ax4.legend()
ax4.grid(True)
ax4.set_yscale("log")

plt.show()
