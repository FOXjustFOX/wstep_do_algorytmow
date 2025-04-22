import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


# Task 1: Node class for linked list implementation of the queue
class Node:
    def __init__(self, task_type, task_size):
        self.task_type = task_type  # A, B, or C
        self.task_size = task_size  # Integer representing task complexity/duration
        self.next = None


# Linked list queue implementation
class TaskQueue:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def enqueue(self, task_type, task_size):
        new_node = Node(task_type, task_size)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1

    def is_empty(self):
        return self.head is None

    def dequeue_specific_type(self, task_type):
        if self.is_empty():
            return None

        # Special case for expert window accepting any task type
        if task_type == "E":
            result = (self.head.task_type, self.head.task_size)
            self.head = self.head.next
            if self.head is None:
                self.tail = None
            self.size -= 1
            return result

        # Looking for specific task type
        prev = None
        current = self.head

        # If first node matches
        if current.task_type == task_type:
            result = (current.task_type, current.task_size)
            self.head = current.next
            if self.head is None:
                self.tail = None
            self.size -= 1
            return result

        # Check rest of the list
        while current and current.task_type != task_type:
            prev = current
            current = current.next

        if current:  # Found a matching task
            result = (current.task_type, current.task_size)
            prev.next = current.next
            if current == self.tail:
                self.tail = prev
            self.size -= 1
            return result

        return None  # No matching task found

    # Optimization: Maintain separate pointers for each task type
    def optimize(self):
        """
        This method illustrates a suggested optimization:
        Create separate queues for each task type for faster retrieval.
        """
        type_a_queue = TaskQueue()
        type_b_queue = TaskQueue()
        type_c_queue = TaskQueue()

        current = self.head
        while current:
            if current.task_type == "A":
                type_a_queue.enqueue("A", current.task_size)
            elif current.task_type == "B":
                type_b_queue.enqueue("B", current.task_size)
            else:  # Type C
                type_c_queue.enqueue("C", current.task_size)
            current = current.next

        return type_a_queue, type_b_queue, type_c_queue

    # For sorting tasks in queue (Part 2f)
    def sort_tasks(self, ascending=True):
        if self.size <= 1:
            return

        # Convert linked list to array for easier sorting
        tasks = []
        current = self.head
        while current:
            tasks.append((current.task_type, current.task_size))
            current = current.next

        # Sort by task size
        tasks.sort(key=lambda x: x[1], reverse=not ascending)

        # Rebuild the queue
        self.head = None
        self.tail = None
        for task_type, task_size in tasks:
            self.enqueue(task_type, task_size)

    def clone(self):
        """Create a copy of the current queue"""
        new_queue = TaskQueue()
        current = self.head
        while current:
            new_queue.enqueue(current.task_type, current.task_size)
            current = current.next
        return new_queue

    def __len__(self):
        return self.size


# Office simulation class
class OfficeSimulation:
    def __init__(self, windows_config):
        """
        Initialize office with specific window configuration
        windows_config: dictionary with keys 'A', 'B', 'C', 'E' and values representing the count of each window type
        """
        self.windows = []
        window_id = 1

        for window_type, count in windows_config.items():
            for _ in range(count):
                # Each window is represented as [id, type, current_task_time_remaining]
                self.windows.append([window_id, window_type, 0])
                window_id += 1

        # Stats counters
        self.window_stats = {window[0]: 0 for window in self.windows}

    def process_queue(self, task_queue):
        time = 0
        queue = task_queue.clone()  # Work on a copy to preserve original queue

        while not queue.is_empty() or any(window[2] > 0 for window in self.windows):
            # Decrease time for busy windows and assign new tasks to free windows
            for i, window in enumerate(self.windows):
                if window[2] <= 0:  # Window is free
                    window_type = window[1]

                    # Try to get a task
                    if window_type == "E":  # Expert window can take any task
                        if not queue.is_empty():
                            task = queue.dequeue_specific_type(
                                "E"
                            )  # Will get first task of any type
                            if task:
                                task_type, task_size = task
                                self.windows[i][2] = task_size
                                self.window_stats[window[0]] += 1
                    else:  # Regular window
                        task = queue.dequeue_specific_type(window_type)
                        if task:
                            task_type, task_size = task
                            self.windows[i][2] = task_size
                            self.window_stats[window[0]] += 1
                else:  # Window is busy
                    window[2] -= 1

            time += 1

        return time

    def get_stats(self):
        return self.window_stats


# Task 1: Create an office with 10 windows (3A, 3B, 3C, 1E)
def task1():
    print("===== Task 1: Office with 10 windows =====")

    # Configuration with 10 windows: 3A, 3B, 3C, 1E
    office_config = {"A": 3, "B": 3, "C": 3, "E": 1}
    office = OfficeSimulation(office_config)

    # Create queue with 40 clients with random task types
    queue = TaskQueue()
    task_counts = {"A": 0, "B": 0, "C": 0}

    for _ in range(40):
        task_type = random.choice(["A", "B", "C"])
        task_counts[task_type] += 1

        # Assign task size based on type
        if task_type == "A":
            task_size = random.randint(1, 4)
        elif task_type == "B":
            task_size = random.randint(5, 8)
        else:  # Type C
            task_size = random.randint(9, 12)

        queue.enqueue(task_type, task_size)

    print(
        f"\nQueue created with: \n{task_counts['A']} type A tasks, \n{task_counts['B']} type B tasks, \n{task_counts['C']} type C tasks\n"
    )

    # Process the queue
    total_time = office.process_queue(queue)
    window_stats = office.get_stats()

    print(f"Total time to process all tasks: {total_time} time units")
    print("Tasks processed by each window:")
    for window_id, count in window_stats.items():
        print(f"Window {window_id}: {count} tasks")


# Task 2: Compare different office configurations
def task2():
    print("\n===== Task 2: Comparing office configurations =====")

    # Three office configurations
    config1 = {"A": 3, "B": 3, "C": 3, "E": 0}  # 9 windows: 3A, 3B, 3C
    config2 = {"A": 2, "B": 2, "C": 2, "E": 3}  # 9 windows: 2A, 2B, 2C, 3E
    config3 = {"A": 1, "B": 2, "C": 3, "E": 1}  # 7 windows: 1A, 2B, 3C, 1E

    # Probabilities for task types (must sum to 1)
    prob_a = 0.2
    prob_b = 0.3
    prob_c = 0.5

    # Function to generate a queue based on probabilities
    def generate_queue(size):
        queue = TaskQueue()
        for _ in range(size):
            rand = random.random()
            if rand < prob_a:
                task_type = "A"
                task_size = random.randint(1, 4)
            elif rand < prob_a + prob_b:
                task_type = "B"
                task_size = random.randint(5, 8)
            else:
                task_type = "C"
                task_size = random.randint(9, 12)
            queue.enqueue(task_type, task_size)
        return queue

    # Test with a single queue of 50 clients
    test_queue = generate_queue(50)

    office1 = OfficeSimulation(config1)
    office2 = OfficeSimulation(config2)
    office3 = OfficeSimulation(config3)

    time1 = office1.process_queue(test_queue)
    time2 = office2.process_queue(test_queue)
    time3 = office3.process_queue(test_queue)

    print("Processing time for a single queue of 50 clients:")
    print(f"Office 1 (3A, 3B, 3C): {time1} time units")
    print(f"Office 2 (2A, 2B, 2C, 3E): {time2} time units")
    print(f"Office 3 (1A, 2B, 3C, 1E): {time3} time units")

    # Test with 100 different queues
    times1 = []
    times2 = []
    times3 = []

    for i in range(100):
        queue = generate_queue(50)

        office1 = OfficeSimulation(config1)
        office2 = OfficeSimulation(config2)
        office3 = OfficeSimulation(config3)

        times1.append(office1.process_queue(queue))
        times2.append(office2.process_queue(queue))
        times3.append(office3.process_queue(queue))

    avg_time1 = sum(times1) / len(times1)
    avg_time2 = sum(times2) / len(times2)
    avg_time3 = sum(times3) / len(times3)

    print("\nAverage processing times over 100 queues:")
    print(f"Office 1 (3A, 3B, 3C): {avg_time1:.2f} time units")
    print(f"Office 2 (2A, 2B, 2C, 3E): {avg_time2:.2f} time units")
    print(f"Office 3 (1A, 2B, 3C, 1E): {avg_time3:.2f} time units")

    # Plot histograms
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(times1, bins=15, alpha=0.7, color="blue")
    plt.axvline(avg_time1, color="red", linestyle="dashed", linewidth=1)
    plt.title("Office 1: 3A, 3B, 3C")
    plt.xlabel("Processing Time")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 2)
    plt.hist(times2, bins=15, alpha=0.7, color="green")
    plt.axvline(avg_time2, color="red", linestyle="dashed", linewidth=1)
    plt.title("Office 2: 2A, 2B, 2C, 3E")
    plt.xlabel("Processing Time")

    plt.subplot(1, 3, 3)
    plt.hist(times3, bins=15, alpha=0.7, color="orange")
    plt.axvline(avg_time3, color="red", linestyle="dashed", linewidth=1)
    plt.title("Office 3: 1A, 2B, 3C, 1E")
    plt.xlabel("Processing Time")

    plt.tight_layout()
    plt.savefig("office_comparison.png")
    plt.show()

    # Test the effect of queue ordering (2f)
    print("\nTesting effect of queue ordering:")

    # Generate a test queue
    order_test_queue = generate_queue(50)

    # Original queue
    office_test = OfficeSimulation(config2)
    original_time = office_test.process_queue(order_test_queue)

    # Ascending order (smallest tasks first)
    asc_queue = order_test_queue.clone()
    asc_queue.sort_tasks(ascending=True)
    office_test = OfficeSimulation(config2)
    asc_time = office_test.process_queue(asc_queue)

    # Descending order (largest tasks first)
    desc_queue = order_test_queue.clone()
    desc_queue.sort_tasks(ascending=False)
    office_test = OfficeSimulation(config2)
    desc_time = office_test.process_queue(desc_queue)

    print(f"Original queue processing time: {original_time}")
    print(f"Ascending order (smallest tasks first): {asc_time}")
    print(f"Descending order (largest tasks first): {desc_time}")

    # Proposed optimization (2e)
    print("\nProposed optimized office configuration based on task probabilities:")

    # Since C tasks are most common (50%) and take longest, allocate more resources there
    # A tasks are least common (20%) and quickest, so fewer windows needed
    optimized_config = {"A": 1, "B": 2, "C": 4, "E": 2}

    optimized_office = OfficeSimulation(optimized_config)
    optimized_times = []

    for i in range(100):
        queue = generate_queue(50)
        optimized_times.append(optimized_office.process_queue(queue))

    avg_optimized_time = sum(optimized_times) / len(optimized_times)
    print(f"Optimized office (1A, 2B, 4C, 2E): {avg_optimized_time:.2f} time units")
    


# Run the tasks
if __name__ == "__main__":
    task1()
    task2()
