import random
import heapq
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- Data Structures ---

# Simple Adjacency List representation
# For unweighted graphs: {node: [neighbor1, neighbor2, ...]}
# For weighted graphs: {node: [(neighbor1, weight1), (neighbor2, weight2), ...]}


# Simple Disjoint Set Union (DSU) for Kruskal's algorithm
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n  # Used for union by rank optimization

    def find(self, i):
        # Path compression
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Union by rank
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True  # Union was successful
        return False  # Elements were already in the same set


# --- Graph Generation ---


def generate_random_unweighted_graph(num_vertices, edge_probability):
    """Generuje losowy nieskierowany graf nieważony."""
    graph = {i: [] for i in range(num_vertices)}
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < edge_probability:
                graph[i].append(j)
                graph[j].append(i)  # Undirected
    return graph


def generate_random_weighted_graph(num_vertices, edge_probability, max_weight=10):
    """Generuje losowy nieskierowany graf ważony z nieujemnymi wagami."""
    graph = {i: [] for i in range(num_vertices)}
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < edge_probability:
                weight = random.randint(
                    1, max_weight
                )  # Non-negative weights for Dijkstra
                graph[i].append((j, weight))
                graph[j].append((i, weight))  # Undirected
    return graph


# --- Task 1: Connected Components ---


def find_connected_components(graph):
    """Znajduje składowe spójne w grafie nieskierowanym."""
    num_vertices = len(graph)
    visited = [False] * num_vertices
    components = []

    # Adjust DFS for potentially weighted list format {node: [(neighbor, weight), ...]}
    def dfs(u, current_component, current_graph):
        visited[u] = True
        current_component.append(u)
        for edge in current_graph.get(u, []):
            # edge can be just a neighbor (unweighted list) or (neighbor, weight) tuple (weighted list)
            if isinstance(edge, tuple):
                neighbor = edge[0]
            else:
                neighbor = edge  # Unweighted case

            if not visited[neighbor]:
                dfs(neighbor, current_component, current_graph)

    # Iterate through all vertices
    for v in range(num_vertices):
        if not visited[v]:
            current_component = []
            dfs(v, current_component, graph)
            components.append(current_component)

    return components


# --- Task 2: Dijkstra's Algorithm ---


def dijkstra(graph, start_node, end_node=None):
    """Implementacja algorytmu Dijkstry dla grafu ważonego."""
    num_vertices = len(graph)
    distances = {v: math.inf for v in range(num_vertices)}
    predecessors = {v: None for v in range(num_vertices)}
    distances[start_node] = 0

    # Priority queue: stores tuples (distance, vertex)
    priority_queue = [(0, start_node)]
    
    # Count of visited (processed) nodes
    nodes_visited = 0

    while priority_queue:
        current_distance, u = heapq.heappop(priority_queue)
        
        # Count this node as visited (processed)
        nodes_visited += 1

        # If we found the shortest path to end_node, stop early
        if end_node is not None and u == end_node:
            break

        # If current distance is greater than already finalized distance, skip
        if current_distance > distances[u]:
            continue

        # Explore neighbors
        for v, weight in graph.get(u, []):
            distance_through_u = distances[u] + weight

            if distance_through_u < distances[v]:
                distances[v] = distance_through_u
                predecessors[v] = u
                heapq.heappush(priority_queue, (distance_through_u, v))

    path = []
    if end_node is not None:
        current = end_node
        while current is not None:
            path.append(current)
            # Handle case where end_node is unreachable
            if predecessors[current] is None and current != start_node:
                path = []  # Clear path as it's unreachable
                break
            current = predecessors[current]

        path.reverse()
        # Return path, distance, and nodes visited if path exists, otherwise indicate no path
        if path and path[0] == start_node:
            return path, distances[end_node], nodes_visited
        else:
            return [], math.inf, nodes_visited  # Indicate no path found

    # If end_node is None, return all distances, predecessors, and nodes visited
    return distances, predecessors, nodes_visited


def display_path_with_distances(graph, path):
    """Wyświetla ścieżkę jako ciąg wierzchołków z odległościami między nimi."""
    if len(path) < 2:
        return
    
    total_distance = 0
    
    for i in range(len(path) - 1):
        current_vertex = path[i]
        next_vertex = path[i + 1]
        
        # Find the edge weight between current and next vertex
        edge_weight = None
        for neighbor, weight in graph.get(current_vertex, []):
            if neighbor == next_vertex:
                edge_weight = weight
                break
        
        if edge_weight is not None:
            total_distance += edge_weight
            if i == 0:
                print(f"  {current_vertex} --({edge_weight})--> {next_vertex}", end="")
            else:
                print(f" --({edge_weight})--> {next_vertex}", end="")
        else:
            print(f"\n    Error: No direct edge found between {current_vertex} and {next_vertex}")
            return
    
    
    
    print(f"\n  Total distance: {total_distance}")


def manhattan_heuristic(node1, node2):
    """
    Prosta heurystyka dla algorytmu A*.
    Zwraca bezwzględną różnicę między indeksami węzłów.
    Jest to heurystyka dopuszczalna dla większości struktur grafowych.
    """
    return abs(node1 - node2)


def a_star(graph, start_node, end_node, heuristic_func=None):
    
    if heuristic_func is None:
        heuristic_func = manhattan_heuristic
    
    num_vertices = len(graph)
    
    # g_score: koszt dotarcia od start_node do każdego węzła
    g_score = {v: math.inf for v in range(num_vertices)}
    g_score[start_node] = 0
    
    # f_score: g_score + heurystyka (szacowany całkowity koszt)
    f_score = {v: math.inf for v in range(num_vertices)}
    f_score[start_node] = heuristic_func(start_node, end_node)
    
    # Śledzi poprzedniki dla rekonstrukcji ścieżki
    predecessors = {v: None for v in range(num_vertices)}
    
    # Open set: węzły do sprawdzenia (priority queue)
    open_set = [(f_score[start_node], start_node)]
    open_set_hash = {start_node}  # Dla szybkiego sprawdzania członkostwa
    
    # Closed set: węzły już sprawdzone
    closed_set = set()
    
    nodes_explored = 0
    
    while open_set:
        # Pobierz węzeł z najniższym f_score
        current_f, current = heapq.heappop(open_set)
        open_set_hash.discard(current)
        
        nodes_explored += 1
        
        if current == end_node:
            # Rekonstruuj ścieżkę
            path = []
            current_node = end_node
            while current_node is not None:
                path.append(current_node)
                current_node = predecessors[current_node]
            path.reverse()
            
            return path, g_score[end_node], nodes_explored
        
        # Przenieś do closed set
        closed_set.add(current)
        
        # Sprawdź wszystkich sąsiadów
        for neighbor, weight in graph.get(current, []):
            if neighbor in closed_set:
                continue
            
            tentative_g_score = g_score[current] + weight
            
            # Jeśli znaleźliśmy lepszą ścieżkę do sąsiada
            if tentative_g_score < g_score[neighbor]:
                predecessors[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic_func(neighbor, end_node)
                
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)
    
    return [], math.inf, nodes_explored  # Brak ścieżki


def dijkstra_multi_source(graph, start_nodes):
    """Implementacja algorytmu Dijkstry z wieloma punktami startowymi."""
    num_vertices = len(graph)
    distances = {v: math.inf for v in range(num_vertices)}
    predecessors = {v: None for v in range(num_vertices)}

    # Priority queue: stores tuples (distance, vertex)
    priority_queue = []

    # Initialize distances and queue for all start nodes
    for start_node in start_nodes:
        if 0 <= start_node < num_vertices:
            distances[start_node] = 0
            heapq.heappush(priority_queue, (0, start_node))
        else:
            print(f"Warning: Start node {start_node} is out of bounds.")

    while priority_queue:
        current_distance, u = heapq.heappop(priority_queue)

        if current_distance > distances[u]:
            continue

        for v, weight in graph.get(u, []):
            distance_through_u = distances[u] + weight

            if distance_through_u < distances[v]:
                distances[v] = distance_through_u
                predecessors[v] = u
                heapq.heappush(priority_queue, (distance_through_u, v))

    # Return distances from the *nearest* source for each node
    # The predecessors map shows the path *back* to the nearest source
    return distances, predecessors


def analyze_dijkstra_tree(predecessors, start_node):
    """Analizuje strukturę utworzoną przez wskaźniki poprzedników z Dijkstry."""
    tree_edges = []
    num_vertices = len(predecessors)

    for v in range(num_vertices):
        if v != start_node and predecessors[v] is not None:
            tree_edges.append((predecessors[v], v))

    
    # In an SPT rooted at `start_node`, the unique path from `start_node` to any vertex `v`
    # is the shortest path in the original graph.

    return tree_edges


# --- Task 3: MST (Kruskal and Prim) ---


def kruskal(graph):
    """Implementacja algorytmu Kruskala dla grafu ważonego."""
    num_vertices = len(graph)
    edges = []
    # Collect all edges with their weights
    for u in range(num_vertices):
        for v, weight in graph.get(u, []):
            if u < v:
                edges.append((weight, u, v))

    # Sort edges by weight
    edges.sort()

    mst = []
    dsu = DSU(num_vertices)
    total_weight = 0

    for weight, u, v in edges:
        # Check if adding this edge creates a cycle using DSU
        if dsu.union(u, v):
            mst.append((u, v, weight))
            total_weight += weight

    # Check if the MST spans all vertices (only possible if original graph was connected)
    # A valid MST for a connected graph will have num_vertices - 1 edges.
    if len(mst) != num_vertices - 1 and num_vertices > 0:
        print(
            "Warning: Kruskal's algorithm did not find a spanning tree (graph might be disconnected)."
        )

    return mst, total_weight


def prim(graph, start_node):
    """Implementacja algorytmu Prima dla grafu ważonego."""
    num_vertices = len(graph)
    # min_cost[v] is the minimum weight to connect vertex v to the current MST
    min_cost = {v: math.inf for v in range(num_vertices)}
    # parent[v] is the vertex in MST that v connects to with min_cost[v]
    parent = {v: None for v in range(num_vertices)}
    # In_mst[v] is true if vertex v is included in MST
    in_mst = {v: False for v in range(num_vertices)}

    # Priority queue: stores tuples (cost, vertex)
    # The cost here is the minimum weight edge connecting vertex to MST
    priority_queue = [(0, start_node)]
    min_cost[start_node] = 0

    mst_edges = []
    total_weight = 0
    edges_count = 0  # To stop when V-1 edges are found (for connected graph)

    while priority_queue and edges_count < num_vertices - 1:
        # Extract vertex with the minimum cost from the priority queue
        current_cost, u = heapq.heappop(priority_queue)

        # If already included in MST, skip
        if in_mst[u]:
            continue

        # Include vertex u in MST
        in_mst[u] = True

        # If u is not the start node and has a parent, add the edge to MST
        if parent[u] is not None:
            mst_edges.append((parent[u], u, min_cost[u]))
            total_weight += min_cost[u]
            edges_count += 1

        # Update min_cost and parent for adjacent vertices
        for v, weight in graph.get(u, []):
            if not in_mst[v] and weight < min_cost[v]:
                min_cost[v] = weight
                parent[v] = u
                # Add/update vertex in priority queue
                heapq.heappush(priority_queue, (min_cost[v], v))

    # Check if MST spans all vertices (only possible if original graph was connected)
    if edges_count != num_vertices - 1 and num_vertices > 0:
        print(
            "Warning: Prim's algorithm did not find a spanning tree (graph might be disconnected)."
        )

    return mst_edges, total_weight


def draw_graph_with_components(graph, components, title="Graf z składowymi spójnymi"):
    """
    Rysuje graf z zaznaczonymi składowymi spójnymi.
    Każda składowa ma inny kolor.
    """
    plt.figure(figsize=(12, 8))
    
    # Kolory dla różnych składowych
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    num_vertices = len(graph)
    
    # Generowanie pozycji wierzchołków w okręgu
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
    positions = {}
    radius = 3
    
    for i in range(num_vertices):
        x = radius * np.cos(angles[i])
        y = radius * np.sin(angles[i])
        positions[i] = (x, y)
    
    # Rysowanie krawędzi
    for node, neighbors in graph.items():
        x1, y1 = positions[node]
        for neighbor in neighbors:
            # Sprawdź czy neighbor to tuple (dla grafów ważonych) czy int (dla nieważonych)
            if isinstance(neighbor, tuple):
                neighbor_node = neighbor[0]
            else:
                neighbor_node = neighbor
                
            x2, y2 = positions[neighbor_node]
            
            # Rysuj krawędź tylko raz (unikaj duplikatów)
            if node < neighbor_node:
                plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.5, linewidth=1)
    
    # Rysowanie wierzchołków z kolorami składowych
    for comp_idx, component in enumerate(components):
        color = colors[comp_idx % len(colors)]
        for node in component:
            x, y = positions[node]
            plt.scatter(x, y, c=color, s=500, alpha=0.8, edgecolors='black', linewidth=2)
            plt.text(x, y, str(node), ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Legenda
    legend_elements = []
    for comp_idx, component in enumerate(components):
        color = colors[comp_idx % len(colors)]
        legend_elements.append(patches.Patch(color=color, label=f'Składowa {comp_idx + 1}: {component}'))
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Informacje o grafie
    plt.figtext(0.02, 0.02, f'Liczba wierzchołków: {num_vertices}\n'
                            f'Liczba składowych: {len(components)}\n'
                            f'Graf spójny: {"Tak" if len(components) == 1 else "Nie"}',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.show()


# --- Execute Tasks ---

if __name__ == "__main__":
    # --- Task 1 Execution ---
    print("--- Task 1: Connected Components ---")
    NUM_VERTICES_1 = 10
    EDGE_PROBABILITY_1 = 0.2  # Probability to make graph potentially disconnected

    # Generate a random unweighted graph (or weighted, doesn't matter for components,
    # but the find_connected_components needs to handle the list structure)
    # Let's use the unweighted generator as requested implicitly by context.
    graph_1 = generate_random_unweighted_graph(NUM_VERTICES_1, EDGE_PROBABILITY_1)
    print(
        f"Generated random graph with {NUM_VERTICES_1} vertices and edge probability {EDGE_PROBABILITY_1}:"
    )
    # Print graph structure
    for u, neighbors in graph_1.items():
        print(f"  {u}: {neighbors}")

    components = find_connected_components(graph_1)

    print("\nFound Connected Components:")
    for i, component in enumerate(components):
        print(f"  Component {i+1}: {component}")

    is_connected = len(components) == 1
    print(f"\nGraph is connected: {is_connected}")
    
    # Rysowanie grafu z składowymi spójnymi
    print("\nGenerating graph visualization...")
    draw_graph_with_components(
        graph_1, 
        components, 
        f"Graf z {NUM_VERTICES_1} wierzchołkami (prawdopodobieństwo krawędzi: {EDGE_PROBABILITY_1})"
    )
    
    print("-" * 30)  # Separator

    # --- Task 2 Execution ---
    print("\n--- Task 2: Dijkstra's Algorithm ---")
    NUM_VERTICES_2 = 8
    EDGE_PROBABILITY_2 = 0.4  # Higher probability for better chance of connectivity
    MAX_WEIGHT_2 = 10

    # Generate a random weighted graph
    graph_2 = generate_random_weighted_graph(
        NUM_VERTICES_2, EDGE_PROBABILITY_2, MAX_WEIGHT_2
    )
    print(
        f"Generated random weighted graph with {NUM_VERTICES_2} vertices and edge probability {EDGE_PROBABILITY_2}:"
    )
    # Print graph structure
    for u, neighbors in graph_2.items():
        print(f"  {u}: {neighbors}")

    # Basic Dijkstra: shortest path between two random nodes
    if NUM_VERTICES_2 > 1:
        start_node_2 = random.randint(0, NUM_VERTICES_2 - 1)
        end_node_2 = random.randint(0, NUM_VERTICES_2 - 1)
        while end_node_2 == start_node_2 and NUM_VERTICES_2 > 1:
            end_node_2 = random.randint(0, NUM_VERTICES_2 - 1)

        print(f"\nRunning Dijkstra from node {start_node_2} to node {end_node_2}:")
        path, distance, nodes_visited = dijkstra(graph_2, start_node_2, end_node_2)

        if distance == math.inf:
            print(f"  No path exists between {start_node_2} and {end_node_2}.")
            print(f"  Dijkstra visited {nodes_visited} nodes")
        else:
            print(f"  Dijkstra visited {nodes_visited} nodes")
            print()  # Add a blank line for better readability
            display_path_with_distances(graph_2, path)

        # A* Algorithm: Compare with Dijkstra for the same path
        print(f"\nRunning A* from node {start_node_2} to node {end_node_2}:")
        path_astar, distance_astar, nodes_explored_astar = a_star(graph_2, start_node_2, end_node_2)
        
        if distance_astar == math.inf:
            print(f"  No path exists between {start_node_2} and {end_node_2}.")
            print(f"  A* explored {nodes_explored_astar} nodes")
        else:
            print(f"  A* explored {nodes_explored_astar} nodes")
            print()  # Add a blank line for better readability
            display_path_with_distances(graph_2, path_astar)
            
        # Compare A* and Dijkstra results
        print(f"\nComparison of Dijkstra vs A*:")
        print(f"  Dijkstra path: {path if distance != math.inf else 'No path'}")
        print(f"  A* path:       {path_astar if distance_astar != math.inf else 'No path'}")
        print(f"  Dijkstra distance: {distance if distance != math.inf else 'Unreachable'}")
        print(f"  A* distance:       {distance_astar if distance_astar != math.inf else 'Unreachable'}")
        print(f"  Dijkstra nodes visited: {nodes_visited}")
        print(f"  A* nodes explored:      {nodes_explored_astar}")
        
        if distance != math.inf and distance_astar != math.inf:
            if abs(distance - distance_astar) < 0.001:  # Account for floating point precision
                print(f"  ✓ Both algorithms found the same shortest distance!")
                if path == path_astar:
                    print(f"  ✓ Both algorithms found the same path!")
                else:
                    print(f"  ✓ Different paths but same distance (multiple optimal paths exist)")
            else:
                print(f"  ✗ Warning: Different distances found! Check implementation.")

    else:
        print("Need more than 1 vertex to run Dijkstra.")

    print("-" * 30)  # Separator

    # --- Task 3 Execution ---
    print("\n--- Task 3: Kruskal's and Prim's Algorithms (MST) ---")

    # Define a sample weighted graph (must be connected for a single MST spanning all vertices)
    # Example graph from a common MST tutorial (e.g., similar to Wiki or standard texts)
    # Vertices: 0, 1, 2, 3, 4, 5, 6, 7, 8
    # Edges: (0,1,4), (0,7,8), (1,2,8), (1,7,11), (2,3,7), (2,8,2), (2,5,4), (3,4,9), (3,5,14), (4,5,10), (5,6,2), (6,7,1), (6,8,6), (7,8,7)
    NUM_VERTICES_3 = 9
    graph_3 = {
        0: [(1, 4), (7, 8)],
        1: [(0, 4), (2, 8), (7, 11)],
        2: [(1, 8), (3, 7), (8, 2), (5, 4)],
        3: [(2, 7), (4, 9), (5, 14)],
        4: [(3, 9), (5, 10)],
        5: [(2, 4), (3, 14), (4, 10), (6, 2)],
        6: [(5, 2), (7, 1), (8, 6)],
        7: [(0, 8), (1, 11), (6, 1), (8, 7)],
        8: [(2, 2), (6, 6), (7, 7)],
    }
    # Ensure all vertices from 0 to NUM_VERTICES_3-1 exist in the graph dict, even if isolated
    for i in range(NUM_VERTICES_3):
        if i not in graph_3:
            graph_3[i] = []

    print(f"Using sample weighted graph with {NUM_VERTICES_3} vertices.")
    # Print graph structure
    for u, neighbors in graph_3.items():
        print(f"  {u}: {neighbors}")

    # Run Kruskal's algorithm
    print("\nRunning Kruskal's Algorithm:")
    mst_kruskal, total_weight_kruskal = kruskal(graph_3)
    print("  MST Edges (u, v, weight):")
    for edge in mst_kruskal:
        print(f"    {edge}")
    print(f"  Total MST Weight: {total_weight_kruskal}")

    # Run Prim's algorithm (choose a start node, e.g., 0)
    start_node_3 = 0
    print(f"\nRunning Prim's Algorithm (starting from node {start_node_3}):")
    mst_prim, total_weight_prim = prim(graph_3, start_node_3)
    print("  MST Edges (parent, child, weight):")
    # Prim's output edges directed towards the new node, Kruskal's are just the edges
    # For comparison, list them sorted or just show the set. Let's just list.
    for edge in mst_prim:
        print(f"    {edge}")
    print(f"  Total MST Weight: {total_weight_prim}")

    # Verification check
    print(f"\nVerification:")
    print(f"  Kruskal's Total Weight: {total_weight_kruskal}")
    print(f"  Prim's Total Weight:    {total_weight_prim}")
    if total_weight_kruskal == total_weight_prim:
        print(
            "  Total weights match. MST algorithms results are consistent (for total weight)."
        )
    else:
        print(
            "  Warning: Total weights do NOT match. Check implementations or graph connectivity."
        )

    print("-" * 30)  # Separator
