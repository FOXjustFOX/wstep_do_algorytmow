import random
import heapq
import math
from pretty_graphs import *

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

    while priority_queue:
        current_distance, u = heapq.heappop(priority_queue)

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

    # If end_node is specified, reconstruct path
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
        # Return path and distance if path exists, otherwise indicate no path
        if path and path[0] == start_node:
            return path, distances[end_node]
        else:
            return [], math.inf  # Indicate no path found

    # If end_node is None, return all distances and predecessors
    return distances, predecessors


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

    # Check basic tree properties: V-1 edges for V vertices, connected (if graph was connected)
    # The key property is that the path from the start_node to any other node v,
    # formed by following the predecessor pointers from v back to start_node,
    # is the shortest path found by Dijkstra. The set of these paths forms the Shortest Path Tree (SPT).
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


# --- Execute Tasks ---

if __name__ == "__main__":
    # --- Task 1 Execution ---
    print_section_header("Task 1: Connected Components")
    NUM_VERTICES_1 = 10
    EDGE_PROBABILITY_1 = 0.2  # Probability to make graph potentially disconnected

    # Generate a random unweighted graph (or weighted, doesn't matter for components,
    # but the find_connected_components needs to handle the list structure)
    # Let's use the unweighted generator as requested implicitly by context.
    graph_1 = generate_random_unweighted_graph(NUM_VERTICES_1, EDGE_PROBABILITY_1)
    
    print_algorithm_info("Connected Components", 
                        "Finding all connected components in an unweighted graph using DFS")
    
    print(f"\n{Colors.BRIGHT_WHITE}Generated random graph with {Colors.CYAN}{NUM_VERTICES_1}{Colors.RESET} vertices and edge probability {Colors.YELLOW}{EDGE_PROBABILITY_1}{Colors.RESET}")
    
    # Print graph structure and stats
    print_graph_structure(graph_1, is_weighted=False)
    print_graph_stats(graph_1, is_weighted=False)

    components = find_connected_components(graph_1)
    print_connected_components(components)

    # --- Task 2 Execution ---
    print_section_header("Task 2: Dijkstra's Algorithm")
    NUM_VERTICES_2 = 8
    EDGE_PROBABILITY_2 = 0.4  # Higher probability for better chance of connectivity
    MAX_WEIGHT_2 = 10

    # Generate a random weighted graph
    graph_2 = generate_random_weighted_graph(
        NUM_VERTICES_2, EDGE_PROBABILITY_2, MAX_WEIGHT_2
    )
    
    print_algorithm_info("Dijkstra's Algorithm", 
                        "Finding shortest paths in weighted graphs with non-negative edge weights")
    
    print(f"\n{Colors.BRIGHT_WHITE}Generated random weighted graph with {Colors.CYAN}{NUM_VERTICES_2}{Colors.RESET} vertices and edge probability {Colors.YELLOW}{EDGE_PROBABILITY_2}{Colors.RESET}")
    
    # Print graph structure and stats
    print_graph_structure(graph_2, is_weighted=True)
    print_graph_stats(graph_2, is_weighted=True)

    # Basic Dijkstra: shortest path between two random nodes
    if NUM_VERTICES_2 > 1:
        start_node_2 = random.randint(0, NUM_VERTICES_2 - 1)
        end_node_2 = random.randint(0, NUM_VERTICES_2 - 1)
        while end_node_2 == start_node_2 and NUM_VERTICES_2 > 1:
            end_node_2 = random.randint(0, NUM_VERTICES_2 - 1)

        print_subsection_header("Single Source Shortest Path")
        path, distance = dijkstra(graph_2, start_node_2, end_node_2)
        print_dijkstra_path(path, distance, start_node_2, end_node_2)

        # Multi-source Dijkstra: shortest distance from nearest source
        num_sources = max(1, NUM_VERTICES_2 // 4)  # Choose a few random sources
        start_nodes_2 = random.sample(range(NUM_VERTICES_2), num_sources)
        
        print_subsection_header("Multi-Source Shortest Distances")
        distances_multi, predecessors_multi = dijkstra_multi_source(
            graph_2, start_nodes_2
        )
        print_multi_source_distances(distances_multi, start_nodes_2)

        # Analyze Dijkstra's tree structure (using the first basic run's results)
        print_subsection_header("Shortest Path Tree Analysis")
        if NUM_VERTICES_2 > 0:
            _, predecessors_single = dijkstra(graph_2, start_node_2)
            dijkstra_tree_edges = analyze_dijkstra_tree(
                predecessors_single, start_node_2
            )
            print_tree_analysis(dijkstra_tree_edges, NUM_VERTICES_2, start_node_2)

    else:
        print_error_message("Need more than 1 vertex to run Dijkstra.")

    # --- Task 3 Execution ---
    print_section_header("Task 3: Minimum Spanning Tree (MST)")

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

    print_algorithm_info("Minimum Spanning Tree", 
                        "Finding MST using Kruskal's and Prim's algorithms")
    
    print(f"\n{Colors.BRIGHT_WHITE}Using sample weighted graph with {Colors.CYAN}{NUM_VERTICES_3}{Colors.RESET} vertices.")
    
    # Print graph structure and stats
    print_graph_structure(graph_3, is_weighted=True)
    print_graph_stats(graph_3, is_weighted=True)

    # Run Kruskal's algorithm
    print_subsection_header("Kruskal's Algorithm")
    mst_kruskal, total_weight_kruskal = kruskal(graph_3)
    print_mst_result(mst_kruskal, total_weight_kruskal, "Kruskal's")

    # Run Prim's algorithm (choose a start node, e.g., 0)
    start_node_3 = 0
    print_subsection_header(f"Prim's Algorithm (starting from node {start_node_3})")
    mst_prim, total_weight_prim = prim(graph_3, start_node_3)
    print_mst_result(mst_prim, total_weight_prim, "Prim's")

    # Verification check
    print_subsection_header("Algorithm Comparison")
    print_mst_comparison(total_weight_kruskal, total_weight_prim)
    
    print(f"\n{Colors.BRIGHT_CYAN}🎯 Both algorithms solve the same problem but with different approaches:{Colors.RESET}")
    print(f"  {Colors.WHITE}• Kruskal's: Edge-based, sorts all edges and uses Union-Find{Colors.RESET}")
    print(f"  {Colors.WHITE}• Prim's: Vertex-based, grows tree from a starting vertex{Colors.RESET}")
    
    print_success_message("All graph algorithms completed successfully!")
