"""
Beautiful terminal graph visualization utilities for lista5.py
"""

import math
from typing import List, Dict, Tuple, Union, Optional

# ANSI color codes for beautiful terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    # Regular colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'


def print_section_header(title: str, width: int = 60):
    """Print a beautiful section header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(width)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * width}{Colors.RESET}")


def print_subsection_header(title: str, width: int = 50):
    """Print a beautiful subsection header"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{'-' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{title.center(width)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{'-' * width}{Colors.RESET}")


def print_graph_structure(graph: Dict, is_weighted: bool = False):
    """Pretty print graph structure"""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}📊 Graph Structure:{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}")
    
    for vertex, edges in graph.items():
        if not edges:
            print(f"  {Colors.BRIGHT_GREEN}Node {vertex}:{Colors.RESET} {Colors.DIM}(isolated){Colors.RESET}")
        else:
            if is_weighted:
                edge_str = ", ".join([f"{Colors.CYAN}{neighbor}{Colors.RESET}({Colors.YELLOW}{weight}{Colors.RESET})" 
                                    for neighbor, weight in edges])
            else:
                edge_str = ", ".join([f"{Colors.CYAN}{neighbor}{Colors.RESET}" for neighbor in edges])
            
            print(f"  {Colors.BRIGHT_GREEN}Node {vertex}:{Colors.RESET} {edge_str}")


def print_connected_components(components: List[List[int]]):
    """Pretty print connected components"""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_MAGENTA}🔗 Connected Components:{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}")
    
    if len(components) == 1:
        print(f"  {Colors.GREEN}✓ Graph is connected!{Colors.RESET}")
        print(f"  {Colors.BRIGHT_WHITE}Single component: {Colors.CYAN}{components[0]}{Colors.RESET}")
    else:
        print(f"  {Colors.YELLOW}⚠ Graph has {len(components)} disconnected components{Colors.RESET}")
        for i, component in enumerate(components, 1):
            color = [Colors.CYAN, Colors.MAGENTA, Colors.YELLOW, Colors.GREEN, Colors.BLUE][i % 5]
            print(f"  {Colors.BRIGHT_WHITE}Component {i}:{Colors.RESET} {color}{component}{Colors.RESET}")


def print_dijkstra_path(path: List[int], distance: float, start: int, end: int):
    """Pretty print Dijkstra's shortest path result"""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}🛣️  Shortest Path (Dijkstra):{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}")
    
    if distance == math.inf:
        print(f"  {Colors.RED}✗ No path exists from {Colors.CYAN}{start}{Colors.RED} to {Colors.CYAN}{end}{Colors.RESET}")
    else:
        path_str = " → ".join([f"{Colors.CYAN}{Colors.BOLD}{node}{Colors.RESET}" for node in path])
        print(f"  {Colors.GREEN}✓ Path found:{Colors.RESET} {path_str}")
        print(f"  {Colors.BRIGHT_WHITE}Total distance:{Colors.RESET} {Colors.YELLOW}{Colors.BOLD}{distance}{Colors.RESET}")


def print_multi_source_distances(distances: Dict[int, float], sources: List[int]):
    """Pretty print multi-source Dijkstra results"""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}🌐 Multi-Source Distances:{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}")
    print(f"  {Colors.BRIGHT_WHITE}Sources: {Colors.CYAN}{sources}{Colors.RESET}")
    print(f"  {Colors.BRIGHT_WHITE}Distances to nearest source:{Colors.RESET}")
    
    for vertex in sorted(distances.keys()):
        dist = distances[vertex]
        if dist == math.inf:
            print(f"    {Colors.BRIGHT_GREEN}Node {vertex}:{Colors.RESET} {Colors.RED}Unreachable{Colors.RESET}")
        elif vertex in sources:
            print(f"    {Colors.BRIGHT_GREEN}Node {vertex}:{Colors.RESET} {Colors.CYAN}{Colors.BOLD}0 (source){Colors.RESET}")
        else:
            print(f"    {Colors.BRIGHT_GREEN}Node {vertex}:{Colors.RESET} {Colors.YELLOW}{dist}{Colors.RESET}")


def print_tree_analysis(tree_edges: List[Tuple], num_vertices: int, start_node: int):
    """Pretty print Dijkstra tree analysis"""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_MAGENTA}🌳 Shortest Path Tree Analysis:{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}")
    print(f"  {Colors.BRIGHT_WHITE}Root node:{Colors.RESET} {Colors.CYAN}{Colors.BOLD}{start_node}{Colors.RESET}")
    print(f"  {Colors.BRIGHT_WHITE}Tree edges:{Colors.RESET}")
    
    if not tree_edges:
        print(f"    {Colors.DIM}(No edges - isolated source or single vertex){Colors.RESET}")
    else:
        for parent, child in tree_edges:
            print(f"    {Colors.CYAN}{parent}{Colors.RESET} → {Colors.CYAN}{child}{Colors.RESET}")
    
    expected_edges = num_vertices - 1
    actual_edges = len(tree_edges)
    
    if actual_edges == expected_edges:
        print(f"  {Colors.GREEN}✓ Perfect spanning tree: {actual_edges}/{expected_edges} edges{Colors.RESET}")
    else:
        print(f"  {Colors.YELLOW}⚠ Partial tree: {actual_edges}/{expected_edges} edges (disconnected graph){Colors.RESET}")


def print_mst_result(mst_edges: List[Tuple], total_weight: float, algorithm_name: str):
    """Pretty print MST algorithm results"""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_GREEN}🌲 {algorithm_name} MST Result:{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}")
    
    if not mst_edges:
        print(f"  {Colors.RED}✗ No MST found (disconnected graph){Colors.RESET}")
        return
    
    print(f"  {Colors.BRIGHT_WHITE}MST Edges:{Colors.RESET}")
    for i, edge in enumerate(mst_edges, 1):
        if len(edge) == 3:  # (u, v, weight) format
            u, v, weight = edge
            print(f"    {i:2d}. {Colors.CYAN}{u}{Colors.RESET} ─({Colors.YELLOW}{weight}{Colors.RESET})─ {Colors.CYAN}{v}{Colors.RESET}")
        else:  # Handle other formats if needed
            print(f"    {i:2d}. {edge}")
    
    print(f"  {Colors.BRIGHT_WHITE}Total Weight:{Colors.RESET} {Colors.YELLOW}{Colors.BOLD}{total_weight}{Colors.RESET}")


def print_mst_comparison(weight1: float, weight2: float, name1: str = "Kruskal", name2: str = "Prim"):
    """Pretty print MST algorithm comparison"""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}⚖️  Algorithm Comparison:{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}")
    print(f"  {Colors.BRIGHT_WHITE}{name1} Total Weight:{Colors.RESET} {Colors.YELLOW}{weight1}{Colors.RESET}")
    print(f"  {Colors.BRIGHT_WHITE}{name2} Total Weight:{Colors.RESET}  {Colors.YELLOW}{weight2}{Colors.RESET}")
    
    if weight1 == weight2:
        print(f"  {Colors.GREEN}✓ Results match! Both algorithms found optimal MST{Colors.RESET}")
    else:
        print(f"  {Colors.RED}✗ Results differ! Check implementation or graph connectivity{Colors.RESET}")


def print_graph_stats(graph: Dict, is_weighted: bool = False):
    """Print beautiful graph statistics"""
    num_vertices = len(graph)
    num_edges = 0
    total_weight = 0
    
    for vertex, edges in graph.items():
        if is_weighted:
            num_edges += len(edges)
            total_weight += sum(weight for _, weight in edges)
        else:
            num_edges += len(edges)
    
    # For undirected graphs, each edge is counted twice
    num_edges //= 2
    if is_weighted:
        total_weight //= 2
    
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_WHITE}📈 Graph Statistics:{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 30}{Colors.RESET}")
    print(f"  {Colors.BRIGHT_WHITE}Vertices:{Colors.RESET} {Colors.CYAN}{num_vertices}{Colors.RESET}")
    print(f"  {Colors.BRIGHT_WHITE}Edges:{Colors.RESET} {Colors.CYAN}{num_edges}{Colors.RESET}")
    
    if is_weighted:
        print(f"  {Colors.BRIGHT_WHITE}Total Weight:{Colors.RESET} {Colors.YELLOW}{total_weight}{Colors.RESET}")
        if num_edges > 0:
            avg_weight = total_weight / num_edges
            print(f"  {Colors.BRIGHT_WHITE}Average Edge Weight:{Colors.RESET} {Colors.YELLOW}{avg_weight:.2f}{Colors.RESET}")
    
    # Density calculation
    max_edges = (num_vertices * (num_vertices - 1)) // 2
    if max_edges > 0:
        density = (num_edges / max_edges) * 100
        print(f"  {Colors.BRIGHT_WHITE}Graph Density:{Colors.RESET} {Colors.MAGENTA}{density:.1f}%{Colors.RESET}")


def print_success_message(message: str):
    """Print a success message"""
    print(f"\n{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_warning_message(message: str):
    """Print a warning message"""
    print(f"\n{Colors.YELLOW}⚠ {message}{Colors.RESET}")


def print_error_message(message: str):
    """Print an error message"""
    print(f"\n{Colors.RED}✗ {message}{Colors.RESET}")


def print_algorithm_info(algorithm: str, description: str):
    """Print algorithm information box"""
    print(f"\n{Colors.BOLD}{Colors.BG_BLUE}{Colors.WHITE} {algorithm} {Colors.RESET}")
    print(f"{Colors.DIM}{description}{Colors.RESET}")
