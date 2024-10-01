# pathfinding.py
import networkx as nx
from heuristics import heuristic_manhattan, heuristic_euclidean

def maze_to_graph(maze):
    G = nx.Graph()
    height, width = maze.shape
    for y in range(height):
        for x in range(width):
            if maze[y, x] == 1:
                G.add_node((y, x))
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx_ < width and maze[ny, nx_] == 1:
                        G.add_edge((y, x), (ny, nx_))
    return G

def a_star_search(graph, start, goal, heuristic=heuristic_manhattan):
    try:
        path = nx.astar_path(graph, start, goal, heuristic=heuristic)
        return path
    except nx.NetworkXNoPath:
        return []

def dijkstra_search(graph, start, goal):
    try:
        path = nx.dijkstra_path(graph, start, goal)
        return path
    except nx.NetworkXNoPath:
        return []

def bfs_search(graph, start, goal):
    try:
        path = nx.shortest_path(graph, start, goal)
        return path
    except nx.NetworkXNoPath:
        return []
