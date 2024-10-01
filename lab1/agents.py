import random
from pathfinding import a_star_search, dijkstra_search, bfs_search
from heuristics import heuristic_manhattan

class Agent:
    def __init__(self, position):
        self.position = position

    def move(self, new_position):
        self.position = new_position

class PacMan(Agent):
    def __init__(self, position, graph):
        super().__init__(position)
        self.graph = graph
        self.safety_distance = 3
        self.long_range_distance = 8
        self.last_positions = []  # Store last few positions
        self.memory_size = 5  # Number of past positions to remember

    def decide_move(self, graph, pellets, ghosts):
        nearby_ghosts = [ghost for ghost in ghosts if heuristic_manhattan(self.position, ghost.position) < self.safety_distance]
        long_range_ghosts = [ghost for ghost in ghosts if self.safety_distance <= heuristic_manhattan(self.position, ghost.position) < self.long_range_distance]

        if nearby_ghosts:
            move = self.handle_nearby_ghosts(graph, pellets, nearby_ghosts)
        elif long_range_ghosts:
            move = self.handle_long_range_ghosts(graph, pellets, long_range_ghosts)
        else:
            move = self.collect_nearest_pellet(graph, pellets)

        # Prevent oscillation
        if self.is_oscillating(move):
            possible_moves = list(graph[self.position])
            # Filter out moves that lead back to recent positions
            new_moves = [m for m in possible_moves if m not in self.last_positions]
            if new_moves:
                move = random.choice(new_moves)

        # Update memory of recent positions
        self.last_positions.append(self.position)
        if len(self.last_positions) > self.memory_size:
            self.last_positions.pop(0)

        return move

    def is_oscillating(self, move):
        # Check if the move leads to a position we've been in recently
        return move in self.last_positions

    def handle_nearby_ghosts(self, graph, pellets, nearby_ghosts):
        nearest_ghost = min(nearby_ghosts, key=lambda g: heuristic_manhattan(self.position, g.position))
        safe_moves = self.find_safe_moves(graph, nearest_ghost)
        
        if safe_moves:
            # Choose the safe move that's closest to a pellet and farthest from ghosts
            best_moves = [move for move in safe_moves if self.evaluate_move(move, pellets, nearby_ghosts) == 
                          max(self.evaluate_move(m, pellets, nearby_ghosts) for m in safe_moves)]
            return random.choice(best_moves)  # Add randomness to break ties
        else:
            # If no safe moves, try to maximize distance from the nearest ghost
            possible_moves = list(graph[self.position])
            best_moves = [move for move in possible_moves if heuristic_manhattan(move, nearest_ghost.position) == 
                          max(heuristic_manhattan(m, nearest_ghost.position) for m in possible_moves)]
            return random.choice(best_moves) if best_moves else self.position

    def handle_long_range_ghosts(self, graph, pellets, long_range_ghosts):
        possible_moves = list(graph[self.position])
        if not possible_moves:
            return self.position

        # Evaluate moves based on pellet proximity and ghost avoidance
        best_moves = [move for move in possible_moves if self.evaluate_move(move, pellets, long_range_ghosts) == 
                      max(self.evaluate_move(m, pellets, long_range_ghosts) for m in possible_moves)]
        return random.choice(best_moves)  # Add randomness to break ties

    def collect_nearest_pellet(self, graph, pellets):
        if pellets:
            nearest_pellet = min(pellets, key=lambda p: heuristic_manhattan(self.position, p))
            path = a_star_search(graph, self.position, nearest_pellet, heuristic=heuristic_manhattan)
            return path[1] if path and len(path) > 1 else self.position
        return self.position

    def find_safe_moves(self, graph, ghost):
        return [move for move in graph[self.position] if heuristic_manhattan(move, ghost.position) >= self.safety_distance]

    def evaluate_move(self, move, pellets, ghosts):
        # Higher score is better
        pellet_score = -min(heuristic_manhattan(move, p) for p in pellets) if pellets else 0
        ghost_score = min(heuristic_manhattan(move, g.position) for g in ghosts)
        return pellet_score + 3 * ghost_score  # Prioritize ghost avoidance

class Ghost(Agent):
    def __init__(self, position, graph, strategy='chase'):
        super().__init__(position)
        self.graph = graph
        self.strategy = strategy
        self.scatter_target = None  # We'll set this in get_scatter_target
        self.chase_time = 0
        self.scatter_time = 0
        self.max_chase_time = 20  # Seconds
        self.max_scatter_time = 7  # Seconds
        self.last_pacman_positions = []
        self.maze_dimensions = self.get_maze_dimensions()
        self.scatter_target = self.get_scatter_target()

    def get_maze_dimensions(self):
        # Find the maximum x and y coordinates in the graph
        max_x = max(pos[1] for pos in self.graph.nodes())
        max_y = max(pos[0] for pos in self.graph.nodes())
        return max_y + 1, max_x + 1  # Adding 1 because coordinates are 0-indexed

    def decide_move(self, pacman_position, game_time):
        self.update_mode(game_time)
        self.update_pacman_history(pacman_position)

        if self.strategy == 'scatter':
            return self.scatter_move()
        elif self.strategy == 'chase':
            return self.chase_move(pacman_position)
        elif self.strategy == 'ambush':
            return self.ambush_move(pacman_position)
        else:
            return self.random_move()

    def update_mode(self, game_time):
        if self.strategy == 'chase':
            self.chase_time += 1
            if self.chase_time >= self.max_chase_time:
                self.strategy = 'scatter'
                self.chase_time = 0
        elif self.strategy == 'scatter':
            self.scatter_time += 1
            if self.scatter_time >= self.max_scatter_time:
                self.strategy = 'chase'
                self.scatter_time = 0

    def update_pacman_history(self, pacman_position):
        self.last_pacman_positions.append(pacman_position)
        if len(self.last_pacman_positions) > 5:
            self.last_pacman_positions.pop(0)

    def scatter_move(self):
        path = a_star_search(self.graph, self.position, self.scatter_target, heuristic=heuristic_manhattan)
        if path and len(path) > 1:
            return path[1]
        return self.random_move()

    def chase_move(self, pacman_position):
        path = a_star_search(self.graph, self.position, pacman_position, heuristic=heuristic_manhattan)
        if path and len(path) > 1:
            return path[1]
        return self.random_move()

    def ambush_move(self, pacman_position):
        predicted_position = self.predict_pacman_move(pacman_position)
        path = a_star_search(self.graph, self.position, predicted_position, heuristic=heuristic_manhattan)
        if path and len(path) > 1:
            return path[1]
        return self.random_move()

    def random_move(self):
        possible_moves = list(self.graph.neighbors(self.position))
        return random.choice(possible_moves) if possible_moves else self.position

    def predict_pacman_move(self, pacman_position):
        return pacman_position
        # if len(self.last_pacman_positions) < 2:
        #     return pacman_position

        # last_pos = self.last_pacman_positions[-2]
        # current_pos = self.last_pacman_positions[-1]
        # dx = current_pos[0] - last_pos[0]
        # dy = current_pos[1] - last_pos[1]

        # predicted_x = current_pos[0] + dx
        # predicted_y = current_pos[1] + dy

        # # Check if the predicted position is valid (exists in the graph)
        # if (predicted_x, predicted_y) in self.graph:
        #     return (predicted_x, predicted_y)
        # else:
        #     # If predicted position is invalid, return current position
        #     return current_pos

    def get_scatter_target(self):
        height, width = self.maze_dimensions
        corners = [(1, 1), (1, width-2), (height-2, 1), (height-2, width-2)]
        valid_corners = [corner for corner in corners if corner in self.graph]
        return random.choice(valid_corners) if valid_corners else self.position