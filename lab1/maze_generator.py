import random
import numpy as np

def generate_maze(width, height, level=1):
    # Start with a grid full of walls (0)
    maze = np.zeros((height, width), dtype=int)
    
    # Create a grid of rooms
    for i in range(1, height-1, 2):
        for j in range(1, width-1, 2):
            maze[i, j] = 1
    
    # Recursive backtracking to create paths
    stack = [(1, 1)]
    visited = set()
    
    while stack:
        current = stack[-1]
        visited.add(current)
        neighbors = get_unvisited_neighbors(current, maze, visited)
        
        if neighbors:
            next_cell = random.choice(neighbors)
            remove_wall(maze, current, next_cell)
            stack.append(next_cell)
        else:
            stack.pop()
    
    # Add difficulty based on level
    add_difficulty(maze, level)
    
    return maze

def get_unvisited_neighbors(cell, maze, visited):
    neighbors = []
    directions = [(-2,0), (2,0), (0,-2), (0,2)]
    for d in directions:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1]:
            if neighbor not in visited and maze[neighbor] == 1:
                neighbors.append(neighbor)
    return neighbors

def remove_wall(maze, cell1, cell2):
    x1, y1 = cell1
    x2, y2 = cell2
    maze[(x1 + x2) // 2, (y1 + y2) // 2] = 1

def add_difficulty(maze, level):
    height, width = maze.shape
    
    # Adjust difficulty based on level
    if level <= 3:
        # For levels 1-3, make the maze very easy
        num_extra_paths = width * 20
    elif level <= 5:
        # For levels 4-5, slightly increase difficulty
        num_extra_paths = width * 5# * 10
    else:
        # For levels 6 and above, use the original difficulty scaling
        num_extra_paths = width#min(level * 2, (height * width) // 20)
    
    for _ in range(num_extra_paths):
        x = random.randint(1, height-2)
        y = random.randint(1, width-2)
        if maze[x, y] == 0:
            maze[x, y] = 1
    
    # Only add teleporters for higher levels
    #if level > 7:
    #    add_teleporters(maze, min(level - 7, 3))

def add_teleporters(maze, num_teleporters):
    height, width = maze.shape
    for _ in range(num_teleporters):
        x1, y1 = random.randint(1, height-2), random.randint(1, width-2)
        x2, y2 = random.randint(1, height-2), random.randint(1, width-2)
        if maze[x1, y1] == 1 and maze[x2, y2] == 1:
            maze[x1, y1] = 2  # Mark as teleporter
            maze[x2, y2] = 2  # Mark as teleporter

# Example usage
if __name__ == "__main__":
    for level in range(1, 11):
        print(f"Level {level} maze:")
        maze = generate_maze(20, 15, level=level)
        
        # Print the maze with characters for better visibility
        for row in maze:
            print(''.join(['█' if cell == 0 else '·' if cell == 1 else 'T' for cell in row]))
        print("\n")