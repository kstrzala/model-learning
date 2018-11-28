import numpy as np
from queue import Queue


def generate_maze(height=5, width=5, prob=0.6):
    """
    Generate valid maze (i.e. nontrivial maze with existing solution) of a given size.
    :return: valid maze
    """
    while True:
        maze = Maze(height, width, prob)
        if maze.valid():
            trajectory = maze.run_optimal_trajectory(form='coordinates', probs=False)
            if len(trajectory)>1:
                maze.reset()
                return maze


def define_probs(up, down, left, right):
    assert (up+down+left+right == 1)
    return np.array([up, down, left, right])


def square_maze_bfs(maze, start=None):
    if start is None:
        start = maze.goal
    mazetab = np.zeros((maze.height, maze.width), dtype=np.int8)
    q = Queue()
    q.put(start)
    soltab = [[None for x in range(maze.width)] for y in range(maze.height)]
    soltab[start[0]][start[1]] = define_probs(0.25, 0.25, 0.25, 0.25)
    while not (q.empty()):
        x,y = q.get()
        mazetab[x,y] = 1
        candidates = []
        if maze.rows[x, y]==1:
            candidates.append(((x, y-1), define_probs(0, 0, 0, 1)))
        if maze.rows[x, y+1]==1:
            candidates.append(((x, y+1), define_probs(0, 0, 1, 0)))
        if maze.columns[x, y]==1:
            candidates.append(((x-1, y), define_probs(0, 1, 0, 0)))
        if maze.columns[x+1, y]==1:
            candidates.append(((x+1, y), define_probs(1, 0, 0, 0)))
        for cand, probs in candidates:
            if mazetab[cand]==0:
                mazetab[cand] = 2
                soltab[cand[0]][cand[1]] = probs
                q.put(cand)
    return mazetab, soltab


class Maze():
    def __init__(self, n, m, p=0.7):
        self.height = n
        self.width = m
        self.binomial_p = p
        self.rows = self._init_rows(self.height, self.width, self.binomial_p)
        self.columns = self._init_cols(self.height, self.width, self.binomial_p)
        self.start = [np.random.randint(0, n), np.random.randint(0, m)]
        self.position = self.start.copy()
        self.goal = [np.random.randint(0, n), np.random.randint(0, m)]
    
    def _init_rows(self, n, m, p):
        rows = np.random.binomial(1, p, size=(n, m-1))
        return np.pad(rows, [(0,0), (1,1)], mode='constant', constant_values=0)
        
    def _init_cols(self, n, m, p):
        cols = np.random.binomial(1, p, size=(n-1, m))
        return np.pad(cols, [(1,1), (0,0)], mode='constant', constant_values=0)
    
    def valid(self):
        if self.start == self.goal:
            return False
        endmaze, solution = square_maze_bfs(self)
        return endmaze[tuple(self.start)] == 1
    
    def get_maze_image(self, fieldsize=8, barsize=2, show_position=False):
        size_func = lambda x: x*(fieldsize + barsize) + barsize
        im = np.zeros((size_func(self.height),size_func(self.width),3),dtype=np.uint8)
        for i in range(self.rows.shape[0]):
            for j in range(self.rows.shape[1]):
                if self.rows[i,j] == 0:
                    im[i*(fieldsize+barsize):(i+1)*(fieldsize+barsize)+barsize,
                       j*(fieldsize+barsize):j*(fieldsize+barsize)+barsize 
                       ,:] = 255
        for i in range(self.columns.shape[0]):
            for j in range(self.columns.shape[1]):
                if self.columns[i,j] == 0:
                    im[i*(fieldsize+barsize):i*(fieldsize+barsize)+barsize,
                       j*(fieldsize+barsize):(j+1)*(fieldsize+barsize)+barsize 
                       ,:] = 255

        im[(fieldsize + barsize) * self.goal[0] + barsize + 2:(fieldsize + barsize) * (self.goal[0] + 1) - 1,
        (fieldsize + barsize) * self.goal[1] + barsize + 2:(fieldsize + barsize) * (self.goal[1] + 1) - 1,
        1] = 255

        if show_position:
            im[(fieldsize+barsize)*self.position[0] + barsize + 2:(fieldsize+barsize)*(self.position[0]+1) -1,
               (fieldsize+barsize)*self.position[1] + barsize + 2:(fieldsize+barsize)*(self.position[1]+1) -1,
                0] = 255

        return im

    def get_position(self):
        return np.array(self.position)


    def get_solution(self):
        _, solution = square_maze_bfs(self)
        return solution
    
    def move(self, command):
        if command=='up':
            if self.columns[self.position[0], self.position[1]] == 1:
                self.position[0] -= 1
        elif command=='down':
            if self.columns[self.position[0]+1, self.position[1]] == 1:
                self.position[0] += 1
        elif command=='left':
            if self.rows[self.position[0], self.position[1]] == 1:
                self.position[1] -= 1
        elif command=='right':
            if self.rows[self.position[0], self.position[1]+1] == 1:
                self.position[1] += 1
                
    def reset(self):
        self.position = self.start.copy()
        
    def teleport(self, position):
        self.position = position
        return self
        
    def run_policy(self, policy, steps=50, probas=False):
        visited_states = []
        if probas:
            solution = self.get_solution()
            problist = []
            solution_list = []
        for i in range(steps):
            p = policy(self)
            move = np.random.choice(['up', 'down', 'left', 'right'], p=p)
            visited_states.append(self.position.copy())
            if probas:
                problist.append(p.copy())
                solution_list.append(solution[self.position[0]][self.position[1]])
            self.move(move)
            if self.position == self.goal:
                break
        if probas:
            return visited_states, problist, solution_list
        return visited_states
    
    def trajectory_to_images(self, trajectory):
        images = [self.teleport(t).get_maze_image() for t in trajectory]
        return np.stack(images).astype(np.float32)


    def trajectory_to_numpy(self, trajectory):
        np_traj = [np.array(t) for t in trajectory]
        return np.stack(np_traj).astype(np.float32)


    def run_optimal_trajectory(self, form='images', probs=True):
        solution = self.get_solution()
        trajectory = []
        problist = []
        while self.position != self.goal:
            trajectory.append(self.position.copy())
            prob = solution[self.position[0]][self.position[1]]
            problist.append(prob.copy())
            move = np.random.choice(['up', 'down', 'left', 'right'], p=prob)
            self.move(move)
        if form == 'images':
            return_value = [self.teleport(t).get_maze_image() for t in trajectory]
        elif form == 'coordinates':
            return_value = trajectory
        else:
            raise RuntimeError('Wrong form')
        if probs:
            return return_value, problist
        return return_value