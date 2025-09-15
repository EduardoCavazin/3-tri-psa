import random
from enum import Enum
from collections import namedtuple, deque
import numpy as np
from config import TIMEOUT_FACTOR

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20

# Loop detection constants
LOOP_HISTORY_SIZE = 30
LOOP_THRESHOLD = 0.8  # 80% repetition

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.timed_out = False
        
        # Behavioral tracking
        self.move_history = deque(maxlen=LOOP_HISTORY_SIZE)
        self.loop_detected = False
        
        # Efficiency tracking
        self.food_events = 0
        self.steps_since_food = 0
        self.steps_per_food = []

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # Track movement for loop detection
        self._track_movement(action)
        
        # 3. check if game over
        reward = 0
        game_over = False
        
        # Check for collision
        if self.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score, self.timed_out
        
        # Check for timeout
        if self.frame_iteration > TIMEOUT_FACTOR * len(self.snake):
            game_over = True
            reward = -10
            self.timed_out = True
            return reward, game_over, self.score, self.timed_out

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            
            # Track food efficiency
            self.food_events += 1
            self.steps_per_food.append(self.steps_since_food)
            self.steps_since_food = 0
        else:
            self.snake.pop()
            self.steps_since_food += 1
        
        # 6. return game over and score
        return reward, game_over, self.score, self.timed_out

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
        
    def _track_movement(self, action):
        """Track movement patterns to detect loops"""
        # Convert action to direction for tracking
        if np.array_equal(action, [1, 0, 0]):      # straight
            move_type = 'S'
        elif np.array_equal(action, [0, 1, 0]):    # right
            move_type = 'R'
        else:                                       # left
            move_type = 'L'
            
        self.move_history.append(move_type)
        
        # Check for loops if we have enough history
        if len(self.move_history) >= LOOP_HISTORY_SIZE:
            self._detect_loop()
    
    def _detect_loop(self):
        """Detect if the agent is stuck in a loop"""
        moves = list(self.move_history)
        total_moves = len(moves)
        
        # Check for simple repetitive patterns
        
        # 1. Check for "always straight" (boring behavior)
        straight_count = moves.count('S')
        if straight_count / total_moves > LOOP_THRESHOLD:
            self.loop_detected = True
            return
        
        # 2. Check for cyclic patterns (2, 3, 4 move cycles)
        for cycle_length in [2, 3, 4]:
            if self._check_cyclic_pattern(moves, cycle_length):
                self.loop_detected = True
                return
                
        # 3. Check for alternating patterns (L-R-L-R)
        if self._check_alternating_pattern(moves):
            self.loop_detected = True
            return
    
    def _check_cyclic_pattern(self, moves, cycle_length):
        """Check if moves follow a cyclic pattern"""
        if len(moves) < cycle_length * 3:  # Need at least 3 repetitions
            return False
            
        # Extract potential pattern
        pattern = moves[-cycle_length:]
        
        # Check how many times this pattern repeats
        repetitions = 0
        for i in range(len(moves) - cycle_length, -1, -cycle_length):
            if moves[i:i+cycle_length] == pattern:
                repetitions += 1
            else:
                break
                
        # If pattern repeats enough times, it's a loop
        return repetitions >= 3 and (repetitions * cycle_length) / len(moves) > LOOP_THRESHOLD
    
    def _check_alternating_pattern(self, moves):
        """Check for simple alternating patterns like L-R-L-R"""
        if len(moves) < 8:  # Need reasonable sample
            return False
            
        # Count alternating L-R or R-L patterns
        alternating_count = 0
        for i in range(len(moves) - 1):
            if (moves[i] == 'L' and moves[i+1] == 'R') or (moves[i] == 'R' and moves[i+1] == 'L'):
                alternating_count += 1
                
        return alternating_count / (len(moves) - 1) > LOOP_THRESHOLD
    
    def get_mean_steps_per_food(self):
        """Calculate mean steps per food for this episode"""
        if not self.steps_per_food:
            return -1  # No food collected
        return sum(self.steps_per_food) / len(self.steps_per_food)