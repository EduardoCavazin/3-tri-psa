import torch
import random
import numpy as np
import csv
import os
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot, plot_extended, save_plots
from config import *

# Reproducibility setup
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = EPS0  # start with max exploration
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3).to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # Metrics tracking
        self.scores = []
        self.best_avg = 0
        
        # Initialize CSV logging
        self.init_csv_logging()


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)
        
    def init_csv_logging(self):
        """Initialize CSV file for logging metrics"""
        if not os.path.exists('metrics.csv'):
            with open('metrics.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['episode', 'score', 'avg100', 'steps', 'epsilon', 'timeouts', 'train_loss'])
    
    def moving_average(self, window=100):
        """Calculate moving average of scores"""
        if len(self.scores) < window:
            return sum(self.scores) / len(self.scores) if self.scores else 0
        return sum(self.scores[-window:]) / window
    
    def log_episode(self, score, steps, timed_out, train_loss):
        """Log episode metrics to CSV"""
        self.scores.append(score)
        avg100 = self.moving_average(100)
        
        # Check if we have a new best average
        if avg100 > self.best_avg:
            self.best_avg = avg100
            self.model.save('best_avg.pth')
            print(f"New best average: {avg100:.2f}, saved to best_avg.pth")
        
        # Log to CSV
        with open('metrics.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.n_games, score, avg100, steps, self.epsilon, int(timed_out), train_loss])
        
        # Checkpoint every N episodes
        if self.n_games % CHECKPOINT_EVERY == 0:
            self.model.save(f'checkpoint_ep{self.n_games}.pth')
            save_plots(self.n_games)
            print(f"Checkpoint saved at episode {self.n_games}")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        return loss

    def train_short_memory(self, state, action, reward, next_state, done):
        loss = self.trainer.train_step(state, action, reward, next_state, done)
        return loss

    def get_action(self, state, eval_mode=False):
        # Exponential epsilon decay
        if not eval_mode:
            self.epsilon = max(EPS_MIN, EPS0 * (DECAY ** self.n_games))
        else:
            self.epsilon = 0  # No exploration in eval mode
            
        final_move = [0,0,0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    plot_epsilons = []
    plot_steps = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score, timed_out = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        short_loss = agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            long_loss = agent.train_long_memory()
            train_loss = long_loss if long_loss is not None else -1
            
            # Reset game
            steps = game.frame_iteration
            game.reset()
            agent.n_games += 1

            # Update record
            if score > record:
                record = score
                agent.model.save()

            # Log episode
            agent.log_episode(score, steps, timed_out, train_loss)
            
            print(f'Game {agent.n_games}, Score: {score}, Record: {record}, Avg100: {agent.moving_average(100):.2f}, Epsilon: {agent.epsilon:.3f}')

            # Update plots
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot_epsilons.append(agent.epsilon)
            plot_steps.append(steps)
            
            # Extended plotting every 100 episodes
            if agent.n_games % 100 == 0:
                plot_extended(plot_scores, plot_mean_scores, plot_epsilons, plot_steps)
            else:
                plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    print(f"Starting training on {device}")
    print(f"Hyperparameters: EPS0={EPS0}, DECAY={DECAY}, EPS_MIN={EPS_MIN}")
    train()