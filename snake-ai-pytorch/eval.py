#!/usr/bin/env python3
"""
Evaluation script for Snake RL Agent
Runs episodes without training, with epsilon=0 (no exploration)
"""

import torch
import numpy as np
import csv
import argparse
import os
from datetime import datetime
from game import SnakeGameAI
from model import Linear_QNet
from agent import Agent
from config import EVAL_EPISODES_DEFAULT, SEED

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """Load a trained model"""
    model = Linear_QNet(11, 256, 3).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
        return model
    else:
        print(f"Model file {model_path} not found!")
        return None

def evaluate_agent(model, episodes=EVAL_EPISODES_DEFAULT, verbose=True):
    """
    Evaluate the agent for a given number of episodes
    
    Args:
        model: Trained neural network model
        episodes: Number of episodes to run
        verbose: Print progress
    
    Returns:
        dict: Evaluation metrics
    """
    # Create a temporary agent just to use its get_state method
    temp_agent = Agent()
    
    # Create game environment
    game = SnakeGameAI()
    
    # Metrics storage
    scores = []
    steps_list = []
    timeouts = 0
    
    for episode in range(episodes):
        game.reset()
        episode_steps = 0
        
        while True:
            # Get current state
            state = temp_agent.get_state(game)
            
            # Get action from model (no exploration)
            state_tensor = torch.tensor(state, dtype=torch.float).to(device)
            q_values = model(state_tensor)
            action_idx = torch.argmax(q_values).item()
            
            # Convert to action format
            action = [0, 0, 0]
            action[action_idx] = 1
            
            # Execute action
            reward, done, score, timed_out = game.play_step(action)
            episode_steps += 1
            
            if done:
                scores.append(score)
                steps_list.append(episode_steps)
                if timed_out:
                    timeouts += 1
                    
                if verbose and (episode + 1) % 10 == 0:
                    print(f"Episode {episode + 1}/{episodes}: Score {score}, Steps {episode_steps}")
                break
    
    # Calculate statistics
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    mean_steps = np.mean(steps_list)
    std_steps = np.std(steps_list)
    timeout_rate = timeouts / episodes
    
    results = {
        'episodes': episodes,
        'scores': scores,
        'steps': steps_list,
        'mean_score': mean_score,
        'std_score': std_score,
        'mean_steps': mean_steps,
        'std_steps': std_steps,
        'timeout_rate': timeout_rate,
        'timeouts': timeouts
    }
    
    return results

def save_eval_results(results, run_id):
    """Save evaluation results to CSV"""
    
    # Create eval.csv if it doesn't exist
    if not os.path.exists('eval.csv'):
        with open('eval.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['run_id', 'episodes', 'mean_score', 'std_score', 
                           'mean_steps', 'std_steps', 'timeout_rate'])
    
    # Append results
    with open('eval.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run_id,
            results['episodes'],
            results['mean_score'],
            results['std_score'],
            results['mean_steps'],
            results['std_steps'],
            results['timeout_rate']
        ])

def print_summary(results, model_path):
    """Print evaluation summary"""
    print("\n" + "="*50)
    print(f"EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {model_path}")
    print(f"Episodes: {results['episodes']}")
    print(f"Mean Score: {results['mean_score']:.2f} ± {results['std_score']:.2f}")
    print(f"Mean Steps: {results['mean_steps']:.1f} ± {results['std_steps']:.1f}")
    print(f"Timeout Rate: {results['timeout_rate']:.1%} ({results['timeouts']}/{results['episodes']})")
    print(f"Best Score: {max(results['scores'])}")
    print(f"Worst Score: {min(results['scores'])}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Snake RL Agent')
    parser.add_argument('--model', type=str, default='models/best_avg.pth',
                       help='Path to model file (default: models/best_avg.pth)')
    parser.add_argument('--episodes', type=int, default=EVAL_EPISODES_DEFAULT,
                       help=f'Number of episodes to evaluate (default: {EVAL_EPISODES_DEFAULT})')
    parser.add_argument('--seed', type=int, default=SEED,
                       help=f'Random seed for reproducibility (default: {SEED})')
    parser.add_argument('--verbose', action='store_true',
                       help='Print progress during evaluation')
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Try to load the specified model, fallback to model.pth
    model = load_model(args.model)
    if model is None:
        fallback_path = 'models/model.pth'
        print(f"Trying fallback model: {fallback_path}")
        model = load_model(fallback_path)
        if model is None:
            print("No trained model found! Please train a model first.")
            return
        args.model = fallback_path
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"Starting evaluation on {device}")
    print(f"Episodes: {args.episodes}")
    print(f"Seed: {args.seed}")
    
    # Run evaluation
    with torch.no_grad():
        results = evaluate_agent(model, args.episodes, args.verbose)
    
    # Generate run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    save_eval_results(results, run_id)
    
    # Print summary
    print_summary(results, args.model)
    
    print(f"\nResults saved to eval.csv with run_id: {run_id}")

if __name__ == '__main__':
    main()