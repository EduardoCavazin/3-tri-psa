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
    loops_detected = 0
    food_events_list = []
    steps_per_food_list = []
    
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
                if game.loop_detected:
                    loops_detected += 1
                    
                food_events_list.append(game.food_events)
                steps_per_food = game.get_mean_steps_per_food()
                steps_per_food_list.append(steps_per_food if steps_per_food != -1 else 0)
                    
                if verbose and (episode + 1) % 10 == 0:
                    print(f"Episode {episode + 1}/{episodes}: Score {score}, Steps {episode_steps}, Food: {game.food_events}, Loop: {game.loop_detected}")
                break
    
    # Calculate statistics
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    mean_steps = np.mean(steps_list)
    std_steps = np.std(steps_list)
    timeout_rate = timeouts / episodes
    loop_rate = loops_detected / episodes
    mean_food_events = np.mean(food_events_list)
    mean_steps_per_food = np.mean([x for x in steps_per_food_list if x > 0]) if any(x > 0 for x in steps_per_food_list) else 0
    
    results = {
        'episodes': episodes,
        'scores': scores,
        'steps': steps_list,
        'mean_score': mean_score,
        'std_score': std_score,
        'mean_steps': mean_steps,
        'std_steps': std_steps,
        'timeout_rate': timeout_rate,
        'timeouts': timeouts,
        'loop_rate': loop_rate,
        'loops_detected': loops_detected,
        'mean_food_events': mean_food_events,
        'mean_steps_per_food': mean_steps_per_food
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
    print("\n" + "="*60)
    print(f"BEHAVIORAL EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Episodes: {results['episodes']}")
    print()
    print("PERFORMANCE METRICS:")
    print(f"  Mean Score: {results['mean_score']:.2f} ± {results['std_score']:.2f}")
    print(f"  Best Score: {max(results['scores'])}")
    print(f"  Worst Score: {min(results['scores'])}")
    print(f"  Mean Steps: {results['mean_steps']:.1f} ± {results['std_steps']:.1f}")
    print()
    print("BEHAVIORAL ANALYSIS:")
    print(f"  Timeout Rate: {results['timeout_rate']:.1%} ({results['timeouts']}/{results['episodes']})")
    print(f"  Loop Detection Rate: {results['loop_rate']:.1%} ({results['loops_detected']}/{results['episodes']})")
    print(f"  Mean Food Events: {results['mean_food_events']:.1f}")
    print(f"  Mean Steps per Food: {results['mean_steps_per_food']:.1f}")
    print()
    print("BEHAVIORAL HEALTH CHECK:")
    if results['loop_rate'] > 0.3:
        print("    WARNING: High loop detection rate - agent may be stuck in repetitive patterns")
    elif results['loop_rate'] > 0.1:
        print("    CAUTION: Some loop behavior detected")
    else:
        print("   GOOD: Low loop behavior")
        
    if results['timeout_rate'] > 0.5:
        print("    WARNING: High timeout rate - agent may be inefficient")
    elif results['timeout_rate'] > 0.2:
        print("    CAUTION: Some timeouts detected")
    else:
        print("   GOOD: Low timeout rate")
        
    if results['mean_steps_per_food'] > 50:
        print("    WARNING: Inefficient food collection")
    elif results['mean_steps_per_food'] > 30:
        print("    CAUTION: Moderate food collection efficiency")
    else:
        print("   GOOD: Efficient food collection")
    print("="*60)

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