#!/usr/bin/env python3
"""
Experiment Runner for Snake AI - Automated Hyperparameter Ablation
================================================================

This script automates training runs with different hyperparameter configurations
for systematic comparison and analysis. Each run is isolated with its own
output directories and metrics collection.

Usage:
    python run_experiments.py --episodes 500
    python run_experiments.py --episodes 300 --max-runs 2 --base-seed 123
    python run_experiments.py --configs-file configs/ablations.json --episodes 400
"""

import argparse
import json
import os
import sys
import csv
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
import traceback

# Import project modules
from config import EXPERIMENT_CONFIGS, EXPERIMENT_EPISODES


def generate_run_id():
    """Generate unique RUN_ID: YYYYMMDD_HHMMSS_<hash4>"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a small hash for uniqueness
    hash_input = f"{timestamp}_{os.getpid()}"
    hash4 = hashlib.md5(hash_input.encode()).hexdigest()[:4]
    return f"{timestamp}_{hash4}"


def load_configs_from_file(config_file):
    """Load experiment configurations from JSON file"""
    try:
        with open(config_file, 'r') as f:
            configs = json.load(f)
        print(f"Loaded {len(configs)} configurations from {config_file}")
        return configs
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        sys.exit(1)


def setup_run_directories(run_id):
    """Create directory structure for a run"""
    dirs = {
        'run_dir': f"runs/{run_id}",
        'plots_dir': f"runs/{run_id}/plots",
        'models_dir': f"models/{run_id}"
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def save_run_config(run_id, config, seed, episodes):
    """Save effective configuration for this run"""
    effective_config = {
        "run_id": run_id,
        "seed": seed,
        "episodes": episodes,
        **config
    }

    config_path = f"runs/{run_id}/config.json"
    with open(config_path, 'w') as f:
        json.dump(effective_config, f, indent=2)

    print(f"Saved config to {config_path}")


def apply_config_overrides(config):
    """Apply configuration overrides to global config variables"""
    import config as cfg

    # Store original values for restoration
    original_values = {}

    for key, value in config.items():
        if hasattr(cfg, key):
            original_values[key] = getattr(cfg, key)
            setattr(cfg, key, value)
            print(f"  {key}: {getattr(cfg, key)}")

    return original_values


def restore_config_values(original_values):
    """Restore original configuration values"""
    import config as cfg

    for key, value in original_values.items():
        setattr(cfg, key, value)


def setup_metrics_csv(run_id):
    """Initialize metrics.csv for this run with proper header"""
    metrics_path = f"runs/{run_id}/metrics.csv"

    # Create the CSV with the same header as in agent.py
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'score', 'avg100', 'steps', 'epsilon', 'timeouts', 'train_loss',
                        'a_straight', 'a_left', 'a_right', 'random_actions', 'loop_detected',
                        'food_events', 'mean_steps_per_food'])

    return metrics_path


def run_training(run_id, config, seed, episodes):
    """Execute training for a single configuration"""
    print(f"\n[START] Starting training run: {run_id}")
    print(f"Configuration: {config}")
    print(f"Seed: {seed}, Episodes: {episodes}")

    # Setup directories
    dirs = setup_run_directories(run_id)

    # Save configuration
    save_run_config(run_id, config, seed, episodes)

    # Setup metrics CSV
    metrics_path = setup_metrics_csv(run_id)

    try:
        # Apply configuration overrides
        original_values = apply_config_overrides(config)

        # Set run-specific parameters
        import config as cfg
        cfg.SEED = seed
        cfg.MAX_EPISODES = episodes

        # Override paths for isolated output
        original_metrics_path = None
        original_model_save = None

        # Import and setup agent with overrides
        from agent import Agent
        from game import SnakeGameAI
        import torch
        import random
        import numpy as np

        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # Monkey patch the CSV logging to use our run-specific path
        def patched_init_csv_logging(self):
            if not os.path.exists(metrics_path):
                with open(metrics_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['episode', 'score', 'avg100', 'steps', 'epsilon', 'timeouts', 'train_loss',
                                   'a_straight', 'a_left', 'a_right', 'random_actions', 'loop_detected',
                                   'food_events', 'mean_steps_per_food'])

        def patched_log_episode(self, score, steps, timed_out, train_loss, loop_detected, food_events, mean_steps_per_food):
            self.scores.append(score)
            avg100 = self.moving_average(100)

            # Check if we have a new best average
            if avg100 > self.best_avg:
                self.best_avg = avg100
                # Save to run-specific directory
                best_model_path = f"models/{run_id}/best_avg.pth"
                self.model.save(best_model_path.replace('models/', '').replace(f'{run_id}/', ''))
                # Move to correct location
                if os.path.exists('models/best_avg.pth'):
                    shutil.move('models/best_avg.pth', best_model_path)
                print(f"New best average: {avg100:.2f}, saved to {best_model_path}")

            # Log to run-specific CSV
            with open(metrics_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.n_games, score, avg100, steps, self.epsilon, int(timed_out), train_loss,
                               self.actions_straight, self.actions_left, self.actions_right, self.random_actions,
                               int(loop_detected), food_events, mean_steps_per_food])

            # Reset episode metrics for next episode
            self.reset_episode_metrics()

            # Checkpoint every N episodes to run-specific directory
            if self.n_games % cfg.CHECKPOINT_EVERY == 0:
                checkpoint_path = f"models/{run_id}/checkpoint_ep{self.n_games}.pth"
                self.model.save(f"checkpoint_ep{self.n_games}.pth")
                if os.path.exists(f'models/checkpoint_ep{self.n_games}.pth'):
                    shutil.move(f'models/checkpoint_ep{self.n_games}.pth', checkpoint_path)

                # Save plots to run-specific directory
                from helper import save_plots
                save_plots(self.n_games, plots_dir=dirs['plots_dir'])
                print(f"Checkpoint saved at episode {self.n_games}")

        # Patch Agent methods
        Agent.init_csv_logging = patched_init_csv_logging
        Agent.log_episode = patched_log_episode

        # Run training
        print("Initializing agent and game...")
        agent = Agent()
        game = SnakeGameAI()

        # Training loop
        plot_scores = []
        plot_mean_scores = []
        plot_epsilons = []
        plot_steps = []
        total_score = 0
        record = 0

        print(f"Starting training for {episodes} episodes...")

        while agent.n_games < episodes:
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
                train_loss = long_loss[0] if long_loss is not None else -1

                # Reset game
                steps = game.frame_iteration
                game.reset()
                agent.n_games += 1

                # Update record
                if score > record:
                    record = score
                    # Save record model to run-specific directory
                    record_path = f"models/{run_id}/model.pth"
                    agent.model.save('model.pth')
                    if os.path.exists('models/model.pth'):
                        shutil.move('models/model.pth', record_path)

                # Get behavioral metrics from game
                loop_detected = game.loop_detected
                food_events = game.food_events
                mean_steps_per_food = game.get_mean_steps_per_food()

                # Log episode
                agent.log_episode(score, steps, timed_out, train_loss, loop_detected, food_events, mean_steps_per_food)

                if agent.n_games % 100 == 0:
                    print(f'Game {agent.n_games}, Score: {score}, Record: {record}, Avg100: {agent.moving_average(100):.2f}, Epsilon: {agent.epsilon:.3f}')

                # Update plots data
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot_epsilons.append(agent.epsilon)
                plot_steps.append(steps)

        print(f"[COMPLETED] Training completed: {episodes} episodes")

        # Final metrics calculation
        final_avg100 = agent.moving_average(100)
        best_avg100 = agent.best_avg

        # Calculate timeout rate
        timeout_episodes = 0
        total_episodes = 0
        food_events_total = 0

        with open(metrics_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_episodes += 1
                if int(row['timeouts']) == 1:
                    timeout_episodes += 1
                food_events_total += int(row['food_events'])

        timeout_rate = timeout_episodes / total_episodes if total_episodes > 0 else 0
        mean_food = food_events_total / total_episodes if total_episodes > 0 else 0

        # Restore original config values
        restore_config_values(original_values)

        return {
            'run_id': run_id,
            'seed': seed,
            'episodes': episodes,
            'final_avg100': final_avg100,
            'best_avg100': best_avg100,
            'timeout_rate': timeout_rate,
            'mean_food': mean_food,
            'config': config
        }

    except Exception as e:
        error_msg = f"Error in run {run_id}: {str(e)}"
        print(f"[ERROR] {error_msg}")

        # Save error to stderr.txt
        error_path = f"runs/{run_id}/stderr.txt"
        with open(error_path, 'w') as f:
            f.write(f"Error in run {run_id}:\n")
            f.write(f"{str(e)}\n\n")
            f.write(traceback.format_exc())

        # Restore original config values
        restore_config_values(original_values)

        return None


def update_summary_csv(results):
    """Update or create runs/summary.csv with results"""
    os.makedirs('runs', exist_ok=True)
    summary_path = 'runs/summary.csv'

    # Check if summary file exists
    file_exists = os.path.exists(summary_path)

    with open(summary_path, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file is new
        if not file_exists:
            writer.writerow(['run_id', 'seed', 'episodes', 'decay', 'timeout_factor', 'gamma',
                           'final_avg100', 'best_avg100', 'timeout_rate', 'mean_food'])

        # Write results
        for result in results:
            if result is not None:
                config = result['config']
                writer.writerow([
                    result['run_id'],
                    result['seed'],
                    result['episodes'],
                    config.get('DECAY', ''),
                    config.get('TIMEOUT_FACTOR', ''),
                    config.get('gamma', ''),
                    f"{result['final_avg100']:.4f}",
                    f"{result['best_avg100']:.4f}",
                    f"{result['timeout_rate']:.4f}",
                    f"{result['mean_food']:.4f}"
                ])


def print_summary_table(results):
    """Print a summary table of all runs"""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    if not results or all(r is None for r in results):
        print("No successful runs to summarize.")
        return

    # Print header
    print(f"{'Run ID':<20} {'Seed':<6} {'Episodes':<8} {'Decay':<7} {'Timeout':<8} {'Gamma':<6} {'Final100':<9} {'Best100':<8} {'TimeoutR':<9} {'MeanFood':<8}")
    print("-" * 80)

    # Print results
    for result in results:
        if result is not None:
            config = result['config']
            print(f"{result['run_id']:<20} {result['seed']:<6} {result['episodes']:<8} "
                  f"{config.get('DECAY', 'N/A'):<7} {config.get('TIMEOUT_FACTOR', 'N/A'):<8} "
                  f"{config.get('gamma', 'N/A'):<6} {result['final_avg100']:<9.4f} "
                  f"{result['best_avg100']:<8.4f} {result['timeout_rate']:<9.4f} {result['mean_food']:<8.4f}")

    print("\n[SUMMARY] Summary saved to: runs/summary.csv")


def main():
    parser = argparse.ArgumentParser(description='Run Snake AI experiments with different hyperparameters')
    parser.add_argument('--episodes', type=int, default=EXPERIMENT_EPISODES,
                        help=f'Number of episodes to train (default: {EXPERIMENT_EPISODES})')
    parser.add_argument('--max-runs', type=int, default=None,
                        help='Maximum number of configurations to run (default: all)')
    parser.add_argument('--base-seed', type=int, default=42,
                        help='Base seed for experiments (default: 42)')
    parser.add_argument('--configs-file', type=str, default=None,
                        help='JSON file with experiment configurations (default: use EXPERIMENT_CONFIGS)')

    args = parser.parse_args()

    # Load configurations
    if args.configs_file:
        configs = load_configs_from_file(args.configs_file)
    else:
        configs = EXPERIMENT_CONFIGS
        print(f"Using default configurations: {len(configs)} configs")

    # Limit configurations if requested
    if args.max_runs:
        configs = configs[:args.max_runs]
        print(f"Limited to {len(configs)} configurations")

    print(f"\n[CONFIG] Experiment Settings:")
    print(f"  Episodes per run: {args.episodes}")
    print(f"  Base seed: {args.base_seed}")
    print(f"  Total configurations: {len(configs)}")
    print(f"  Estimated total episodes: {len(configs) * args.episodes}")

    # Run experiments
    results = []

    for i, config in enumerate(configs):
        print(f"\n[RUN] Configuration {i+1}/{len(configs)}: {config}")

        run_id = generate_run_id()
        seed = args.base_seed + i

        result = run_training(run_id, config, seed, args.episodes)
        results.append(result)

        # Update summary after each run
        if result:
            update_summary_csv([result])

    # Final summary - only print table, don't update CSV again
    successful_results = [r for r in results if r is not None]
    print_summary_table(results)

    successful_runs = len([r for r in results if r is not None])
    failed_runs = len([r for r in results if r is None])

    print(f"\n[FINISH] Experiment completed!")
    print(f"[SUCCESS] Successful runs: {successful_runs}")
    print(f"[FAILED] Failed runs: {failed_runs}")

    if successful_runs > 0:
        print(f"\n[RESULTS] Results stored in:")
        print(f"  - Individual runs: runs/YYYYMMDD_HHMMSS_<hash>/")
        print(f"  - Summary: runs/summary.csv")
        print(f"  - Models: models/YYYYMMDD_HHMMSS_<hash>/")


if __name__ == '__main__':
    main()