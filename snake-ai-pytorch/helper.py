import matplotlib.pyplot as plt
from IPython import display
import os

plt.ion()

def plot(scores, mean_scores):
    """Original plotting function - kept for compatibility"""
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

def plot_extended(scores, mean_scores, epsilons, steps):
    """Extended plotting with multiple metrics"""
    # Create plots directory if it doesn't exist
    os.makedirs('runs/plots', exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Score plot
    ax1.plot(scores, label='Score', alpha=0.7)
    ax1.plot(mean_scores, label='Mean Score (all time)', linewidth=2)
    if len(scores) >= 100:
        moving_avg = [sum(scores[max(0, i-99):i+1])/min(i+1, 100) for i in range(len(scores))]
        ax1.plot(moving_avg, label='Moving Avg (100)', linewidth=2)
    ax1.set_title('Score per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Epsilon plot
    ax2.plot(epsilons, 'r-', linewidth=2)
    ax2.set_title('Epsilon Decay')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.grid(True, alpha=0.3)
    
    # Steps plot
    ax3.plot(steps, 'g-', alpha=0.7)
    if len(steps) >= 100:
        steps_avg = [sum(steps[max(0, i-99):i+1])/min(i+1, 100) for i in range(len(steps))]
        ax3.plot(steps_avg, 'g-', linewidth=2, label='Moving Avg (100)')
    ax3.set_title('Steps per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Score histogram
    ax4.hist(scores, bins=20, alpha=0.7, edgecolor='black')
    ax4.set_title('Score Distribution')
    ax4.set_xlabel('Score')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(.1)
    
def save_plots(episode):
    """Save current plots to PNG files"""
    os.makedirs('runs/plots', exist_ok=True)
    
    try:
        # Save the current figure
        plt.savefig(f'runs/plots/training_ep{episode}.png', dpi=150, bbox_inches='tight')
        print(f"Plots saved to runs/plots/training_ep{episode}.png")
    except Exception as e:
        print(f"Error saving plots: {e}")
