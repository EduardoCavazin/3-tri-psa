import matplotlib.pyplot as plt
from IPython import display
import os
from config import ENABLE_GRAD_NORM

# Try to import pandas, fallback if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available, advanced plots disabled. Install with: pip install pandas")

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

def plot_extended(scores, mean_scores, epsilons, steps, plots_dir='runs/plots'):
    """Extended plotting with multiple metrics"""
    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
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
    
def plot_learning_metrics(episode, metrics_csv='metrics.csv', plots_dir='runs/plots'):
    """Create comprehensive learning metrics plots from CSV data"""
    if not PANDAS_AVAILABLE:
        print("Pandas not available, skipping advanced plots")
        return

    if not os.path.exists(metrics_csv):
        print(f"No {metrics_csv} found for plotting")
        return
        
    try:
        # Read metrics data
        df = pd.read_csv(metrics_csv)
        if len(df) < 10:  # Need some data to plot
            return
            
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create learning metrics plot
        fig_size = (20, 12) if ENABLE_GRAD_NORM else (20, 9)
        rows = 3 if ENABLE_GRAD_NORM else 2
        fig, axes = plt.subplots(rows, 3, figsize=fig_size)
        
        # Row 1: Performance metrics
        # Score and avg100
        axes[0, 0].plot(df['episode'], df['score'], alpha=0.6, label='Score')
        axes[0, 0].plot(df['episode'], df['avg100'], linewidth=2, label='Avg100')
        axes[0, 0].set_title('Score Evolution')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Epsilon decay
        axes[0, 1].plot(df['episode'], df['epsilon'], 'r-', linewidth=2)
        axes[0, 1].set_title('Epsilon Decay (Exploration)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Epsilon')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Steps per episode
        axes[0, 2].plot(df['episode'], df['steps'], alpha=0.6, label='Steps')
        if len(df) >= 100:
            steps_ma = df['steps'].rolling(100, min_periods=1).mean()
            axes[0, 2].plot(df['episode'], steps_ma, linewidth=2, label='MA(100)')
        axes[0, 2].set_title('Steps per Episode')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Steps')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Row 2: Learning metrics
        # Training loss
        valid_loss = df[df['train_loss_mean'] != -1]
        if len(valid_loss) > 0:
            axes[1, 0].plot(valid_loss['episode'], valid_loss['train_loss_mean'], alpha=0.6, label='Loss')
            if len(valid_loss) >= 100:
                loss_ma = valid_loss['train_loss_mean'].rolling(100, min_periods=1).mean()
                axes[1, 0].plot(valid_loss['episode'], loss_ma, linewidth=2, label='MA(100)')
            axes[1, 0].set_title('Training Loss Evolution')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No valid loss data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Training Loss (No Data)')
        
        # Behavioral issues
        timeout_rate = df['timeouts'].rolling(100, min_periods=1).mean()
        loop_rate = df['loop_detected'].rolling(100, min_periods=1).mean()
        axes[1, 1].plot(df['episode'], timeout_rate, label='Timeout Rate (MA100)', linewidth=2)
        axes[1, 1].plot(df['episode'], loop_rate, label='Loop Rate (MA100)', linewidth=2)
        axes[1, 1].set_title('Behavioral Issues')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Action distribution (latest 1000 episodes)
        recent_data = df.tail(min(1000, len(df)))
        action_means = [
            recent_data['a_straight'].mean(),
            recent_data['a_left'].mean(), 
            recent_data['a_right'].mean()
        ]
        axes[1, 2].bar(['Straight', 'Left', 'Right'], action_means, color=['blue', 'red', 'green'], alpha=0.7)
        axes[1, 2].set_title('Action Distribution (Recent 1000 ep)')
        axes[1, 2].set_ylabel('Avg Actions per Episode')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Row 3: Gradient norms (if enabled)
        if ENABLE_GRAD_NORM and 'grad_norm_mean' in df.columns:
            valid_grad = df[df['grad_norm_mean'] != -1]
            if len(valid_grad) > 0:
                axes[2, 0].plot(valid_grad['episode'], valid_grad['grad_norm_mean'], alpha=0.6, label='Grad Norm')
                if len(valid_grad) >= 100:
                    grad_ma = valid_grad['grad_norm_mean'].rolling(100, min_periods=1).mean()
                    axes[2, 0].plot(valid_grad['episode'], grad_ma, linewidth=2, label='MA(100)')
                axes[2, 0].set_title('Gradient Norm Evolution')
                axes[2, 0].set_xlabel('Episode')
                axes[2, 0].set_ylabel('Gradient Norm')
                axes[2, 0].legend()
                axes[2, 0].grid(True, alpha=0.3)
                
                # Gradient norm histogram
                axes[2, 1].hist(valid_grad['grad_norm_mean'], bins=30, alpha=0.7, edgecolor='black')
                axes[2, 1].set_title('Gradient Norm Distribution')
                axes[2, 1].set_xlabel('Gradient Norm')
                axes[2, 1].set_ylabel('Frequency')
                axes[2, 1].grid(True, alpha=0.3)
                
                # Loss vs Grad Norm correlation
                axes[2, 2].scatter(valid_loss['train_loss_mean'], valid_grad['grad_norm_mean'], alpha=0.5)
                axes[2, 2].set_title('Loss vs Gradient Norm')
                axes[2, 2].set_xlabel('Loss')
                axes[2, 2].set_ylabel('Gradient Norm')
                axes[2, 2].grid(True, alpha=0.3)
            else:
                for i in range(3):
                    axes[2, i].text(0.5, 0.5, 'No gradient data', ha='center', va='center', transform=axes[2, i].transAxes)
        
        plt.suptitle(f'Training Metrics Dashboard - Episode {episode}', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'{plots_dir}/learning_metrics_ep{episode}.png', dpi=150, bbox_inches='tight')
        print(f"Learning metrics plots saved to {plots_dir}/learning_metrics_ep{episode}.png")
        
        plt.show(block=False)
        plt.pause(0.1)
        
    except Exception as e:
        print(f"Error creating learning metrics plots: {e}")

def save_plots(episode, plots_dir='runs/plots', metrics_csv='metrics.csv'):
    """Save current plots to PNG files"""
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # Generate comprehensive learning metrics plots
        plot_learning_metrics(episode, metrics_csv, plots_dir)
        
        # Save the current figure (if any)
        if plt.get_fignums():  # Check if there are any figures
            plt.savefig(f'{plots_dir}/training_ep{episode}.png', dpi=150, bbox_inches='tight')
            print(f"Additional plots saved to {plots_dir}/training_ep{episode}.png")
    except Exception as e:
        print(f"Error saving plots: {e}")
