import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set professional plotting style
plt.style.use('ggplot')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 300
})

def extract_data(output_dir):
    results = []
    # Expected folder pattern: whisper-{size}-{lang}-qlora
    for folder in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        parts = folder.split('-')
        if len(parts) >= 4 and parts[0] == 'whisper':
            size = parts[1].capitalize()
            lang = parts[2].upper()
            
            json_path = os.path.join(folder_path, 'comparison_results.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # We focus on WER and CER as requested
                    results.append({
                        'Language': lang,
                        'Model Size': size,
                        'Metric': 'WER',
                        'Baseline': data.get('base_wer'),
                        'Fine-tuned (LoRA)': data.get('adapter_wer')
                    })
                    results.append({
                        'Language': lang,
                        'Model Size': size,
                        'Metric': 'CER',
                        'Baseline': data.get('base_cer'),
                        'Fine-tuned (LoRA)': data.get('adapter_cer')
                    })
    
    return pd.DataFrame(results)

def plot_language_results(df, lang, full_lang_name):
    lang_df = df[df['Language'] == lang]
    if lang_df.empty:
        print(f"No data found for {full_lang_name} ({lang})")
        return

    metrics = ['WER', 'CER']
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Custom colors for thesis look
    colors = ['#4C72B0', '#55A868'] # Deep blue and Green
    
    for i, metric in enumerate(metrics):
        metric_df = lang_df[lang_df['Metric'] == metric]
        
        # Prepare data for grouped bar chart
        model_sizes = metric_df['Model Size'].unique()
        x = np.arange(len(model_sizes))
        width = 0.35
        
        baseline_vals = metric_df.sort_values('Model Size')['Baseline'].values
        finetuned_vals = metric_df.sort_values('Model Size')['Fine-tuned (LoRA)'].values
        
        rects1 = axes[i].bar(x - width/2, baseline_vals, width, label='Baseline', color=colors[0], alpha=0.8, edgecolor='black')
        rects2 = axes[i].bar(x + width/2, finetuned_vals, width, label='Fine-tuned (LoRA)', color=colors[1], alpha=0.8, edgecolor='black')
        
        # Add labels, title and custom x-axis tick labels, etc.
        axes[i].set_ylabel(f'{metric} (%)')
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(model_sizes)
        
        # Increase Y-limit to provide headroom for labels and legend
        max_val = max(max(baseline_vals), max(finetuned_vals))
        axes[i].set_ylim(0, max_val * 1.25) 
        
        axes[i].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.8)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        def autolabel(rects, ax):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

        autolabel(rects1, axes[i])
        autolabel(rects2, axes[i])

    plt.suptitle(f'Whisper Performance Comparison: {full_lang_name}', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_filename = f'{full_lang_name.lower()}_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_filename}")
    plt.close()

if __name__ == "__main__":
    output_dir = 'output'
    if not os.path.exists(output_dir):
        print(f"Error: {output_dir} directory not found.")
    else:
        results_df = extract_data(output_dir)
        if results_df.empty:
            print("No results found in the output subdirectories.")
        else:
            print("Aggregated Results:")
            print(results_df.to_string(index=False))
            
            # Plot for Hindi
            plot_language_results(results_df, 'HI', 'Hindi')
            
            # Plot for Marathi
            plot_language_results(results_df, 'MR', 'Marathi')
            
            print("\nAnalysis complete. All plots saved.")
