import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_sqt_comparisons():
    """
    Load SqT results from CSV and create comparison plots.
    """
    # Check if the CSV file exists
    if not os.path.exists('csvs/SqT_results.csv'):
        print("Error: SqT_results.csv file not found in comparison_plots directory.")
        return
    
    # Load the data
    print("Loading SqT results data...")
    df = pd.read_csv('csvs/SqT_results.csv')
    
    # Create directory if it doesn't exist
    os.makedirs('comparison_plots', exist_ok=True)
    
    # Extract relevant columns
    SqT_orig = df['SqT_orig'].values
    SqT_err_orig = df['SqT_err_orig'].values
    SqT_mean = df['SqT_pred_mean'].values
    SqT_std = df['SqT_pred_std'].values
    
    # Plot 1: SqT vs x1
    plt.figure(1, figsize=(10, 6))
    
    # Plot original data with error bars
    plt.errorbar(df['x1'], SqT_orig, yerr=SqT_err_orig, fmt='o', color='blue', 
                 alpha=0.7, label='Original Data')
    
    # Plot model predictions with uncertainty
    plt.errorbar(df['x1'], SqT_mean, yerr=SqT_std, fmt='s', color='red', 
                 alpha=0.7, label='Model Predictions')
    
    plt.title('SqT Predictions vs Original Data (x1)')
    plt.xlabel('$x_1$')
    plt.ylabel('SqT')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparison_plots/SqT_vs_x1_comparison.png', dpi=300)
    print("Saved plot: SqT_vs_x1_comparison.png")
    
    # Plot 2: SqT vs qT
    plt.figure(2, figsize=(10, 6))
    
    # Plot original data with error bars
    plt.errorbar(df['qT'], SqT_orig, yerr=SqT_err_orig, fmt='o', color='blue', 
                 alpha=0.7, label='Original Data')
    
    # Plot model predictions with uncertainty
    plt.errorbar(df['qT'], SqT_mean, yerr=SqT_std, fmt='s', color='red', 
                 alpha=0.7, label='Model Predictions')
    
    plt.title('SqT Predictions vs Original Data (qT)')
    plt.xlabel('$q_T$')
    plt.ylabel('SqT')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparison_plots/SqT_vs_qT_comparison.png', dpi=300)
    print("Saved plot: SqT_vs_qT_comparison.png")
    
    # Additional statistics
    mean_rel_error = (abs(SqT_mean - SqT_orig) / SqT_orig).mean() * 100
    print(f"Mean relative error: {mean_rel_error:.2f}%")
    
    # Calculate chi-squared if errors are available
    if 'SqT_err_orig' in df.columns:
        chi2 = sum(((SqT_mean - SqT_orig) / SqT_err_orig) ** 2)
        reduced_chi2 = chi2 / len(SqT_orig)
        print(f"Chi-squared: {chi2:.2f}")
        print(f"Reduced chi-squared: {reduced_chi2:.2f}")

if __name__ == "__main__":
    plot_sqt_comparisons()
    plt.close('all')  # Close all figures after saving