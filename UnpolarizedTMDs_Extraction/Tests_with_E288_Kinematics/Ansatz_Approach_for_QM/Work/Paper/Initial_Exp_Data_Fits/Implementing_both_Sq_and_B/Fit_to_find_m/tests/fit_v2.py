import pandas as pd
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
import matplotlib.pyplot as plt

# Define the model function S(q_T) = 1/(2π m) * exp(-q_T^2/(2m))
def model_function(qT, m):
    return 1/(2*np.pi*m) * np.exp(-qT**2/(2*m))

# Load the data from CSV file
def fit_parameter_m():
    # Read the CSV file
    functions_results = pd.read_csv("comparison_results.csv")
    
    # Extract data
    qT = functions_results['qT'].values
    SqT_mean = functions_results['SqT_mean'].values
    SqT_std = functions_results['SqT_std'].values
    
    # Data exploration to help with initial guess
    print("Data Statistics:")
    print(f"qT range: {min(qT)} to {max(qT)}")
    print(f"SqT_mean range: {min(SqT_mean)} to {max(SqT_mean)}")
    
    # Find the maximum value of SqT_mean for a better initial guess
    max_SqT_index = np.argmax(SqT_mean)
    max_SqT_value = SqT_mean[max_SqT_index]
    
    # The theoretical maximum of the function is 1/(2πm) at qT=0
    # So a good initial guess for m would be 1/(2π*max_SqT_value)
    # However, if the peak is not at qT=0, we need to adjust this
    
    # Try several initial guesses and pick the best
    m_guesses = np.logspace(-3, 1, 10)  # Try values from 0.001 to 10
    best_chi2 = float('inf')
    best_m_guess = 1.0
    
    for m_guess in m_guesses:
        # Create a least squares cost function
        least_squares = LeastSquares(qT, SqT_mean, SqT_std, model_function)
        
        # Initialize Minuit with this guess
        minuit = Minuit(least_squares, m=m_guess)
        minuit.limits["m"] = (0, None)  # m should be positive
        
        # Run migrad with a single call
        minuit.migrad()
        
        # Check if this is better
        if minuit.fval < best_chi2:
            best_chi2 = minuit.fval
            best_m_guess = m_guess
    
    print(f"Best initial guess for m: {best_m_guess}")
    
    # Create the final least squares cost function
    least_squares = LeastSquares(qT, SqT_mean, SqT_std, model_function)
    
    # Initialize Minuit with the best initial guess
    minuit = Minuit(least_squares, m=best_m_guess)
    minuit.limits["m"] = (0, None)  # m should be positive
    
    # Set more detailed options for the minimizer
    minuit.strategy = 2  # More careful minimization (0=fast, 1=default, 2=careful)
    minuit.tol = 0.0001  # Tolerance for convergence
    
    # Perform the fit with multiple attempts
    for strategy in [0, 1, 2]:
        minuit.strategy = strategy
        minuit.migrad()  # Find minimum
        
        # Try simplex method if migrad doesn't converge well
        if not minuit.valid:
            print(f"Trying simplex with strategy {strategy}")
            minuit.simplex()
            minuit.migrad()  # Run migrad again after simplex
    
    # Compute errors
    minuit.hesse()
    
    # Try to improve further with minos error analysis
    try:
        minuit.minos()
    except:
        print("Minos error analysis failed, using Hesse errors")
    
    # Print fit results
    print(f"\nFit Results for Parameter m:")
    print(f"m = {minuit.values['m']:.5f} ± {minuit.errors['m']:.5f}")
    print(f"χ²/ndof = {minuit.fval:.2f}/{len(qT) - len(minuit.values)} = {minuit.fval/(len(qT) - len(minuit.values)):.3f}")
    
    # Check if the fit is reasonable
    if minuit.fval/(len(qT) - len(minuit.values)) > 5:
        print("\nWarning: Chi-squared per degree of freedom is still high.")
        print("Possible issues:")
        print("1. The model may not be appropriate for this data")
        print("2. The uncertainties may be underestimated")
        print("3. The data may contain outliers")
        
        # Calculate and print residuals to check for patterns
        residuals = SqT_mean - model_function(qT, minuit.values['m'])
        normalized_residuals = residuals / SqT_std
        
        print("\nResidual Analysis:")
        print(f"Mean of normalized residuals: {np.mean(normalized_residuals):.3f}")
        print(f"Standard deviation of normalized residuals: {np.std(normalized_residuals):.3f}")
        print(f"Range of normalized residuals: {min(normalized_residuals):.3f} to {max(normalized_residuals):.3f}")
    
    # Plot the data and the fit
    plt.figure(figsize=(12, 8))
    
    # Create a grid of subplots
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Data and fit plot
    ax1 = plt.subplot(gs[0])
    ax1.errorbar(qT, SqT_mean, yerr=SqT_std, fmt='o', label='Data', markersize=4)
    
    # Fitted function
    qT_fine = np.linspace(min(qT), max(qT), 1000)
    SqT_fit = model_function(qT_fine, minuit.values['m'])
    ax1.plot(qT_fine, SqT_fit, 'r-', label=f'Fit: m = {minuit.values["m"]:.5f}')
    
    ax1.set_xlabel('$q_T$')
    ax1.set_ylabel('$S(q_T)$')
    ax1.legend()
    ax1.set_title('Minuit Fit of $S(q_T) = \\frac{1}{2\\pi m} \\exp(-\\frac{q_T^2}{2m})$')
    ax1.grid(True)
    
    # Residuals plot
    ax2 = plt.subplot(gs[1])
    residuals = SqT_mean - model_function(qT, minuit.values['m'])
    normalized_residuals = residuals / SqT_std
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.3)
    ax2.axhline(y=-1, color='r', linestyle='--', alpha=0.3)
    ax2.scatter(qT, normalized_residuals, color='blue', s=20)
    ax2.set_xlabel('$q_T$')
    ax2.set_ylabel('Norm. Residuals')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_minuit_fit_result.png', dpi=300)
    plt.show()
    
    return minuit

if __name__ == "__main__":
    result = fit_parameter_m()
    
    # Print the parameter value in a format easy to copy
    print(f"\nFinal result: m = {result.values['m']}")