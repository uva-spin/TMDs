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
    
    # Create a least squares cost function
    least_squares = LeastSquares(qT, SqT_mean, SqT_std, model_function)
    
    # Initialize Minuit with an initial guess for m
    m_initial_guess = 1.0  # You may need to adjust this based on your data
    minuit = Minuit(least_squares, m=m_initial_guess)
    
    # Set parameter limits if needed (remove or adjust as necessary)
    minuit.limits["m"] = (0, None)  # m should be positive
    
    # Perform the fit
    minuit.migrad()  # Find minimum
    minuit.hesse()   # Compute errors
    
    # Print fit results
    print(f"Fit Results for Parameter m:")
    print(f"m = {minuit.values['m']:.5f} ± {minuit.errors['m']:.5f}")
    print(f"χ²/ndof = {minuit.fval:.2f}/{len(qT) - len(minuit.values)}")
    
    # Plot the data and the fit
    plt.figure(figsize=(10, 6))
    
    # Data points with error bars
    plt.errorbar(qT, SqT_mean, yerr=SqT_std, fmt='o', label='Data')
    
    # Fitted function
    qT_fine = np.linspace(min(qT), max(qT), 1000)
    SqT_fit = model_function(qT_fine, minuit.values['m'])
    plt.plot(qT_fine, SqT_fit, 'r-', label=f'Fit: m = {minuit.values["m"]:.5f}')
    
    plt.xlabel('$q_T$')
    plt.ylabel('$S(q_T)$')
    plt.legend()
    plt.title('Minuit Fit of $S(q_T) = \\frac{1}{2\\pi m} \\exp(-\\frac{q_T^2}{2m})$')
    plt.grid(True)
    plt.savefig('minuit_fit_result.png')
    plt.show()
    
    return minuit

if __name__ == "__main__":
    result = fit_parameter_m()