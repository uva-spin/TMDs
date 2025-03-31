import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



# def SqT(mm,qT,qTmax):
#     qT = tf.where(qT > qTmax, qTmax, qT)  # Replace qT with 3 if qT > 3
#     return (8 * mm * mm + qT**4) / (32 * np.pi * mm) * tf.exp(-qT**2 / (2 * mm))

# def SqT(mm,qT,qTmax):
#     qT = tf.where(qT > qTmax, qTmax, qT)  # Replace qT with 3 if qT > 3
#     return 1/ (2 * np.pi * mm) * tf.exp(-qT**2 / (2 * mm))

def SqT1(mm,qT,qTmax):
    qT = tf.where(qT > qTmax, qTmax, qT)  # Replace qT with 3 if qT > 3
    return (8 * mm * mm + qT**4) / (32 * np.pi * mm) * tf.exp(-qT**2 / (2 * mm))

def SqT2(mm,qT,qTmax):
    qT = tf.where(qT > qTmax, qTmax, qT)  # Replace qT with 3 if qT > 3
    return 1/ (2 * np.pi * mm) * tf.exp(-qT**2 / (2 * mm))
    


def plot_SqT_comparison(mm):
    qT_values = np.linspace(0, 6, 100)
    qT_max_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    colors = ['green', 'blue', 'orange', 'brown', 'red']

    plt.figure(figsize=(10, 6))
    
    for qTmax, color in zip(qT_max_values, colors):
        SqT_values = SqT1(mm, qT_values, qTmax)
        plt.plot(qT_values, SqT_values, label=f'qTmax={qTmax}', linestyle='-', color=color)

    for qTmax, color in zip(qT_max_values, colors):
        SqT_values = SqT2(mm, qT_values, qTmax)
        plt.plot(qT_values, SqT_values, label=f'qTmax={qTmax}', linestyle='--', color=color)
    

    plt.xlabel(r'$q_T$', fontsize=14)
    plt.ylabel(r'$S(q_T)$', fontsize=14)
    plt.title(f'Comparison of $S(q_T)$ for m = {mm}', fontsize=16)
    plt.legend(fontsize=12)
    #plt.ylim(0,0.1)
    plt.grid(True)
    plt.savefig(f"SqT_comparison_plot_at_m={mm}.pdf")
    plt.close()

plot_SqT_comparison(0.5)