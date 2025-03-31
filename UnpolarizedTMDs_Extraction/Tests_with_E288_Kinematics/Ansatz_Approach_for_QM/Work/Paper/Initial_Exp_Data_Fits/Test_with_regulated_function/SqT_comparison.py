import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



# def SqT(mm,qT,qTmax):
#     qT = tf.where(qT > qTmax, qTmax, qT)  # Replace qT with 3 if qT > 3
#     return (8 * mm * mm + qT**4) / (32 * np.pi * mm) * tf.exp(-qT**2 / (2 * mm))

def SqT(mm,qT,qTmax):
    qT = tf.where(qT > qTmax, qTmax, qT)  # Replace qT with 3 if qT > 3
    return 1/ (2 * np.pi * mm) * tf.exp(-qT**2 / (2 * mm))
    

# # Generate QM Range for Comparison
# qT_values = np.linspace(0, 6, 100)
# SqT_values_10 = SqT(0.5,qT_values,1.0)
# SqT_values_20 = SqT(0.5,qT_values,2.0)
# SqT_values_30 = SqT(0.5,qT_values,3.0)
# SqT_values_40 = SqT(0.5,qT_values,4.0)
# SqT_values_50 = SqT(0.5,qT_values,5.0)

# plt.figure(1,figsize=(10, 6))
# plt.plot(qT_values, SqT_values_10, label='qTmax=1.0', linestyle='-', color='green')
# plt.plot(qT_values, SqT_values_20, label='qTmax=2.0', linestyle='-', color='blue')
# plt.plot(qT_values, SqT_values_30, label='qTmax=3.0', linestyle='-', color='orange')
# plt.plot(qT_values, SqT_values_40, label='qTmax=4.0', linestyle='-', color='brown')
# plt.plot(qT_values, SqT_values_50, label='qTmax=5.0', linestyle='-', color='red')
# plt.xlabel(r'$Q_M$', fontsize=14)
# plt.ylabel(r'$S(q_T)$', fontsize=14)
# plt.title('Comparison of $S(q_T)$', fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.savefig("SqT_comparison_plot.pdf")
# plt.close()


def plot_SqT_comparison(mm):
    qT_values = np.linspace(0, 6, 100)
    qT_max_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    colors = ['green', 'blue', 'orange', 'brown', 'red']

    plt.figure(figsize=(10, 6))
    
    for qTmax, color in zip(qT_max_values, colors):
        SqT_values = SqT(mm, qT_values, qTmax)
        plt.plot(qT_values, SqT_values, label=f'qTmax={qTmax}', linestyle='-', color=color)
    
    plt.xlabel(r'$Q_M$', fontsize=14)
    plt.ylabel(r'$S(q_T)$', fontsize=14)
    plt.title(f'Comparison of $S(q_T)$ for m = {mm}', fontsize=16)
    plt.legend(fontsize=12)
    plt.ylim(0,0.1)
    plt.grid(True)
    plt.savefig(f"SqT_comparison_plot_at_m={mm}.pdf")
    plt.close()

plot_SqT_comparison(0.5)
plot_SqT_comparison(1.0)
plot_SqT_comparison(1.5)
plot_SqT_comparison(2.0)
plot_SqT_comparison(3.0)
plot_SqT_comparison(4.0)