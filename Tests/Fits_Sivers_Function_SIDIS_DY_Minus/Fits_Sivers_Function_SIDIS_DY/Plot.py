import numpy as np
import matplotlib.pyplot as plt

fig,ax = plt.subplots(1)

# create some x data and some integers for the y axis
x = np.array([3,5,2,4])
y = np.arange(4)

# plot the data
ax.plot(x,y)

# tell matplotlib which yticks to plot 
ax.set_yticks([0,1,2,3])

# labelling the yticks according to your list
ax.set_yticklabels(['A','B','C','D'])