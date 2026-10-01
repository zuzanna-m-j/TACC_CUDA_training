import numpy as np 
import matplotlib.pyplot as plt


data = np.loadtxt("bandwidth.data")
x = data[:,0]
y = data[:,1]
plt.plot(x, y, marker = "o", lw = 1.5)
plt.xlabel("stride")
plt.ylabel("Bandwidth [GB/s]")
plt.savefig("bandwidth.png",dpi = 300)


