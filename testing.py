import numpy as np
import matplotlib.pyplot as plt

errs = np.loadtxt('outiters.txt')
rms = []
mae = []
for i,x in enumerate(errs):
    if i % 2 == 0:
        mae.append(x)
    elif i % 2 != 0:
        rms.append(x)
plt.plot(mae)
plt.xlabel("Iteratons")
plt.ylabel("Mean Absolute Error")
plt.show()
plt.plot(rms)
plt.xlabel("Iteratons")
plt.ylabel("Root Mean Squared Error")
plt.show()
print(min(mae))
print(min(rms))


# a = act[:,2]
# p = pred[:,2]
#
# diff = np.abs(p-a) / N
# dsq = np.square(p-a) / N
#
# MAE = np.sum(diff)
# RMS = np.sqrt(np.sum(dsq))
# print(MAE)
# print(RMS)