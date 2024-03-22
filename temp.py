import numpy as np
import matplotlib.pyplot as plt

# Given equation parameters
W_value = 16 * 3.875 * 64 * 2**20 / 2**6   # 64 cores, 2MB L2, 1.85MB L3 
print(W_value)
# S range for plotting
S = np.linspace(1000, 100000, 400) # S > 0

# Calculating W from the given equation
W = W_value / S
# Define the function
y = S * W * (52 - np.log2(S) + 2 * 16)
y /= 2**30  # Convert to MB

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(S, y, label='y = S * W * (52 - log2(S) + 2)')
plt.title('Plot of y = S * W * (52 - log2(S) + 2) with respect to S')
plt.xlabel('S')
plt.ylabel("y (GB)")
plt.legend()
plt.grid(True)
plt.savefig("temp.png")
