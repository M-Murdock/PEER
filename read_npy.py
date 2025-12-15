import numpy as np
import sys

def read(POLICY):
    DIR = "trained_policies/"
    # POLICY = "q_table_bottomright.npy"
    # POLICY = "q_table_orbit.npy"
    # Set print options to display the full array without truncation
    np.set_printoptions(threshold=sys.maxsize)

    # Load the .npy file
    data = np.load(DIR+POLICY, allow_pickle=True)

    return data

# Print the data (the array)
# print(data)

# # Optional: Print metadata
# print("\nShape:", data.shape)
# print("Data Type:", data.dtype)
circle = read("q_table_orbit.npy")

corner = read("q_table_bottomright.npy")

print(circle[5])
print(corner[5])