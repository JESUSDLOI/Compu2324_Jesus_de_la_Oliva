import pandas as pd
import matplotlib.pyplot as plt

# Ask the user for the filename
filename = input("Enter the filename of the .txt file: ")

# Read the data from the file
# Assumes the file has columns separated by spaces or tabs
data = pd.read_csv(filename, sep=",")

# Plot the data
# This example assumes you have two columns named 'x' and 'y'
plt.plot(data['x'], data['y'])

# Show the plot
plt.show()