import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_data = "./HousingData.csv"

data = pd.read_csv(file_data)

m, c = -13.978323519135387, 11.548749680948196

plt.scatter(x=data["NOX"], y=data["DIS"])

x_values = np.linspace(min(data["NOX"]), max(data["NOX"]), 100)
y_values = m * x_values + c

plt.plot(x_values, y_values, color="red")

plt.show()
