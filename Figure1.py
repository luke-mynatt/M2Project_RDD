import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming dwi is your DataFrame and bac1 is a column in it.
# Replace this with loading your actual data.
# dwi = pd.read_csv('path_to_your_data.csv') 
dwi = pd.read_stata('hansen_dwi.dta')

# print(dwi.shape)
# print(dwi.head())
# Creating a histogram
plt.hist(dwi['bac1'], bins=np.arange(min(dwi['bac1']), max(dwi['bac1']) + 0.001, 0.001), color="#8aa1b4")

# Adding a vertical line at x = 0.08
plt.axvline(x=0.08, linewidth=1, linestyle='--', color='tomato', alpha=0.7)

# Setting labels and title
plt.xlabel('BAC', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.title('BAC histogram\nFigure 1, Hansen (2015)', fontsize=15)

# Customizing the plot
plt.tight_layout()  # Adjust the layout to make room for the labels
plt.grid(False)  # Turn off the grid, which is not present in the classic ggplot theme
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show plot
plt.show()
