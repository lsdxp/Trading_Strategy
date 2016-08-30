import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import csv

def line_plot(prices,prices2,prices3):
    plt.title('prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.plot(prices,'r')
    plt.plot(prices2,'b')
    plt.plot(prices3,'g')

    plt.grid(True)
    plt.show()

def join_desnity(prices,prices3):
    data = np.column_stack((prices, prices3))
    df = pd.DataFrame(data, columns=["x", "y"])
    sns.jointplot(x="x", y="y", data=df)
    sns.plt.show()




