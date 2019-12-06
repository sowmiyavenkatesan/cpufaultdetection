import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
data = pd.read_csv("computer_hardware_dataset.csv")
data=data[['MMIN','ERP']]
plt.figure(figsize=(16, 8))
plt.scatter(
    data['MMIN'],
    data['ERP'],
    c='black'
)
plt.xlabel("MINIMUM MEMORY")
plt.ylabel("MAX EFFICIENCY")
plt.show()
X = data['MMIN'].values.reshape(-1,1)
y = data['ERP'].values.reshape(-1,1)
reg = LinearRegression()
reg=reg.fit(X, y)
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))
predictions = reg.predict(X)
plt.figure(figsize=(16, 8))
plt.scatter(
    data['MMIN'],
    data['ERP'],
    c='black'
)
plt.plot(
    data['MMIN'],
    predictions,
    c='blue',
    linewidth=2
)
plt.xlabel("MINIMUM MEMORY OF CPU")
plt.ylabel("CPU EFFICIENCY")
plt.show()
print(reg.score(X,y))
