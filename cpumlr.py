data = pd.read_csv("computer_hardware_dataset.csv")
Xs = data.drop(['ERP','vendor_name','model_name','CACH','CHMIN','CHMAX'], axis=1)
print(Xs)
y = data['ERP'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(Xs, y)
print("The linear model is: Y = {:.5} + {:.5}*MYCT + {:.5}*MMIN + {:.5}*MMAX + {:.5}*PRP".format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][2], reg.coef_[0][3]))
predictions = reg.predict(Xs)
plt.figure(figsize=(16, 8))
plt.scatter(
    data['MYCT'],data['ERP'],c='black'
)
plt.scatter(data['MMIN'],data['ERP'], c='black')
plt.scatter(data['MMAX'], data['ERP'],c='black')
plt.scatter(data['PRP'],data['ERP'], c='black')

plt.plot(
    data['MYCT'],
    data['MMIN'],
    data['MMAX'],
    data['PRP'],
    predictions,
    c='blue',
    linewidth=2
)

plt.show()
print(reg.score(Xs,y))
