# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sn
from IPython.display import display
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import r2_score# function for Correlation table 
from statsmodels.stats.outliers_influence import variance_inflation_factor



def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v
#===============================================
'''-> Read Excel data sheet'''
df = pd.read_excel("Insurance.xlsx")

'''-> to display all shrinked columns'''
pd.set_option('display.max_columns', 30)

'''-> correlation function call'''
k = df.corr(method=histogram_intersection)

''' -> to plot a dataframe in a scatter map'''
df.plot(kind='scatter',x='Insured',y='Payment',color='red')

''' -> correlation between Claims and Payment'''
## print("correlation between Claims and Payment : ",df['Claims'].corr(df['Payment']))

''' -> correlation between Insured and Payment'''
##print("correlation between Insured and Payment : ",df['Insured'].corr(df['Payment']))

'''-> Fitting a Linear Regression model '''
X = df['Insured'].values.reshape(-1,1)
y = df['Payment'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(X, y)
''' varience inflation factor'''
##vif1 = pd.DataFrame()
##vif1["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
##vif1["features"] = X.columns
##vif1.round(1)

X = np.column_stack((df['Claims'], df['Insured'], df['Bonus']))
y = df['Payment']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()       
#--------------

X1 = df['Insured'].values.reshape(-1,1)
y1 = df['Claims'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(X1, y1)

''' varience inflation factor'''
##vif2 = pd.DataFrame()
##vif2["VIF Factor"] = [variance_inflation_factor(X1.values, i) for i in range(X.shape[1])]
##vif2["features"] = X1.columns
##vif2.round(1)

X1 = np.column_stack((df['Payment'], df['Insured'], df['Bonus']))
y1 = df['Claims']
X2 = sm.add_constant(X)
est = sm.OLS(y1, X2)
est2 = est.fit()
    #print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0])) 
    #print(est2.summary())




'''-> output presenting code '''
print("Column Names: ",list(df.columns))
display(df.head(4))
df.info()
print(df.describe().transpose())
display(k)
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))
print(est2.summary())
df.plot()
plt.show()
