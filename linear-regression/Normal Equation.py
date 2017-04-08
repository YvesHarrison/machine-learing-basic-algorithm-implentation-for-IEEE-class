import csv
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
#read file
with open('train.csv','rb') as csvfile:
    reader=csv.DictReader(csvfile)
    y=[row['price'] for row in reader]

with open('train.csv','rb') as csvfile:
    reader=csv.DictReader(csvfile)
    x=[row['sqft_living'] for row in reader]

csvfile.close()

m=len(x)
x_train=[int(i)/10 for i in x]
y_train=[int(i)/10 for i in y]
x0=[0.1 for i in x]
x_matrix = np.mat(x_train)
y_matrix = np.mat(y_train)
x0_matrix=np.mat(x0)
x0_T=x0_matrix.T
x_T=x_matrix.T
x_T=hstack((x0_T,x_T))

#training
x_matrix=x_T.T
tmp1=np.dot(x_matrix,x_T)
tmp2=np.linalg.inv(tmp1)
theta=tmp2*x_matrix*y_matrix.T
print theta


x1=np.linspace(0,14000)
theta0=double(theta[0][0])
theta1=double(theta[1][0])
y1= theta0+theta1*x1

plt.figure()
plt.plot(x1,y1)
plt.plot(x,y,'ro')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.title('Normal Equation')
plt.show()

with open('train.csv','rb') as csvfile:
    reader=csv.DictReader(csvfile)
    y_test=[row['price'] for row in reader]

with open('train.csv','rb') as csvfile:
    reader=csv.DictReader(csvfile)
    x_test=[row['sqft_living'] for row in reader]
csvfile.close()
x_test=[int(i) for i in x_test]
y_test=[int(i) for i in y_test]
n=len(x_test)
mse=0
for i in range(n):
    mse+=(theta0+theta1*x_test[i]-y_test[i])**2
mse=(mse/n)**0.5
print mse