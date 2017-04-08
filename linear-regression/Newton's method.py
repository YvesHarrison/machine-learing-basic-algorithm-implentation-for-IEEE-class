import csv
import numpy as np
import matplotlib.pyplot as plt
import math

#read file
with open('train.csv','rb') as csvfile:
    reader=csv.DictReader(csvfile)
    y=[row['price'] for row in reader]

with open('train.csv','rb') as csvfile:
    reader=csv.DictReader(csvfile)
    x=[row['sqft_living'] for row in reader]

csvfile.close()
m=len(x)
x=[int(i) for i in x]
y=[int(i) for i in y]

#set learning rate
theta0=0.0
theta1=0.0
theta3=0.0
theta4=0.0
m=len(x)
theta=[0.0,0.0]
epsilon = 1
error0=0
flag=0
cnt=0

diff = [0.0, 0.0]
theta_m=np.mat([[0.0],[0.0]])
#training
while(1):
    error0=0
    theta3 = 0.0
    theta4 = 0.0
    theta5 = 0.0
    theta6 = 0.0
    for i in range (m):
        #theta1 = theta1 - (2 * theta1 * x[i] * x[i] + 2 * theta0 * x[i] - 2 * y[i]* x[i]) / (2 * x[i] * x[i])
        theta3+=(2 * theta1 * x[i] * x[i] + 2 * theta0 * x[i] - 2 * y[i]* x[i])
        theta5+=(2 * x[i] * x[i])
        #theta0=theta0-(2*theta0+2*theta1*x[i]-2*y[i])/2
        theta4 += (2*theta0+2*theta1*x[i]-2*y[i])
        theta6 +=2


        #f0=theta0+ theta1 * x[i]-y[i]
        #f1= (theta0+theta1*x[i]-y[i])*x[i]
        #J=np.mat([[f0],[f1]])
        #H=np.mat([[2.0,2*x[i]],[2*x[i],2*x[i]*x[i]]])#Singular matrix
        #print H
        #theta_m=theta_m-H.I*J
        #data=theta_m.tolist()
        #print data
        #theta0=float(data[0][0])
        #theta1 = float(data[1][0])
    theta1= theta1-theta3/theta5
    theta0= theta0-theta4/theta6

    for i in range (m):
        error0+=(theta1*x[i]+theta0-y[i])**2
    if(abs(theta1-flag)<0.1):
        break
    else:
        flag = theta1

x1=np.linspace(0,14000)
print theta1,theta0
y1= theta0+theta1*x1

plt.figure()
plt.plot(x1,y1)
plt.plot(x,y,'ro')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.title('Newton Method ')
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