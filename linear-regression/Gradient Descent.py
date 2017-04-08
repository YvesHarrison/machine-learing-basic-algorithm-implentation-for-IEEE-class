import csv
import numpy as np
import matplotlib.pyplot as plt
#read file
with open('train.csv','rb') as csvfile:
    reader=csv.DictReader(csvfile)
    y=[row['price'] for row in reader]

with open('train.csv','rb') as csvfile:
    reader=csv.DictReader(csvfile)
    x=[row['sqft_living'] for row in reader]
csvfile.close()
#set learning rate
#alpha= 0.000000000001
alpha= 0.00000035
theta0=0.0
theta1=283.0
m=len(x)
diff=[0.0,0.0]
epsilon = 1
error0=0
error1=0
flag=0
cnt=0
#training
print m
while (cnt<100):
    cnt+=1
    diff=[0.0,0.0]
    for i in range (m):
        diff[0]+=theta0+ theta1 * int(x[i])-int(y[i])  
        diff[1]+=(theta0+theta1*int(x[i])-int(y[i]))*int(x[i])
        
    theta0=theta0-alpha/m*diff[0]  
    theta1=theta1-alpha/m*diff[1]
    #print theta0
    #print theta1
    error1=0
    for i in range(m):  
        error1+=(theta0+theta1*int(x[i])-int(y[i]))**2
    #print error1
    if abs(error1-error0)< epsilon:  
        break
    if abs(flag-error1)< epsilon:
        break
    else:
        flag=error1

print theta0
print theta1
x1=np.linspace(0,14000)
y1= theta0+theta1*x1

plt.figure()
plt.plot(x1,y1)
plt.plot(x,y, 'ro')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.title('Gradient Descent')
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

