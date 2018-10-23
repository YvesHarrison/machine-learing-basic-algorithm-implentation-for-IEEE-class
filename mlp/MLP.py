import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import time


def read_data(file_name):
    data_O = []
    data_X = []
    data_D = []
    with open(file_name, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for i, rows in enumerate(reader):
                row = rows
                #print row
                tmp = row
                for i in range(0,17):
                    if i > 0:
                        tmp[i] = int(tmp[i])
                # print row
                #row = [float(i) for i in row]
                #print tmp[1]
                if tmp[0] == 'D':
                    data_D.append(tmp)
                if tmp[0] == 'X':
                    data_X.append(tmp)
                if tmp[0] == 'O':
                    data_O.append(tmp)
        return data_D, data_O, data_X

def separation(data):
    label = []
    feature = []
    for i in range (0,len(data)):
        if data[i][0] == 'O':
            label.append(0)
        else:
            label.append(1)
        feature.append(data[i][1:])
    return label, feature

def logistic(x):
    return 1.0/(1.0 + np.exp(-x))

def backprop(x,y,W1,W2,b1,b2,hidden_layer_size,learning_rate):
    W1_grad = 0
    W2_grad = 0
    b1_grad = 0
    b2_grad = 0
    for j in range(0, len(x)):
        hidden = logistic(np.dot(W1, np.reshape(x[j], (len(x[0]), 1)))+b1)
        res = logistic(np.dot(W2, hidden)+b2)

        b2_grad += -res * (1 - res) * (y[j] - res)
        W2_grad += -res * (1 - res) * (y[j] - res) * hidden
        W1_grad += -np.dot(np.reshape(x[j].T, (len(x[0]), 1)),W2 * np.reshape(hidden * (1 - hidden),(1, hidden_layer_size))) * res * (1 - res) * (y[j] - res)
        b1_grad += -W2 * np.reshape(hidden * (1 - hidden),(1, hidden_layer_size)) * res * (1 - res) * (y[j] - res)
    #print b2_grad
    #print W2_grad
    #print W1_grad
    #print b1_grad
    return learning_rate * W1_grad.T/len(x),learning_rate * W2_grad.T/len(x),learning_rate * b1_grad.T/len(x),learning_rate * b2_grad.T/len(x)

def predict(x, W1, W2,b1,b2):
    x1=np.reshape(x,(len(x),1))
    layer_input = np.dot(W1,x1)+b1
    layer_output = logistic(layer_input)
    output = logistic(np.dot(W2,layer_output)+b2)
    if output>0.5:
        return 1
    else:
        return 0

def earlystop(x,y,xtt,ytt,W1,W2,b1,b2):
    right1=0
    right0=0
    for k in range(0, len(x)):
        if predict(x[k], W1, W2, b1, b2) == y[k]:
            right0 += 1
    for k in range(0, len(xtt)):
        if predict(xtt[k], W1, W2, b1, b2) == ytt[k]:
            right1 += 1
    return right0,right1

def train(x, y, xtt,ytt,hidden_layer_size,iterate,learning_rate):
    low = 0.0-math.sqrt(6.0/(len(x[0])+hidden_layer_size))
    high = math.sqrt(6.0/(len(x[0])+hidden_layer_size))
    W1 = np.random.uniform(low, high, (hidden_layer_size, len(x[0])))
    W2 = np.random.uniform(low, high, (1, hidden_layer_size))
    b1 = np.random.uniform(low, high, (hidden_layer_size, 1))
    b2 = np.random.uniform(low, high, (1, 1))
    #print W1,W2
    accuracy0 = 0
    accuracy1 = 0
    cnt=0
    py0=[]
    py1=[]
    px=[]
    for i in range (0,iterate):
        cnt+=1
        x1,x2,x3,x4= backprop(x,y,W1,W2,b1,b2,hidden_layer_size,learning_rate)
        W1 -= x1
        W2 -= x2
        b1 -= x3
        b2 -= x4
        it0,it1=earlystop(x,y,xtt,ytt,W1,W2,b1,b2)
        py0.append(it0)
        py1.append(it1)
        px.append(cnt)
        #print it1
        #if (it0>accuracy0 and it1<accuracy1):
            #print cnt, "iterations"
            #break
        #else:
            #accuracy0=it0
            #accuracy1=it1
        #break
    # print W1,W2
    plt.figure()
    plt.plot(px, py0)
    plt.plot(px, py1, 'ro')
    plt.xlabel('number of training round')
    plt.ylabel('number of right prediction')
    plt.title('prediction for test and training set')
    plt.show()
    return W1, W2, b1, b2

def test(x1, y1, x2, y2, hidden_layer_size,iterate,learning_rate):
    xt=x1[:int(len(x1)*0.7)]
    yt=y1[:int(len(x1)*0.7)]
    x = x1[int(len(x1) * 0.7):]
    y = y1[int(len(x1) * 0.7):]
    for i in range(0,int(0.7*len(x2))):
        xt.append(x2[i])
        yt.append(y2[i])
    for i in range(int(0.7*len(x2)),len(x2)):
        x.append(x2[i])
        y.append(y2[i])
    #print len(x1)
    right=0
    #print len(x1[:int(len(x1)*0.7)])
    print "Trainning set size is: ",len(xt)
    print "Testing set size is: ",len(x)
    W1, W2, b1, b2=train(np.array(xt), yt,np.array(x),y, hidden_layer_size,iterate,learning_rate)
    for i in range(0,len(x)):
        if predict(x[i], W1, W2, b1, b2) == y[i]:
            right+=1
    return float(right)/float(len(x))

np.random.seed(0)
data_d,data_o,data_x=read_data('letter-recognition.data')
data_d_label,data_d_feature = separation(data_d)
data_o_label,data_o_feature = separation(data_o)
data_x_label,data_x_feature = separation(data_x)
#print separation(data_d)
#print separation(data_o)
#print separation(data_x)
k=test(data_d_feature,data_d_label,data_o_feature,data_o_label,10,200,0.1)
print "accuracy for O and D is: ",k
k=test(data_x_feature,data_x_label,data_o_feature,data_o_label,10,200,0.1)
print "accuracy for O and X is: ",k