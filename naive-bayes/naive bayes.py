import csv
import numpy as np
import matplotlib.pyplot as plt
import math

#read file
with open('train.csv','rb') as csvfile:
    reader=csv.DictReader(csvfile)
    train_data=[row for row in reader]
csvfile.close()
with open('test.csv','rb') as csvfile:
    reader=csv.DictReader(csvfile)
    test_data=[row for row in reader]
csvfile.close()

cnt1=0
cnt2=0

#print train_data[0]["word_freq_conference"]
m=len(train_data)
n=len(test_data)
print n
#print train_data[0]
#print test_data[0]
for i in range(0,m):
    if(train_data[i]['is_spam']=='0'):
        cnt1+=1;
    else:
        cnt2+=1;
attrs = [key for key in train_data[0]]
prioy0=float(cnt1+1)/m
prioy1=float(cnt2+1)/m
prob={}
for attr in attrs:
    if attr=='is_spam':
        continue
    record = [float(i[attr]) for i in train_data]
    cnt3=0;
    cnt4=0;
    cnt5=0;
    cnt6=0;
    for i in range(0,m):
        if train_data[i]['is_spam']=='0':
            if record[i]>0:
                cnt3+=1
            else:
                cnt4+=1
        else:
            if record[i] > 0:
                cnt5 += 1
            else:
                cnt6 += 1
    prob[attr]=[float(cnt3)/cnt1,float(cnt4)/cnt1,float(cnt5)/cnt2,float(cnt6)/cnt2]
    print attr,prob[attr]
cnt_right=0
for i in range(0,n):
    P_0=prioy0
    P_1=prioy1
    for attr in attrs:
        if attr == 'is_spam':
            continue
        if float(test_data[i][attr])>0:
            P_0 = P_0 * prob[attr][0]
            P_1 = P_1 * prob[attr][2]
        else:
            P_0 = P_0 * prob[attr][1]
            P_1 = P_1 * prob[attr][3]
    #print i,P_0,P_1
    if(P_0>P_1 and test_data[i]['is_spam']=='0'):
        cnt_right+=1
    if (P_0 < P_1 and test_data[i]['is_spam'] == '1'):
        cnt_right += 1
print cnt_right