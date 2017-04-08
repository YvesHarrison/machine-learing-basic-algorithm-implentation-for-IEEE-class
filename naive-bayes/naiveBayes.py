import csv
import random
import matplotlib.pyplot as plt
import numpy

def getData(csv_file):
    csvfile = file(csv_file,'rb')
    reader = csv.DictReader(csvfile)
    column = [row for row in reader]
    csvfile.close()
    return column

def getProbability(train_data):
    attrs = [key for key in train_data[0]] #attributes in data
    p=10 #parition number
    Ni,N=11,2 #the value of features, the type of class
    #get priors
    is_spam = [int(i['is_spam']) for i in train_data]
    condi_prob = {}
    train_len,spam_num = float(len(is_spam)),float(sum(is_spam))
    good_num = train_len-spam_num
    condi_prob['prior']=[(good_num+1)/(train_len+N),(spam_num+1)/(train_len+N)]
    for attr in attrs:
        if attr=='is_spam':
            continue
        #get class conditional prob
        record = [float(i[attr]) for i in train_data]
        maxR,minR=max(record),min(record)
        #print attr,maxR,minR,len(record)
        width = (maxR+0.1-minR)/p
        #divide the data into p interval
        partition = [minR+i*width for i in range(p)]
        count = [[0 for i in range(p+1)] for j in range(2)]
        cnt=0
        for r in record:
            index = int(round((r-minR)/width))
            type_inx=0
            if(is_spam[cnt]>0):
                type_inx=1
            if(index>=p):
                index=p-1
            count[type_inx][index]+=1
            if r==minR:
                count[type_inx][-1]+=1
            cnt+=1

        count[0][0]-=count[0][-1]
        count[1][0]-=count[1][-1]
        for i in range(p+1):
            count[0][i]=float(count[0][i]+1)/(good_num+Ni)
            count[1][i]=float(count[1][i]+1)/(spam_num+Ni)
        condi_prob[attr]=[count[0],count[1],partition]
        #print condi_prob[attr],sum(count[0]),sum(count[1])

    condi_prob['is_spam']=is_spam
    return condi_prob

def find_inx(value,partition):
    l = len(partition)
    inx=l
    if value==partition[0]:
        return inx
    for i in range(1,l):
        if(value<=partition[i]):
            return i-1
    return inx

def classify(test_file,condi_prob):
    test_data = getData(test_file)
    #test_data=test_data[24:26]
    cnt=0
    Ni,N=10,2 #the value of features, the type of class
    test_l=len(test_data)
    wrong = 0
    equal = 0
    for row in test_data:
        prob = [0,0]
        prob[0]=condi_prob['prior'][0]
        prob[1]=condi_prob['prior'][1]
        res = int(row['is_spam'])
        for key in condi_prob:
            if key=='is_spam' or key=='prior':
                continue
            test_inx=find_inx(float(row[key]),condi_prob[key][2])
            prob[0]*=condi_prob[key][0][test_inx]
            prob[1]*=condi_prob[key][1][test_inx]
            #print prob[0],prob[1]
            #using larange smoothing
        predict = 0
        if prob[1]>prob[0]:
            cnt+=1
            predict=1
        if predict!=res:
            wrong+=1
        if prob[0]==prob[1]:
            equal+=1
    print cnt,wrong,test_l,equal

def main():
    train_data = getData('train.csv')
    condi_prob = getProbability(train_data)
    classify('test.csv',condi_prob)

if __name__ =="__main__":
    main()

