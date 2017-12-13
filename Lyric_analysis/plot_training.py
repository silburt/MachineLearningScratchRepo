#This script plots the loss vs. epoch
import matplotlib.pyplot as plt
import numpy as np

files = [4,6,8,10,12,15]
for f in files:
    #file = 'output/sl%d_word.txt'%f
    file = 'sl%d_word.txt'%f

    lines = open(file,'r').readlines()
    loss = []
    for l in lines:
    #    if ('Epoch' in l) and ('loss:' in l):
    #        loss.append(float(l.split('Epoch')[0].split('loss:')[1]))
        if 'val_loss' in l:
            try:
                loss.append(float(l.split('val_loss:')[1]))
            except:
                pass

    plt.plot(range(1,len(loss)+1), loss)
    plt.xlabel('epoch')
    plt.ylabel('categorical crossentropy loss')

    #plt.plot([34,34],[0,5],'--')
    #plt.ylim([2,3.3])

    plt.savefig('%s.png'%file.split('.txt')[0])
    plt.clf()
