#This script plots the loss vs. epoch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import glob

files = glob.glob('output/train_*_drop*.txt')

for file in files:
    lines = open(file,'r').readlines()
    loss = []
    for l in lines:
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
