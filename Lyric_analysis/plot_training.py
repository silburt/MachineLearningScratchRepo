#This script plots the loss vs. epoch
import matplotlib.pyplot as plt
import numpy as np

file = 'output/long_novalid.txt'

lines = open(file,'r').readlines()
loss = []
for l in lines:
    if ('Epoch' in l) and ('loss:' in l):
        loss.append(float(l.split('Epoch')[0].split('loss:')[1]))
#    if 'val_loss' in l:
#        loss.append(float(l.split('val_loss:')[1]))

plt.plot(range(1,len(loss)+1), loss)
plt.xlabel('epoch')
plt.ylabel('categorical crossentropy loss')
plt.show()
