#This script plots the loss vs. epoch
import matplotlib.pyplot as plt
import numpy as np

#file = 'output/long_novalid_1024lstm.txt'
file = 'output/sl100.txt'

lines = open(file,'r').readlines()
loss = []
for l in lines:
#    if ('Epoch' in l) and ('loss:' in l):
#        loss.append(float(l.split('Epoch')[0].split('loss:')[1]))
    if 'val_loss' in l:
        loss.append(float(l.split('val_loss:')[1]))

plt.plot(range(1,len(loss)+1), loss)
plt.xlabel('epoch')
plt.ylabel('categorical crossentropy loss')

#plt.plot([34,34],[0,5],'--')
#plt.ylim([2,3.3])

plt.savefig('%s.png'%file.split('.txt')[0])
plt.show()
