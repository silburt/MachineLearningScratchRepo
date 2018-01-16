#This script plots the loss vs. epoch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

seq_lengths = [25,50,75,100,125,150,175]
for seq in seq_lengths:
    #file = 'output/sl%d_word.txt'%f
    file = 'output/sl%d_character.txt'%f

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
