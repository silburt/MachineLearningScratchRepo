#This script plots the loss vs. epoch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import glob

#files = glob.glob('output/train_*.txt')
files = glob.glob('output/strip.txt')
for file in files:
    lines = open(file, 'r', encoding="utf-8").readlines()
    train_loss, val_loss = [], []
    for i in range(len(lines)):
        l = lines[i]
        if 'val_loss' in l:
            try:
                train_loss.append(float(lines[i-3].split('loss:')[1].split('\x08')[0]))
                val_loss.append(float(l.split('val_loss:')[1].split('\x08')[0]))
            except:
                pass

    x = range(1,len(val_loss)+1)
    plt.plot(x, train_loss, label='train_loss')
    plt.plot(x, val_loss, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('categorical crossentropy loss')
    plt.legend(loc='upper right')

    plt.savefig('%s.png'%file.split('.txt')[0])
    plt.clf()
