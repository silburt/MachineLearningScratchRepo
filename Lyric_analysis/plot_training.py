#This script plots the loss vs. epoch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import glob

files = glob.glob('output/train_*.txt')
for file in files:
    lines = open(file, 'r', encoding="utf-8").readlines()
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    for i in range(len(lines)):
        l = lines[i]
        if 'val_loss' in l:
            try:
                values = l.split('-')
                train_loss.append(float(values[2].split(':')[1]))
                train_acc.append(float(values[3].split(':')[1]))
                val_loss.append(float(values[4].split(':')[1]))
                val_acc.append(float(values[5].split(':')[1]))
            except:
                pass

    x = range(1,len(val_loss)+1)
    plt.plot(x, train_loss, label='train_loss')
    plt.plot(x, val_loss, label='val_loss')
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.ylabel('categorical crossentropy loss')
    plt.legend(loc='upper right')

    plt.savefig('%s.png'%file.split('.txt')[0])
    plt.clf()
