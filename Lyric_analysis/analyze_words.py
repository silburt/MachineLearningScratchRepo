#http://www.markhneedham.com/blog/2015/01/19/pythonnltk-finding-the-most-common-phrases-in-how-i-met-your-mother/
# https://inzaniak.github.io/pybistuffblog/posts/2017/04/20/python-count-frequencies-with-nltk.html
# https://stackoverflow.com/questions/14364762/counting-n-gram-frequency-in-python-nltk

from collections import Counter
from itertools import combinations
import nltk
import numpy as np
#nltk.download('punkt')
import glob
import matplotlib.pyplot as plt
from process_lyrics import *
np.random.seed(12)

def get_common_pairs(cnt1, pos1, cnt2, pos2, master_labels, n_words, pad):
    labels1, count1 = zip(*cnt1.most_common(10*n_words))
    labels2, count2 = zip(*cnt2.most_common(10*n_words))
    for i in range(n_words):
        l = labels1[i]
        if l not in master_labels[0:n_words]:
            try:
                _c, _p = cnt2[l], labels2.index(l)
                pos2.append(min(_p,(pad-3)*np.random.random()+n_words+1))
            except:
                pos2.append((pad-3)*np.random.random()+n_words+1)
            master_labels.append(l)
            pos1.append(i)
    return master_labels, pos1, pos2

# https://stackoverflow.com/questions/11763613/python-list-of-ngrams-with-frequencies
####### Main Functions ###########
def get_stats(dir, norm, n_songs, print_=0):
    files = glob.glob('%s/*.txt'%dir)
    cnt, bi_cnt, tri_cnt = Counter(), Counter(), Counter()
    words_per_song = 0
    n_processed_songs = 0
    for f in files:
        #lyric = get_clean_lyric(f)
        lyric = process_song(f, 'word')
        if len(lyric) > 15: #ignore instrumentals
            bi_lyric=nltk.FreqDist(nltk.ngrams(lyric, 2))
            tri_lyric=nltk.FreqDist(nltk.ngrams(lyric, 3))
            
            #normalize
            normalize = 1
            n_words = len(lyric)
            if norm == 1:
                normalize = float(n_words*n_songs)
            
            words_per_song += float(n_words)/float(n_songs)
            for l in lyric:
                cnt[l] += 1/normalize
            for b in bi_lyric:
                bi_cnt[b] += 1/normalize
            for t in tri_lyric:
                tri_cnt[t] += 1/normalize

            n_processed_songs += 1
        if n_processed_songs > n_songs:
            break

    if print_ == 1:
        print("########%s########"%dir)
        print("----Top single words:----")
        print(cnt.most_common(50))
        print("----Top paired words:----")
        print(bi_cnt.most_common(20))
        print("----Top tri-words:----")
        print(tri_cnt.most_common(20))
        print("Total number of unique words = %d"%len(cnt))
        print("Avg words per song = %f"%words_per_song)

    return [cnt, bi_cnt, tri_cnt, words_per_song]

####### Arguments ###########
if __name__ == '__main__':
    #dirs = ['playlists/edm','playlists/hip-hop','playlists/rock', 'playlists/country', 'playlists/pop']
    dirs = ['playlists/edm','playlists/pop']
    norm = 0
    n_songs = 996
    n_grams = 1
    n_common_words = 80

    #get data
    data = {}
    for i in range(len(dirs)):
        data[i] = get_stats(dirs[i],norm,n_songs)

    #plot common pairs between genres
    plot_common = 1
    if plot_common == 1:
        combos = list(combinations(range(len(dirs)),2))
        pad = 20
        for i1,i2 in combos:
            master_labels, pos1, pos2  = [], [], []
            master_labels, pos1, pos2 = get_common_pairs(data[0][n_grams-1], pos1, data[1][n_grams-1], pos2, master_labels, n_common_words, pad)
            master_labels, pos2, pos1 = get_common_pairs(data[1][n_grams-1], pos2, data[0][n_grams-1], pos1, master_labels, n_common_words, pad)
            pos = [pos1, pos2]
            
            #plot prep
            plt.figure(figsize=(8,6))
            plt.plot([0,n_common_words], [0,n_common_words])
            plt.plot([0,n_common_words], [0,n_common_words/2], 'g')
            plt.plot([0,n_common_words/2], [0,n_common_words], 'g')
            plt.fill_between([0,n_common_words+pad],[n_common_words,n_common_words],[n_common_words+pad,n_common_words+pad], facecolor='red',alpha=0.5, linewidth=0)
            plt.axvspan(n_common_words, n_common_words+pad, alpha=0.5, color='red')
            plt.plot(pos[0], pos[1], '.', color='black')
            for i in range(len(pos[0])):
                la = master_labels[i]
                if la == 'nigga':
                    la = 'ni**a'
                elif la == 'niggas':
                    la = 'ni**as'
                plt.text(pos[0][i]+0.2, pos[1][i]+0.2, la, size=10, rotation=20)
            name1, name2 = dirs[i1].split('playlists/')[1], dirs[i2].split('playlists/')[1]
            plt.xlabel('%s rank'%name1)
            plt.ylabel('%s rank'%name2)
            plt.xlim([0,n_common_words+pad])
            plt.ylim([0,n_common_words+pad])
            plt.savefig('images/wordcorr_%s_%s.png'%(name1,name2))
            plt.clf()

    #plot distribution of words
    plot_words = 1
    if plot_words == 1:
        ylabel = 'total counts'
        if norm == 1:
            ylabel = 'counts/(song*word)'
        for i in range(len(data)):
            labels, count = zip(*data[i][n_grams-1].most_common(n_common_words))
            #ran = np.arange(1,len(count)+1)
            #print(zip(labels,count,ran,ran**(0.55)*np.asarray(count)))
            #labels, count = spam_filter(labels, count)
            x, name = range(len(count)), dirs[i].split('playlists/')[1]
            plt.plot(x, count, label='avg. words per song=%d'%data[i][3])
            plt.xticks(x, labels, rotation='vertical',fontsize=11)
            plt.ylabel(ylabel)
            plt.legend(fontsize=12)
            plt.title(name)
            plt.savefig('images/worddist_%s.png'%name)
            plt.clf()


