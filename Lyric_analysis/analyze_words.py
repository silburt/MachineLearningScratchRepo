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

#improve the plotting spacing
def update_max_pos(max_pos,max_pos_spacer,spacer=5):
    if max_pos_spacer == 0:
        return max_pos+spacer, 1
    else:
        return max_pos-spacer, 0

#get lyrics and remove some useless characters
def get_clean_lyric(f):
    lyric = open(f,'r').read().lower()
    lyric = lyric.replace(',','')
    lyric = lyric.replace(';','')
    lyric = lyric.replace('.','')
    return lyric.split()

#remove useless words
def spam_filter(labels, counts):
    spam = ['chorus','verse','produce']
    labels, counts = list(labels), list(counts)
    i, n_labels = 0, len(labels)
    while i < n_labels:
        l = labels[i]
        if any(ext in l for ext in spam):
            del labels[i]
            del counts[i]
            n_labels -= 1
        else:
            i += 1
    return labels, counts

#rank the words that are severely less prominent as still rank ordered list. I.e. your list shouldnt have integer 1000 showing up, make it len(list)+1
def get_common_pairs(data, n_words, n_grams, max_pos):
    lbls1, cnt1 = zip(*data[0][n_grams-1].most_common(10*n_words))
    lbls2, cnt2 = zip(*data[1][n_grams-1].most_common(10*n_words))
    lbls1, cnt1 = spam_filter(lbls1, cnt1)
    lbls2, cnt2 = spam_filter(lbls2, cnt2)
    
    pos1, pos2, counts1, counts2, labels, i = [], [], [], [], [], 0
    max_pos_spacer = 0      #vary max_pos so that things don't overlap when plotted
    #for top matches from dir[0], find corresponding matches in dir[1]
    for i in range(n_words):
        l = lbls1[i]
        try:
            _c, _p = data[1][n_grams-1][l], lbls2.index(l)
            counts2.append(_c)
            pos2.append(min(_p,max_pos))
            max_pos,max_pos_spacer = update_max_pos(max_pos,max_pos_spacer)
        except:
            counts2.append(0)
            pos2.append(max_pos)
            max_pos,max_pos_spacer = update_max_pos(max_pos,max_pos_spacer)
        labels.append(l)
        pos1.append(i)

    #vice versa
    for i in range(n_words):
        l = lbls2[i]
        if l not in lbls1[0:n_words]:
            try:
                _c, _p = data[0][n_grams-1][l], lbls1.index(l)
                counts1.append(_c)
                pos1.append(min(_p,max_pos))
                max_pos,max_pos_spacer = update_max_pos(max_pos,max_pos_spacer)
            except:
                counts1.append(0)
                pos1.append(max_pos)
                max_pos,max_pos_spacer = update_max_pos(max_pos,max_pos_spacer)
            labels.append(l)
            pos2.append(i)

    return labels, [pos1, pos2], [counts1, counts2]

# https://stackoverflow.com/questions/11763613/python-list-of-ngrams-with-frequencies
####### Main Functions ###########
def get_stats(dir, norm, n_songs, print_=0):
    files = glob.glob('%s/*.txt'%dir)
    
    cnt, bi_cnt, tri_cnt = Counter(), Counter(), Counter()
    words_per_song = 0
    n_processed_songs = 0
    for f in files:
        lyric = get_clean_lyric(f)
        if len(lyric) > 15: #ignore instrumentals
            bi_lyric=nltk.FreqDist(nltk.ngrams(lyric, 2))
            tri_lyric=nltk.FreqDist(nltk.ngrams(lyric, 3))
            
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
        print "########%s########"%dir
        print "----Top single words:----"
        print cnt.most_common(50)
        print "----Top paired words:----"
        print bi_cnt.most_common(20)
        print "----Top tri-words:----"
        print tri_cnt.most_common(20)
        print "Total number of unique words = %d"%len(cnt)
        print "Avg words per song = %f"%words_per_song

    return [cnt, bi_cnt, tri_cnt, words_per_song]

####### Arguments ###########
if __name__ == '__main__':
    dirs = ['playlists/edm','playlists/hip-hop','playlists/rock', 'playlists/country', 'playlists/pop']
    norm = 0
    n_songs = 800
    n_grams = 1
    n_common_words = 50

    #get data
    data = {}
    for i in range(len(dirs)):
        data[i] = get_stats(dirs[i],norm,n_songs)

    #plot common pairs between genres
    plot_common = 1
    if plot_common == 1:
        combos = list(combinations(range(len(dirs)),2))
        max_pos = 100
        for i1,i2 in combos:
            d = [data[i1],data[i2]]
            labels, pos, counts = get_common_pairs(d, n_common_words, n_grams, max_pos)
            
            r_labels = np.arange(max_pos)
            plt.plot(r_labels, r_labels)
            plt.plot(r_labels, 2*r_labels, 'g')
            plt.plot(2*r_labels, r_labels, 'g')
            plt.plot(pos[0], pos[1], '.')
            line, = plt.plot([0,max_pos],[max_pos,max_pos], '--')
            plt.plot([max_pos,max_pos],[0,max_pos], '--', color=line.get_color())
            for i in range(len(pos[0])):
                plt.text(pos[0][i]+0.5, pos[1][i]+0.5, labels[i], size=6)
            name1, name2 = dirs[i1].split('playlists/')[1], dirs[i2].split('playlists/')[1]
            plt.xlabel('%s rank'%name1)
            plt.ylabel('%s rank'%name2)
            plt.xlim([0,max_pos+10])
            plt.ylim([0,max_pos+10])
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
            labels, count = spam_filter(labels, count)
            x, name = range(len(count)), dirs[i].split('playlists/')[1]
            plt.plot(x, count, label='avg. words per song=%d'%data[i][3])
            plt.xticks(x, labels, rotation='vertical',fontsize=8)
            plt.ylabel(ylabel)
            plt.legend(fontsize=8)
            plt.title(name)
            plt.savefig('images/worddist_%s.png'%name)
            plt.clf()


