#http://www.markhneedham.com/blog/2015/01/19/pythonnltk-finding-the-most-common-phrases-in-how-i-met-your-mother/
# https://inzaniak.github.io/pybistuffblog/posts/2017/04/20/python-count-frequencies-with-nltk.html
# https://stackoverflow.com/questions/14364762/counting-n-gram-frequency-in-python-nltk

from collections import Counter
import nltk
import numpy as np
#nltk.download('punkt')
import glob
import matplotlib.pyplot as plt

####### Helper Functions ###########
def name(dir):
    return dir.split('playlists/')[1]

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
            print "found spam: %s"%l
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
    #for top matches from dir[0], find corresponding matches in dir[1]
    for i in range(n_words):
        l = lbls1[i]
        try:
            _c, _p = data[1][n_grams-1][l], lbls2.index(l)
            counts2.append(_c)
            pos2.append(min(_p,max_pos))
        except:
            counts2.append(0)
            pos2.append(max_pos)
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
            except:
                counts1.append(0)
                pos1.append(max_pos)
            labels.append(l)
            pos2.append(i)

    return labels, [pos1,pos2], [counts1,counts2]

# https://stackoverflow.com/questions/11763613/python-list-of-ngrams-with-frequencies
####### Main Functions ###########
def get_stats(dir, norm, n_songs, print_=0):
    files = glob.glob('%s/*.txt'%dir)

    np.random.seed(42)
    seeds = np.random.randint(0,len(files),n_songs)
    
    cnt, bi_cnt, tri_cnt = Counter(), Counter(), Counter()
    words_per_song = 0
    for s in seeds:
        lyric = open(files[s],'r').read().lower().split()
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
    dirs = ['playlists/rock','playlists/hip-hop', 'playlists/country', 'playlists/pop']
    norm = 0
    n_songs = 800

    data = {}
    for i in range(len(dirs)):
        data[i] = get_stats(dirs[i],norm,n_songs)

    #get top common pairs
    combos = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    n_words, n_grams, max_pos = 50, 1, 100
    for i1,i2 in combos:
        d = [data[i1],data[i2]]
        labels, pos, counts = get_common_pairs(d, n_words, n_grams, max_pos)
        
        r_labels = np.arange(max(pos[0]))
        plt.plot(r_labels, r_labels)
        plt.plot(r_labels, 2*r_labels, 'g')
        plt.plot(2*r_labels, r_labels, 'g')
        plt.plot(pos[0], pos[1], '.')
        for i in range(len(pos[0])):
            plt.text(pos[0][i]+0.5, pos[1][i]+0.5, labels[i], size=8)
        name1, name2 = dirs[i1].split('playlists/')[1], dirs[i2].split('playlists/')[1]
        plt.xlabel('%s rank'%name1)
        plt.ylabel('%s rank'%name2)
        plt.xlim([0,max_pos+10])
        plt.ylim([0,max_pos+10])
        plt.savefig('images/wordcorr_%s_%s.png'%(name1,name2))
        plt.clf()



