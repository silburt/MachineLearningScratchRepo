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

#rank the words that are severely less prominent as still rank ordered list. I.e. your list shouldnt have integer 1000 showing up, make it len(list)+1
def get_common_pairs(data, dirs, n_words, n_grams):
    labels1, counts1 = zip(*data[0][n_grams-1].most_common(n_words))
    labels2, _ = zip(*data[1][n_grams-1].most_common(3*n_words))
    labels1, labels2 = list(labels1), list(labels2)
    pos1, pos2, counts2, i = [], [], [], 0
    for l in labels1:
        try:
            counts2.append(data[1][n_grams-1][l])
            pos2.append(labels2.index(l))
            pos1.append(i)
            print i, labels2.index(l), l
            i += 1
        except:
            print "%s not in %s lyric base, skipping word."%(l,dirs[1])
    return list(labels1), [pos1,pos2], [counts1, counts2]

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
        print "******Top single words:******"
        print cnt.most_common(50)
        print "******Top paired words:******"
        print bi_cnt.most_common(20)
        print "******Top tri-words:******"
        print tri_cnt.most_common(20)
        print "Total number of unique words = %d"%len(cnt)
        print "Avg words per song = %f"%words_per_song

    return [cnt, bi_cnt, tri_cnt, words_per_song]

####### Arguments ###########
if __name__ == '__main__':
    dirs = ['playlists/country','playlists/hip-hop']
    norm = 0
    n_songs = 800

    data = {}
    for i in range(len(dirs)):
        data[i] = get_stats(dirs[i],norm,n_songs)

    #plotting
    #f, (ax1, ax2) = plt.subplots(1,2, figsize=[14, 4])

    #get top common pairs
    n_words, n_grams = 50, 1
    labels, pos, counts = get_common_pairs(data, dirs, n_words, n_grams)
    
    r_labels = range(len(labels))
    plt.plot(r_labels, r_labels)
    plt.plot(pos[0], pos[1], '.')
    for i in range(len(pos[0])):
        plt.text(pos[0][i]+0.2, pos[1][i]+0.2, labels[i], size=10)
    plt.xlabel(dirs[0].split('playlists/')[1])
    plt.ylabel(dirs[1].split('playlists/')[1])
    plt.show()



