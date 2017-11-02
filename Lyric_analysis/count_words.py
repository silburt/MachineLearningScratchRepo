#http://www.markhneedham.com/blog/2015/01/19/pythonnltk-finding-the-most-common-phrases-in-how-i-met-your-mother/
# https://inzaniak.github.io/pybistuffblog/posts/2017/04/20/python-count-frequencies-with-nltk.html
# https://stackoverflow.com/questions/14364762/counting-n-gram-frequency-in-python-nltk

from collections import Counter
import nltk
#nltk.download('punkt')
import glob

dir = 'playlists/country/'
norm = 1

files = glob.glob('%s*.txt'%dir)
cnt, bi_cnt, tri_cnt = Counter(), Counter(), Counter()
for f in files:
    lyric = open(f,'r').read().lower().split()
    bi=nltk.FreqDist(nltk.ngrams(lyric, 2))
    tri=nltk.FreqDist(nltk.ngrams(lyric, 3))
    for word in lyric:
        cnt[word] += 1
    for b in bi:
        bi_cnt[b] += 1
    for t in tri:
        tri_cnt[t] += 1

if norm==1:
    n_files = float(len(files))
    for key in cnt:
        cnt[key] = round(cnt[key]/n_files,2)
    for key in bi_cnt:
        bi_cnt[key] = round(bi_cnt[key]/n_files,3)
    for key in tri_cnt:
        tri_cnt[key] = round(tri_cnt[key]/n_files,3)

print "******Top single words:******"
print cnt.most_common(50)
print "******Top paired words:******"
print bi_cnt.most_common(20)
print "******Top tri-words:******"
print tri_cnt.most_common(20)
print "******Total number of unique words:******"
print len(cnt)

# https://stackoverflow.com/questions/11763613/python-list-of-ngrams-with-frequencies
