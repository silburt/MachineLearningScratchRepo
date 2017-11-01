from collections import Counter
import glob

dir = 'playlists/country/'
files = glob.glob('%s*.txt'%dir)
cnt = Counter()
for f in files:
    lyric = open(f,'r').read().lower().split()
    for word in lyric:
        cnt[word] += 1

print cnt.most_common(10)
print len(cnt)
