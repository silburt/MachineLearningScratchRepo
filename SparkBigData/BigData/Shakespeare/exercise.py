from pyspark import SparkConf, SparkContext
import json

conf = SparkConf().setMaster("local").setAppName("Test_App")
sc = SparkContext(conf = conf)

rdd = sc.textFile('Complete_Shakespeare.txt')

# count number of lines
f = open('output.txt', 'w')

f.write("Number of lines is %d\n"%rdd.count())   # count number of lines
f.write("Number of words is %d\n"%rdd.flatMap(lambda line: line.split()).count()) #count number of words
f.write("Number of unique words is %d\n"%rdd.flatMap(lambda line: line.split()).distinct().count()) #counts number of distinct words

# count the occurrence of each word
# line.split = lines -> words
# map(lambda word:(word,1)) adds 1 to each occurrence of the word
# reduceByKey(lambda a,b:a+b) combines same keys, a,b:a+b adds two (word,1) entries together  
counts = rdd.flatMap(lambda line: line.split()).map(lambda word: (word,1)).reduceByKey(lambda a,b:a+b)
counts.saveAsTextFile('out')  #this is the directory, not the file. Must be a new directory

# show the top 5 most frequent words
# flip keys/values, then sort by Key (wayyy faster than sort by value)
# 'False' argument in sortByKey indicates descending order vs. ascending
# take top 5
f.write('Top 5 most popular words:\n')
#f.write(json.dumps(counts.map(lambda (a,b):(b,a)).sortByKey(False).take(5)))
f.write(json.dumps(counts.takeOrdered(5, key = lambda x: -x[1])))
f.close()


''' 
**Compiling notes**
interact                          #gives you non-login node 
module load spark
module unload python2
module load tensorflow/1.1.0
spark-submit exercise.py


**Other notes**
-RDDs are immutable and thus cannot be updated
-RDDs don't require Key/Value pairs (which are always (a,b). I.e. (a,b,c) is not going to fly), but extra transformations exist for Key/Value pairs.
-RDDs are its own data type and are not ordered. Thus e.g. taking the third entry in an RDD doesn't really make sense. Each time you do rdd.take(5) you can get different answers, it's more of a sampling
-reduceByKey always has two variables x,y which correspond to the values that are being combined in some way.
'''
