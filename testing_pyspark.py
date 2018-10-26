#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:06:53 2018

@author: crystinrodrick
"""

import pyspark 
import re 
import pandas as pd 
from os.path import dirname
from os.path import join
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from pyspark import SparkConf, SparkContext

# Read the input file 
app_path = dirname(__file__)
raw_data = pd.read_csv(join(app_path, 'proposistions.csv'), skip_blank_lines=True)
tokenizer = WordPunctTokenizer()
tokens = tokenizer.tokenize((" ".join(raw_data['text'])).lower())

# Exclude stop words
fr_stops = set(stopwords.word('french'))
tokens = [ re.sub('[\W_]+', ' ', token) for token in tokens if token not in fr_stops]

# Initialize Spark context 
conf = SparkConf().setMaster("local").setAppName("Wordcount")
spark_rdd = sc.parallelize(tokens)

# Spark Map Reduce
Wordcounts = spark_rdd.flatMap(lambda line: line.split(" ")) \
                               .map(lambda word: (word, 1)) \
                               .reduceByKey(lambda a, b: a + b) \
                               .map(lambda x:(x[1],x[0])) \
                               .sortByKey(False)
                               
print(Wordcounts.take(100))
