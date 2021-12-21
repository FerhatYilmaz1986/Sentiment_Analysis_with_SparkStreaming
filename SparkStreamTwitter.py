#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!wget -q https://dlcdn.apache.org/spark/spark-3.0.3/spark-3.0.3-bin-hadoop2.7.tgz
#!tar xzf spark-3.0.3-bin-hadoop2.7.tgz
#!pip install findspark


# In[1]:


import os
os.environ["SPARK_HOME"] ="/Users/ferhatyilmaz/Downloads/spark-3.0.3-bin-hadoop2.7"
os.environ["PYSPARK_PYTHON"] = "/Users/ferhatyilmaz/opt/anaconda3/envs/SPark/bin/python3.5"
#os.environ["PYSPARK_PYTHON"]="/usr/local/bin/python3"
import findspark
findspark.init()


# In[2]:


from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import desc


# In[3]:


sc = SparkContext()
ssc = StreamingContext(sc, 10 )
sqlContext = SQLContext(sc)


# In[4]:


socket_stream = ssc.socketTextStream("localhost", 5555)
lines = socket_stream.window( 20 )


# In[5]:


from collections import namedtuple
fields = ("tag", "count" )
Tweet = namedtuple('tweets', fields )


# In[6]:


# Use Parenthesis for multiple lines or use \.
( lines.flatMap( lambda text: text.split( " " ) ) #Splits to a list
  .filter( lambda word: word.lower().startswith("#") ) # Checks for hashtag calls
  .map( lambda word: ( word.lower(), 1 ) ) # Lower cases the word
  .reduceByKey( lambda a, b: a + b ) # Reduces
  .map( lambda rec: Tweet( rec[0], rec[1] ) ) # Stores in a Tweet Object
  .foreachRDD( lambda rdd: rdd.toDF().sort( desc("count") ) # Sorts Them in a DF
  .limit(10).registerTempTable("tweets") ) ) # Registers to a table.


# In[7]:


import time
from IPython import display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
# Only works for Jupyter Notebooks!
get_ipython().magic('matplotlib inline')


# In[8]:


ssc.start()


# In[9]:


top_10_tweets = sqlContext.sql( 'Select tag, count from tweets' )
top_10_df = top_10_tweets.toPandas()
top_10_df.to_csv('newtweets.csv')


# In[3]:


#ssc.stop()


# In[4]:


#sc.stop()


# In[ ]:




