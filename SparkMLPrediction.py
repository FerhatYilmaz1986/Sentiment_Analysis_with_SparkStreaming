#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!wget -q https://dlcdn.apache.org/spark/spark-3.0.3/spark-3.0.3-bin-hadoop2.7.tgz
#!tar xf spark-3.0.3-bin-hadoop2.7.tgz
#!pip install findspark
#!pip install pyspark


# In[1]:


get_ipython().system('pwd')


# In[2]:


get_ipython().system('java -version')


# In[3]:


import os
#os.environ["SPARK_HOME"] = "/Users/ferhatyilmaz/opt/anaconda3/lib/python3.9/site-packages/pyspark"
os.environ["SPARK_HOME"] ="/Users/ferhatyilmaz/Downloads/spark-3.0.3-bin-hadoop2.7"
os.environ["PYSPARK_PYTHON"] = "/Users/ferhatyilmaz/opt/anaconda3/envs/SPark/bin/python3.5"
#os.environ['SPARK_LOCAL_IP']="localhost" 
#os.environ['JAVA_HOME']="/usr/libexec/java_home -v 1.8"


# In[4]:


#import os
#os.environ["SPARK_HOME"] = "/content/spark-3.0.3-bin-hadoop2.7"
import findspark
findspark.init()


# In[5]:


import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
from pyspark.ml.classification import LogisticRegression
spark = SparkSession.builder.master('local[*]').config("spark.executor.memory", '8g').config("spark.driver.memory",'8g').appName('Training_model').getOrCreate()


# In[7]:


#spark = SparkSession.builder.appName('Training_model').getOrCreate()


# In[6]:


training_set = spark.read.csv('training.1600000.processed.noemoticon.csv', header=False, inferSchema= True, sep=',')
validation_set = spark.read.csv("testdata.manual.2009.06.14.csv", header=False, inferSchema= True, sep=',')


# In[7]:


training_set.show(5)


# In[8]:


validation_set.show(5)


# In[9]:


train_df = training_set.withColumnRenamed('_c0', 'sentiment').withColumnRenamed('_c5', 'tweet')
train_df.show(5)


# In[10]:


validation_df = validation_set.withColumnRenamed('_c0', 'sentiment').withColumnRenamed('_c5', 'tweet')
validation_df = validation_df.select('tweet')
validation_df.show(5)


# In[11]:


train_df.groupby('sentiment').count().show()


# In[12]:


train_df.filter((train_df.sentiment==0)|(train_df.sentiment==4)).count()


# In[13]:


from pyspark.ml.feature import (CountVectorizer, Tokenizer, 
                                StopWordsRemover, IDF)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline


# In[14]:


tokenizer = Tokenizer(inputCol = 'tweet', outputCol = 'token_text')
stop_remove = StopWordsRemover(inputCol = 'token_text', outputCol = 'stop_token')
count_vec = CountVectorizer(inputCol = 'stop_token', outputCol = 'c_vec')
idf = IDF(inputCol = 'c_vec', outputCol = 'tf_idf')
label_idx = StringIndexer(inputCol='sentiment', outputCol='label',handleInvalid='skip')
clean_up = VectorAssembler(inputCols = ['tf_idf'], outputCol = 'features')
pipeline = Pipeline(stages=[label_idx, tokenizer, stop_remove, count_vec, idf, clean_up])
pipeline_pred = Pipeline(stages=[tokenizer, stop_remove, count_vec, idf, clean_up])


# In[15]:


cleaner = pipeline.fit(train_df)
train_df = cleaner.transform(train_df)
train_df = train_df.select('label', 'features')
train_df.show(3)


# In[16]:


cleaner = pipeline_pred.fit(validation_df)
validation_df = cleaner.transform(validation_df)
validation_df = validation_df.select('features')
validation_df.show(3)


# In[17]:


train, test = train_df.randomSplit([0.7, 0.3])


# In[18]:


train, test = test.randomSplit([0.7, 0.3])


# In[19]:


train, test = test.randomSplit([0.7, 0.3])


# In[57]:


lr = LogisticRegression(labelCol = 'label',featuresCol='features', maxIter=10)
sentiment_detector = lr.fit(train)
predictions = sentiment_detector.transform(test)
#predictions.show(3)


# In[20]:


from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
sentiment_detector = dt.fit(train)
predictions = sentiment_detector.transform(test)


# In[58]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator()
print("Test Accuracy: " + str(evaluator.evaluate(predictions)))


# In[ ]:


#pipeline.save('/content/drive/MyDrive/mypipeline')


# In[ ]:


#pipeline = Pipeline.load('/content/drive/MyDrive/mypipeline')
#model = LogisticRegressionModel.load('/content/drive/MyDrive/mymodel')


# In[25]:


#Get tweets from spark streaming output
newtweets = spark.read.csv("newtweets.csv", header=False, inferSchema= True, sep=',')


# In[26]:


newtweets = newtweets.withColumnRenamed('_c0', 'tweet')
newtweets.show(5)


# In[27]:


cleaner = pipeline_pred.fit(newtweets)
newtweets = cleaner.transform(newtweets)
newtweets = newtweets.select('features')
newtweets.show(3)


# In[28]:


new_predictions = sentiment_detector.transform(newtweets)
new_predictions.show(3)


# In[31]:


new_predictions.write.parquet('pred.parquet')


# In[54]:


#Save predictins to S3
import boto3
import glob
#Creating Session With Boto3.
session = boto3.Session(aws_access_key_id='ASIAXMXG5UAVTHJOPCVL',aws_secret_access_key='LMbOgFBgrJm2UAlfjuXRnxfPD9ut+SSY5M8rJJWl',aws_session_token='FwoGZXIvYXdzEAkaDNsuia/I/1l+V9Cf4SK+ASDNLxYZvVBmSUSqha6GJOGC6zMPjREQWYZ26V+VjimwGBwdzcNoPxDYtBPjyTQwL6zQSEklts4pgzOt6rDSxH1ORhi43/7NG/+9QfvVjGo2UMqIK0DA9LlInEuvgmdcjjE2lOJjE61I6gI4FuXsr2gqNsO0x5TZP4HhywHjJ8qIdQde39QkkLPz1Xaovzm+16Y7v82+X+HbXl2NRAi34QJemiwK3nZDn/eMCVUEcVxdQiazHSWkMdats0/3l5MogI7/jQYyLZJL1YWs0TPQBiSV1yjleSQ+XbGmpnNUukCShQQtIxXrP5ntXHTpsAl+Bvfuug==')

#Creating S3 Resource From the Session.
s3 = session.resource('s3')
#s3 = boto3.resource('s3')
BUCKET = "ferhatsparkbucket"

for i in range(len(glob.glob("/Users/ferhatyilmaz/Downloads/pred.parquet/*.parquet"))):
    s3.Bucket(BUCKET).upload_file(glob.glob("/Users/ferhatyilmaz/Downloads/pred.parquet/*.parquet")[i], "pred.parquet")


# In[ ]:





# In[21]:


#spark.stop()


# In[ ]:




