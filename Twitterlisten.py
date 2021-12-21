#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import socket
import json
import pymongo


# In[2]:


# Set up your credentials
access_token = ""
access_secret = ""
consumer_key = ""
consumer_secret = ""


# In[3]:


class TweetsListener(StreamListener):
    def __init__(self, csocket):
        self.client_socket = csocket
        

    def on_data(self, data):
        try:
            msg = json.loads( data )
            print(msg['text'].encode('utf-8'))
            #self.db.stream.insert_one(msg)
            producer.send(topic_name, msg['text'].encode('utf-8'))
            self.client_socket.send( msg['text'].encode('utf-8'))
            return True

        except BaseException as e:
            print("Error on_data: %s" % str(e))
            return True

    def on_error(self, status):
        print(status)
        return True


# In[4]:


def sendData(c_socket):
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    twitter_stream = Stream(auth, TweetsListener(c_socket))
    twitter_stream.filter(track=['football'])


# In[ ]:


if __name__ == "__main__":
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)         # Create a socket object
    host = "localhost"     # Get local machine name
    port = 5555                 # Reserve a port for your service.
    s.bind((host, port))        # Bind to the port
    
    print("Listening on port: %s" % str(port))
    s.listen(5)                 # Now wait for client connection.
    c, addr = s.accept()        # Establish connection with client.
    sendData(c)


# In[7]:


#c.close()


# In[ ]:




