#!/usr/bin/env python
# coding: utf-8

# ## NEXT WORD PREDICTION USING RNN

# In[1]:


#importing necessary libraries
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.layers import Dense, LSTM ,Embedding
from keras.models import Sequential


# In[2]:


#source text
data='''Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems 
to extract knowledge and insights from noisy, structured and unstructured data, and apply knowledge from data across
a broad range of application domains. Data science is related to data mining, machine learning and big data.
Data science is a concept to unify statistics, data analysis, informatics, and their related methods in order to 
understand and analyse actual phenomena with data.It uses techniques and theories drawn from many fields within 
the context of mathematics, statistics, computer science, information science, and domain knowledge.'''


# In[3]:


#integer encode text
tokenizer=Tokenizer()
tokenizer.fit_on_texts([data])
encoded_data= tokenizer.texts_to_sequences([data])[0]
encoded_data


# In[4]:


#determining the vocabulary size
vocab_size=len(tokenizer.word_index)+1
print("Vocabulary Size is {}".format(vocab_size))


# In[5]:


#creating a sequence of words to fitthe model wth one word as input and one word as output
#create word- word sequences
sequences=list()
for i in range(1,len(encoded_data)):
    sequence=encoded_data[i-1:i+1]
    sequences.append(sequence)
    
print('Total Sequences: {}' .format(len(sequences)))   


# In[6]:


#input output pairs
sequences


# In[7]:


#split the sequences into input element X and output elememnt Y
sequences=np.asarray(sequences)
X,y=sequences[:,0],sequences[:,1]


# In[8]:


X[:5]


# In[9]:


y[:5]


# In[10]:


# one hot encode outputs
y = np_utils.to_categorical(y, num_classes=vocab_size)
# define model
y[:5]


# In[11]:


#Model Buildng
model=Sequential()
model.add(Embedding(vocab_size,10,input_length=1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())


# In[12]:


#compiling the network
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[13]:


model.fit(X,y,epochs=100)


# In[14]:


#generate a sequence from the model
def generate_seq(model,tokenizer,enter_text,n_predict):
    input,result=enter_text,enter_text
    #generate a fix number of words
    for i in range (n_predict):
        
        #encode the text as integers
        
        encoded=tokenizer.texts_to_sequences([input])[0]
        encoded=np.asarray(encoded)
        
        #predict a word in vocabulary
        predicted_word = np.argmax(model.predict(encoded))
        
        
        #map predicted word index to word
        out_word=''
        for word,index in tokenizer.word_index.items():
            if index==predicted_word:
                out_word=word
                break;
        #append to  input
        input, result= out_word, result+" "+out_word
            
    return result;   
        
        


# In[15]:


#evaluating 
print(generate_seq(model, tokenizer, 'statistics', 6))


# In[16]:


#evaluating 
print(generate_seq(model, tokenizer, 'informatics', 6))


# In[ ]:




