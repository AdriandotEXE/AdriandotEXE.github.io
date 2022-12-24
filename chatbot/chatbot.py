#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import SGD
import random


# # Creating a Chatbot via AI

# In[2]:


words=[]
classes=[]
documents=[]
ignore_words=['?','!']
data_file=open(r'C:\Users\adria\Downloads\Chatbot\Simple-Python-Chatbot-master\intents.json').read()
intents=json.loads(data_file)


# In[3]:


from nltk.corpus import stopwords

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #Every word retrieved is tokenized
        w=nltk.word_tokenize(pattern)
        words.extend(w)
               
        #Documents are added here
        documents.append((w,intent['tag']))
        #Classes are added to the class list here
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
                         


# In[4]:


#Convert words to lowercase then convert words into thier base meaning 
words=[lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words=sorted(list(set(words)))
classes= sorted(list(set(classes)))
print(len(documents), "documents")
print(len(words), "unique words that have been lemmatized", words)

pickle.dump(words,open(r'C:\Users\adria\Downloads\Chatbot\Simple-Python-Chatbot-master\words.pkl','wb'))
pickle.dump(classes,open(r'C:\Users\adria\Downloads\Chatbot\Simple-Python-Chatbot-master\classes.pkl','wb'))


# # Creating Training Data

# In[5]:


import random
#Intialize All Training Data

training_list=[]
output_empty_list=[0]*len(classes)

for file in documents:
    
    #Intialize all words here
    satchel=[]
    
    #list of words that have been tokenized for the patterns list
    pattern_words=file[0]
    
    #Each words is brought down to its base and is used to represent words related to the base word
    pattern_words=[lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    #create a satchel of words that form an array of 1, if the word matches a word in the pattern list
    for w in words:
        satchel.append(1) if w in pattern_words else satchel.append(0)
        
    #output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row= list(output_empty_list)
    output_row[classes.index(file[1])]=1
    training_list.append([satchel,output_row])
    
#Mix all features and convert them into a np.array

random.shuffle(training_list)
training_list=np.array(training_list,dtype=object)
#Train and Test lists are created here

x_train= list(training_list[:,0])
y_train=list(training_list[:,1])

print("All training data has been created")    


# # Creating Neural Network Model

# In[6]:


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(x_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.AdrianEXE', hist)

print("model created")


# # Cleaning Data

# In[10]:


def sentence_cleanser(sentence):
    # tokenize the pattern - split words into array
    words_in_sentence = nltk.word_tokenize(sentence)
    # lemmatize or convert words into base words
    words_in_sentence = [lemmatizer.lemmatize(word.lower()) for word in words_in_sentence]
    return words_in_sentence
# return satchel of words array: 0 or 1 for each word in the satchel that exists in the sentence

def tie(sentence, words, show_details=True):
    # tokenize the pattern
    words_in_sentence = sentence_cleanser(sentence)
    # bag of satchel - matrix of N words, vocabulary matrix
    satchel = [0]*len(words) 
    for words in words_in_sentence:
        for i,w in enumerate(words):
            if w == words: 
                # assign 1 if current word is in the vocabulary position
                satchel[i] = 1
                if show_details:
                    print ("found in satchel: %s" % w)
    return(np.array(satchel))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = tie(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# In[11]:


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res


# # Creating GUI using tkinter

# In[12]:


#Creating GUI with tkinter
import tkinter
from tkinter import *
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="arrow")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Helvetica 15 bold",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Helvetica 15 bold")


#EntryBox.bind("<Return>", send)
#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
base.mainloop()


# In[2]:


#Training Montage
python train_chatbot.py



# In[ ]:


#Showtime
python chatgui.py


# In[ ]:




