#Importing the libraries that are needed
from flask import Flask,render_template,request
import pickle
import random
import joblib
import json
import nltk
import tflearn
import tensorflow
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()
import numpy
# loading a pickled data file
words=[]
labels=[]
training=[]
output=[]
words,labels,training,output = joblib.load(open('/Users/marcell/Python_Projects/Medium/Webapp/model/data.pkl, 'rb'))

# defining a network and loading a pretrained model into it.
tensorflow.reset_default_graph()
net=tflearn.input_data(shape=[None,len(training[0])])
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,len(output[0]),activation="softmax")
net=tflearn.regression(net)
model=tflearn.DNN(net)
model.load(f'C:/Users/nmayavan.EAD/bot/Scripts/model.tflearn')

#model = joblib.load(open('C:/Users/nmayavan.EAD/bot/Scripts/webapp/model/bot_model.pkl', 'rb'))
app = Flask(__name__, template_folder='/Users/marcell/Python_Projects/Medium/Webapp/templates/')
with open("intents.json") as file:
    data=json.load(file)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

#app = Flask(__name__, template_folder='../templates/')
#Global variables for persistence across methods (and requests)
model_input=""
model_output=""
@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index1.html') #,display_mode="none")
@app.route('/chat',methods=['GET'])
#methods=['GET']
def chat():
    while True:
        
    # retrieve global variables to store input and output
        global model_input
        global model_output
    #print("Start talking with the bot (type quit to stop)!")
    #while True:
        # get text from the incoming request (submitted on predict button click)
        #text = request.form['inputtext']
        inp=request.args.get('msg')
        if (inp.lower() == "quit"):
            return redirect("https://www.google.com", code=200) 
    #prediction = model.predict(text)
        results= model.predict([bag_of_words(inp, words)])
        results_index=numpy.argmax(results)
        tag=labels[results_index]
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        
    # store model input and output
    #model_input = text
    #model_output = random.choice(responses)
        return random.choice(responses)
    #return render_template('index1.html',model_output) 

if __name__ == '__main__':
    app.run(debug=True,use_reloader=True)
#debug=True,use_reloader=False