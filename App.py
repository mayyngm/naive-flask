from flask import Flask
from flask import render_template
from flask import request
import tweepy
from NaiveBayes import NaiveBayes as naivebayes
import pandas as pd

#run public: flask run --host=0.0.0.0

consumer_key = 'UheaaaRrgbQS7cv2lTmz7Zs11'
consumer_secret = 'SrfJrnw34ivgtv7fIsL09sHxQ1rot5uzO5ashT6GpdO0Lv67f4'
access_token = '988010457893036034-dKwURBlownfMabSyXb8NnVDnShIPsvG'
access_token_secret = 'fsUbI4QjYUVJUbTTjt2d94NJDKRvXTsEuJ6C78zChGPHx'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
tweetCount = 100


lastTweets = []
# created_at = []
jenisVaksin = ['sinovac', 'pfizer', 'astra', 'moderna']
hasil = {}
hasilDetail = []
app = Flask(__name__)

bayes = naivebayes()

@app.route('/datates',methods=['GET', 'POST'])
def index():
    global lastTweets
    global jenisVaksin
    global hasil
    global hasilDetail
    # global created_at
    if request.method == 'POST' and request.values.get('query') is not None:
        keyword = request.values.get('keyword')
        public_tweets = api.search_tweets(q=keyword+' -filter:retweets', count=tweetCount, lang='id')
        lastTweets = []
        dtHasil = []
        for tweet in public_tweets:
            isContainJenis = False
            for jenis in jenisVaksin:
                if jenis in tweet.text.lower():
                    isContainJenis = True
                    break
            if isContainJenis:
                tweetDate = str(tweet.created_at).split(' ')[0]
                lastTweets.append(tweet.text)
                dtHasil.append([tweetDate, tweet.text])
                # created_at.append(tweet.created_at)
        return render_template('index.html', data=dtHasil)
        # data2 = created_at
    elif request.method == 'POST' and request.values.get('proses') is not None:
        hasil = {}
        print(lastTweets)
        maxData = 0
        maxPositive = 0
        maxLabel = ''
        hasilDetail = []
        hasilPreprop = []
        for vaksin in jenisVaksin:
            filtered = []
            for tweet in lastTweets:
                if vaksin in str(tweet).lower():
                    filtered.append(str(tweet))
            hasil[vaksin], dtPredict = bayes.predict(filtered)
            hasilPreprop.append(bayes.getPreprop())
            for i in range(len(dtPredict)):
                dtPredict[i].append(vaksin)
            hasilDetail.append(dtPredict)
            if hasil[vaksin][0]>maxData:
                maxData = hasil[vaksin][0] 
            if hasil[vaksin][1]>maxData:
                maxData = hasil[vaksin][1]
            if hasil[vaksin][1]>maxPositive:
                maxPositive = hasil[vaksin][1]
                maxLabel = vaksin
            #print(filtered)
            #print(hasil[vaksin])
        hasil['max_data'] = maxData
        if maxLabel=='astra':
            maxLabel = 'astrazeneca'
        hasil['max_label'] = maxLabel

        return render_template('recommend.html', data=[hasil, hasilDetail], preprop = hasilPreprop)
    else:
        return render_template('index.html')

@app.route('/recommend')
def recommend():
    if len(hasil)>0:
        return render_template('recommend.html', data=[hasil,hasilDetail])
    else:
        return render_template('index.html')

@app.route('/')
def training():
    f = open('data_training/dt3b.csv', encoding="utf8")
    lines = f.readlines()
    allData = []
    dtTabel = []
    ind = 0
    for line in lines:
        parts = line.split(',')
        if ind>0:
            dtText = ''
            for i in range(1, len(parts)):
                dtText+=' '+parts[i]
            dtTabel.append([parts[0],dtText])
        ind+=1
    f.close()
    allData.append(dtTabel)
    allData.append(bayes.getTrainAcc())
    return render_template('training.html', data=allData)

if __name__ == "__main__":
    app.run(debug=True)