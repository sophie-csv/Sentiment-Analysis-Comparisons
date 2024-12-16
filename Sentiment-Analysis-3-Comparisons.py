#Importing basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import nltk


plt.style.use('ggplot')



df = pd.read_csv("local_pub_testimonials.csv")

# 1- Transformers model

from tqdm.notebook import tqdm
from tqdm.auto import tqdm
!pip install tqdm --upgrade
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

df = df.dropna(subset=["Comment"])

tokenizer = AutoTokenizer.from_pretrained("karimbkh/BERT_fineTuned_Sentiment_Classification_Yelp")
model = AutoModelForSequenceClassification.from_pretrained("karimbkh/BERT_fineTuned_Sentiment_Classification_Yelp")
pipe = pipeline("text-classification", model="karimbkh/BERT_fineTuned_Sentiment_Classification_Yelp")


def classify_text(text):
    try:
        if isinstance(text, str) and text.strip():  # Ensure text is a non-empty string
            return pipe(text[0:512])[0]['label']
        else:
            return 'Neutral'  # Handle empty or non-string inputs
    except Exception as e:
        print(f"Error processing text: {text[0:512]}. Error: {e}")
        return 'Error'  # Handle exceptions during classification

# Wrap the apply function with tqdm for a progress bar
tqdm.pandas(desc="Classifying" )  # Initialize tqdm for pandas apply
df['Sentiment'] = df['Comment'].progress_apply(classify_text)  # Use progress_apply instead of appl

df.Sentiment.value_counts()

# 2-TextBlob model

from textblob import TextBlob
!pip install textblob

def TBsentiment(review):
    output = TextBlob(review)
    if output.sentiment.polarity > 0:
        return 'LABEL_1'
    if output.sentiment.polarity < 0:
        return 'LABEL_0'
    else:
        return 'neutral'
    
df['TEXTBLOB'] = df['Comment'].apply(TBsentiment)


df.drop([ 'Published To', 'Comment Date'], axis=1)

# 3- VADER model
from nltk.sentiment.vader import SentimentIntensityAnalyzer
!pip install nltk 
nltk.download("all")
sia = SentimentIntensityAnalyzer()

df['index'] = np.arange(1, len(df) + 1)

res = {}
for i, row in tqdm(df.iterrows(),total=len(df)):
    comment = row['Comment']
    myid = row['index']
    res[myid] = sia.polarity_scores(comment)

    if res[myid]['pos'] > (res[myid]['neg'] or res[myid]['neu']):
        label = "LABEL_1"
    if res[myid]['neg'] > (res[myid]['pos'] or res[myid]['neu']):
        label = "LABEL_0"
    if res[myid]['neu'] > (res[myid]['pos'] or res[myid]['neg']):
        label = "NEUTRAL"

    res[myid]['VADER'] = label 

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})

df['VADER'] = vaders['VADER'].values
df.rename(columns={'Sentiment':'TRANSFORMER'}, inplace=True)
df.drop(['Unnamed: 0','Comment Date', 'Published To', 'index'], axis=1)

# Comparison
df


