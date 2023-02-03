# create streamlit page with reading in the data topic_model_info and creating the plot for the streamlit page
import re

import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from gensim.models.phrases import Phraser, Phrases
# nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud

st.set_page_config(
    page_title="Topic Model EDA",
    page_icon="👋",
)


st.set_option('deprecation.showPyplotGlobalUse', False)
# read in the data

WPT = nltk.WordPunctTokenizer()
@st.cache
def tokenize(text):
    tokens = WPT.tokenize(text) #W+ means that either a word character (A-Za-z0-9_) or a dash (-) can go there.
    return tokens
    
review_df = pd.read_pickle("reviews.pkl")
sentiment_df = pd.read_csv("sentiment_labels.csv", index_col=0)
review_df['snippet_full_Token'] = review_df.loc[:,"snippet"].apply(lambda x: tokenize(x))

# set the title of the streamlit page
st.title('Topic Model Visualization')

# set the text of the streamlit page
st.text('Google Maps Yorumları için Topic Model Uygulanması ve Görselleştirilmesi')

# show the data as a table
st.table(review_df[["snippet", "snippet_Clean", "snippet_Lemma"]].sample(3))

# sum the token length for the snippet_full_token column
word_count = review_df['snippet_full_Token'].apply(len).sum()
word_count_lemma = review_df['snippet_Token'].apply(len).sum()
word_count_bigram = review_df['snippet_Bigram_Token'].apply(len).sum()
word_count_list = [word_count, word_count_lemma, word_count_bigram]
# barchart for the word count
st.subheader("Temizlik öncesi ve sonrası toplam kelime sayısı")
st.bar_chart(pd.DataFrame(word_count_list, columns=['Kelime Sayısı'], index=[
             'snippet_full_Token', 'snippet_Token', 'snippet_Bigram_Token']))


# mean length of the snippet column and snippet_clean column in terms of token length
st.subheader('Yorum bazında ortalama kelime sayısı')
st.bar_chart(pd.DataFrame(review_df[['snippet_full_Token', 'snippet_Token', 'snippet_Bigram_Token']].applymap(
    len).mean(), columns=['Ortalama uzunluk'], index=['snippet_full_Token', 'snippet_Token', 'snippet_Bigram_Token']))


st.subheader("Temizlik öncesi oluşan kelime bulutu")
text = review_df['snippet'].str.cat(sep=' ')
wordcloud = WordCloud(width=600, height=300, max_words=100, max_font_size=75, background_color='white',
                      collocations=False,
                      ).generate(text)
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
st.pyplot()

st.subheader("Yorumların pozitiflik ve negatiflik dağılımı")
sentiment_df["review_label"] = sentiment_df["review_label"].replace({1:"Pozitif", 0:"Negatif"})
st.bar_chart(sentiment_df["review_label"].value_counts(normalize=True))
st.caption('_3 yıldız üstü değerlendirmeler pozitif, 3 yıldız altı değerlendirmeler negatif olarak kabul edilmiştir.')