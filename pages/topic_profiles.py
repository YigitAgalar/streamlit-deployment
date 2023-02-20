import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud

topic_data = pd.read_pickle('sentiment_profile.pkl')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.title('Şubeler')

il = st.sidebar.selectbox("Hangi ili görüntülemek istediğnizi seçiniz", topic_data['il'].unique())

sube = st.sidebar.selectbox(
    "İncelemek istediğiniz şubenin adresini seçiniz", topic_data[topic_data["il"]==il]['place_address'].unique())


st.title('Şube Profili')
st.subheader(sube)
st.subheader("Şubeye ait yorumların konularına göre pozitiflik oranı")
for topic in topic_data['CustomName'].unique():

    data_id = topic_data[topic_data['place_address']
                         == sube]['data_id'].values[0]
    grouped_df = topic_data[topic_data["CustomName"] == topic].groupby(
        "data_id").agg({"review_label": "mean"}).reset_index()

    progress = grouped_df[grouped_df["data_id"]
                          == data_id]["review_label"].values[0]
    st.subheader(topic.upper() + " KÜMESİ")
    st.write(
        f"Gelen yorumların %{round(progress*100,2)} 'ü şubenin *{topic}* alanı hakkında olumlu düşünüyor")
    st.progress(progress.astype(float))


st.title("Örnek Yorumlar")
st.subheader("Pozitif yorumlar")
positive_reviews = []
for topic in topic_data['CustomName'].unique(): 
    try:
        if len(topic_data[(topic_data["CustomName"] == topic) & (topic_data["data_id"] == data_id) & (topic_data["review_label"] == 1)]["snippet"].sample()) > 0:
            positive_reviews.append(
                f'{topic_data[(topic_data["CustomName"]==topic)&(topic_data["data_id"]==data_id)&(topic_data["review_label"]==1)]["snippet"].sample().values[0]}')
    except:
        positive_reviews.append("nan")
        continue


for topic in topic_data['CustomName'].unique():
  
   try:
       if len(topic_data[(topic_data["CustomName"]==topic)&(topic_data["data_id"]==data_id)&(topic_data["review_label"]==1)]["snippet"].sample())>0:
           st.success(f'{topic_data[(topic_data["CustomName"]==topic)&(topic_data["data_id"]==data_id)&(topic_data["review_label"]==1)]["snippet"].sample().values[0]} ')
   except:
       pass


st.subheader("Negatif yorumlar")


for topic in topic_data['CustomName'].unique():
  
   try:
       if len(topic_data[(topic_data["CustomName"]==topic)&(topic_data["data_id"]==data_id)&(topic_data["review_label"]==0)]["snippet"].sample())>0:
           st.warning(f'{topic_data[(topic_data["CustomName"]==topic)&(topic_data["data_id"]==data_id)&(topic_data["review_label"]==0)]["snippet"].sample().values[0]}')
   except:
       pass
