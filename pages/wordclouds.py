import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud


st.set_option('deprecation.showPyplotGlobalUse', False)
# read in the data
topic_model_info = pd.read_csv('topic_model_info.csv')


# create wordcloud plot for the streamlit page and set the title of the plot
for k in topic_model_info.CustomName.unique():
    text = topic_model_info[topic_model_info.CustomName ==
                            k]['Document'].str.cat(sep=' ')
    wordcloud = WordCloud(width=400, height=200, max_font_size=50, max_words=100,
                          background_color="white", collocations=False).generate(text)
    plt.figure(figsize=(6, 4))
    plt.title(k.upper() + " Kümesi", fontsize=20)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot()

