import streamlit as st
from textblob import TextBlob
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as xp

page_bg_img="""
<style>
[data-testid="stApp"]{
 background: linear-gradient(90deg, #F8D5C6, #2E537C)
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)

background_color = 'orange'

st.markdown(f'<style>.stApp {{ background-color:{background_color };}}</style>', unsafe_allow_html=True)
st.title("Election Prediction Using Sentiment Analysis")

if "button_clicked" not in st.session_state:
  st.session_state.button_clicked=False

def callback():
  st.session_state.button_clicked=True

def analyzer():
 
 bjp = pd.read_csv("bjp_reviews.csv")
 inc = pd.read_csv('inc_reviews.csv')

 col1,col2=st.columns(2)
 with col1:
   st.title("BJP TWEETS REVIEW")
   st.dataframe(bjp)
 with col2: 
   st.title("INC TWEETS REVIEW")
   st.dataframe(inc)

 bjp['Tweet'] = bjp['Tweet'].astype(str)
 inc['Tweet'] = inc['Tweet'].astype(str)

 def find_polarity(review):
   return TextBlob(review).sentiment.polarity
 
 bjp['Polarity'] = bjp['Tweet'].apply(find_polarity)
 inc['Polarity'] = inc['Tweet'].apply(find_polarity)
 
 bjp['Label'] = np.where(bjp['Polarity']>0,'positive','negative')
 bjp['Label'][bjp['Polarity']==0]='Neutral'

 inc['Label'] = np.where(inc['Polarity']>0,'positive','negative')
 inc['Label'][inc['Polarity']==0]='Neutral'

 neutral_bjp = bjp[bjp['Polarity']==0.0000]

 remove_neutral_bjp = bjp['Polarity'].isin(neutral_bjp['Polarity'])
 bjp.drop(bjp[remove_neutral_bjp].index,inplace=True)
 
 neutral_inc = inc[inc['Polarity']==0.0000]

 remove_neutral_inc = inc['Polarity'].isin(neutral_inc['Polarity'])
 inc.drop(inc[remove_neutral_inc].index,inplace=True)
 
 np.random.seed(10)
 remove_n = 8481
 drop_indices = np.random.choice(bjp.index,remove_n,replace=False)
 df_bjp = bjp.drop(drop_indices)

 #bjp
 np.random.seed(10)
 remove_n = 367
 drop_indices1 = np.random.choice(inc.index,remove_n,replace=False)
 df_inc = inc.drop(drop_indices1)

 bjp_count = df_bjp.groupby('Label').count()
 neg_bjp = (bjp_count['Polarity'][0] / 1000) * 100
 pos_bjp = (bjp_count['Polarity'][1] / 1000) * 100
 
 inc_count = df_inc.groupby('Label').count()
 neg_inc = (inc_count['Polarity'][0] / 1000) * 100
 pos_inc = (inc_count['Polarity'][1] / 1000) * 10

 with st.container():
   col1,col2,col3=st.columns(3)
   if col1.button("bar graph"):
    politicians = ['BJP','INC']

    neg_list = [neg_bjp,neg_inc]
    pos_list = [pos_bjp,pos_inc]

    st.title ("BAR GRAPH REPRESNTATION")
    fig_bar= go.Figure(
     data = [
       go.Bar(name='Negative',x=politicians,y=neg_list,marker=dict(color='red')),
       go.Bar(name='Positive',x=politicians,y=pos_list,marker=dict(color='green'))
       ]
     )
    fig_bar.update_layout(barmode='group')
    st.plotly_chart(fig_bar,use_container_width=True)
    #  st.markdown('<br>', unsafe_allow_html=True)
   if col2.button("pie diagram"):
     st.title("PIE DIAGRAM REPRESENTATION")
     st.write("BJP REVIEWS")
     fig_pie_bjp = go.Figure(
      go.Pie(labels=['positive','negative'],values=[pos_bjp,neg_bjp],marker=dict(colors=['green','red']))
      )
     st.plotly_chart(fig_pie_bjp)
     st.write("INC REVIEWS")
 
     fig_pie_inc = go.Figure(
     go.Pie(labels=['positive','negative'],values=[pos_inc,neg_inc],marker=dict(colors=['green','red']))
     )
     st.plotly_chart(fig_pie_inc)
   if col3.button("wordcloud"):
     st.set_option('deprecation.showPyplotGlobalUse', False)
     st.title("INC TWEET WORDCLOUD")
     # Load the CSV file
     df = pd.read_csv('inc_reviews.csv')

     # Combine all the text in the CSV file into a single string
     text = ' '.join(str(item) for item in df['Tweet'].tolist())
     
     # Create a WordCloud object
     wc = WordCloud(width=800, height=400, max_words=200, background_color='black')
     
     # Generate the word cloud
     wc.generate(text)
     plt.figure(figsize=(10, 5))
     plt.imshow(wc, interpolation='bilinear')
     plt.axis('off')
     st.pyplot()
     
     st.title("BJP TWEET WORDCLOUD")
     # Load the CSV file
     df2= pd.read_csv('bjp_reviews.csv')
     
     # Combine all the text in the CSV file into a single string
     text2 = ' '.join(str(item) for item in df2['Tweet'].tolist())
     
     # Create a WordCloud object
     wc2= WordCloud(width=800, height=400, max_words=200, background_color='black')
     
     # Generate the word cloud
     wc2.generate(text2)
     plt.figure(figsize=(10, 5))
     plt.imshow(wc2, interpolation='bilinear')
     plt.axis('off')
     st.pyplot()
 pass


# Create a transparent background button
if (
     st.button("Analyzer",on_click=callback) 
     or st.session_state.button_clicked
     ):
    analyzer()





