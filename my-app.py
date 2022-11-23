import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import altair as alt
import math
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from nltk.corpus import stopwords

st.set_page_config(layout="wide")

## Read Data ##
df = pd.read_csv("cleaned_yelp_data_st.csv")
df['date'] = pd.to_datetime(df['date'])
df['month_review'] = df['date'].dt.strftime('%Y-%m')
df['Weekday'] = df['date'].dt.day_name()

st.title("Food and Service Aspect-Based Sentiment Analysis")
st.write("This web-application aims to provide users a 2 part service. Below are two tabs, one which allows you to explore the supplied dataset further through an exploratory data analysis and the other which allows you to have an outputted prediction of Food and Service polarity given a text input.")
st.write("The dataset was provided by Yelp through this link: https://www.yelp.com/dataset. We have filtered the dataset to contain reviews with only food and service related reviews.")
st.write("Below is a look into the dataset.")
st.dataframe(df[['business_id', 'review_id', 'stars', 'date', 'text', 'name', 'city', 'state', 'categories', 'nouns', 'adjs', 'sentiment']].head(3))
tab1, tab2 = st.tabs(["Data Visualization", "Review Prediction"])
model1 = TFDistilBertForSequenceClassification.from_pretrained("food_model")
model2 = TFDistilBertForSequenceClassification.from_pretrained("service_model")

def get_next_adj(pos_tag, index_target):
    for i, j in enumerate(pos_tag[index_target+1:]):
        if j in ['JJ', 'JJR', 'JJS', 'VBD', 'VBN', 'RB', 'RBR', 'RBS', 'NN', 'NNP']:
            return i+index_target+1
stop_words = set(stopwords.words('english'))

def get_sent(text, food_service):
    food_corpus = ['food','dish','burger','chicken','sauce',\
                'fry','taste','cheese','flavor','salad','love','try',\
                'sandwich','pizza', 'lunch', 'breakfast',\
                'dinner', 'delicious', 'hoagie', 'yummy']
    service_corpus = ['clean','location','staff','place',\
                    'well','friendly','service','order',\
                    'time', 'wait','come','customer', 'nice', \
                    'table', 'customer service',\
                    'waiter', 'waitress', 'he was', 'she was']
    s = ''
    for sentence in text.replace('!', '.').split("."):
        food = False
        service = False

        if any(w.lower() in sentence.lower() for w in food_corpus):
            food = True
        if any(r.lower() in sentence.lower() for r in service_corpus):
            service = True
        if food == True and service == False:
            s += sentence + '.'
        if food == True and service == True: # if contains both then do something 
            tokenized = sent_tokenize(sentence) # need to tokenize sentence that contains both 
            for i in tokenized:
                wordsList = nltk.word_tokenize(i)
                wordsList = [w for w in wordsList if not w in stop_words]
                tagged = nltk.pos_tag(wordsList)
            unzipped = list(zip(*tagged))
            if food_service == 'food':
                corpus = food_corpus
            else:
                corpus = service_corpus
            for i in corpus:
                try: 
                    index_target = unzipped[0].index(i)
                    # if sentence starts with adjective # if corpus is last
                    # sometimes sentence will start with a differennt POS but follow immediately with an adjective
                    if index_target == 0: # if corpus is first
                        s += ' ' + ' '.join([unzipped[0][index_target], unzipped[0][index_target+1]]) + '.'
                    elif index_target == (len(unzipped[0])-1):
                        print()
                        s += ' ' + ' '.join([unzipped[0][index_target-1], unzipped[0][index_target]]) + '.'                    
                    elif unzipped[1][0] in ['JJ', 'JJR', 'JJS', 'VBD', 'VBN', 'RB', 'RBR', 'RBS', 'NN','NNP'] or unzipped[1][1] in ['JJ', 'JJR', 'JJS', 'VBD', 'VBN', 'RB', 'RBR', 'RBS', 'NN', 'NNP']: 
                        s += ' ' + ' '.join([unzipped[0][index_target-1], unzipped[0][index_target+1]]) + '.'
                    else: 
                        index_adj = get_next_adj(unzipped[1], index_target)
                        s += ' ' + ' '.join([unzipped[0][index_target], unzipped[0][index_adj]]) + '.'
                except: 
                    pass
    return s

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def make_prediction(text, food_service):
    if food_service == 'food':
        text = get_sent(text, 'food')
    else:
        text = get_sent(text, 'service')
    p = tokenizer.encode(text,
        truncation=True,
        padding=True,
        return_tensors="tf")

    if food_service == 'food':
        tf_output = model1.predict(p)[0]
    else:
        tf_output = model2.predict(p)[0]
    tf_prediction = tf.nn.softmax(tf_output, axis=1)
    labels = ['Negative','Positive']
    label = tf.argmax(tf_prediction, axis=1)
    label = label.numpy()
    return labels[label[0]]

def get_sentiment(food, service):
    if food == 'Positive':
        food_sentiment =  " 👍 "
    else:
        food_sentiment =  " 👎 "
    if service == 'Positive':
        service_sentiment = " 👍 "
    else:
        service_sentiment = " 👎 "
    return food_sentiment, service_sentiment

with tab2:
    st.header("Food and Service Sentiment Prediction")
    st.write('Using your inputted text, our pre-trained model will output a "👍" or "👎" for food and service opinions revealed in your review.')
    with st.form("my_form", clear_on_submit=True):
        text = st.text_area("Please input a review that include food and/or service opinions.", help = 'Type or copy and paste a review then click the "Get Prediction" button to recieve your output. Balloons will appear if you are successful', placeholder = 'The food was delicious, but the staff was rude.')
        submitted = st.form_submit_button("🥨 Get Prediction 🥧")
    if submitted:
        with st.container():
            st.write(text)
            food = make_prediction(text, food_service='food')
            service = make_prediction(text, food_service='service')
            food_sentiment, service_sentiment = get_sentiment(food, service)
            st.write("Food: " + food_sentiment + " Service: " + service_sentiment)
            st.balloons()
         
    
with st.sidebar:
    st.header("Data Visualization Filters")
    st.subheader("Filter Businesses")
    check_all = st.checkbox('I would like to see all', value=True)
    if check_all:
        with tab1:
            st.caption("Showing visualizations for all businesses and all locations.")
            row1_spacer1, row1_1, row1_spacer2 = st.columns((.2, 7.1, .2))
            with row1_1:
                wordcloud = WordCloud(width=1600, height=800, random_state=0, max_words=500, background_color='white')
                st.header("Word Cloud For Reviews")
            row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
            with row2_1: 
                st.write("A word cloud is a collection of words where the bolder and bigger the word is witin the image, the more frequent the word is used within the given text.")
                ## Wordcloud for all reviews nouns ##
                pos_tag_select = st.selectbox("Please select the part of speech you'd like to analyze", ('Nouns', 'Adjectives'))
                sentiment_select = st.selectbox('Please select the type of sentiment for reviews.', ('Positive', 'Negative', 'Both'))
                if sentiment_select == 'Positive' or sentiment_select == 'Negative':
                    sentiment = sentiment_select
                else: 
                    sentiment = ''
                
                if pos_tag_select == 'Nouns':
                    pos_tag = pos_tag_select.lower()
                else:
                    pos_tag = 'adjs'
            with row2_2:
                st.header('')
                st.subheader("Top " + pos_tag_select + " in " + sentiment + " Reviews")
                if sentiment_select  == 'Both':
                    wordcloud.generate(str(set(df[pos_tag.lower()])))
                    fig = plt.figure(figsize=(18,10))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.tight_layout(pad=10)
                    plt.show()
                    st.pyplot(fig)
                elif sentiment_select == 'Negative': 
                    ## Wordcloud for neg reviews nouns ##
                    wordcloud.generate(str(set(df[pos_tag.lower()][df['sentiment'] == 0])))
                    fig = plt.figure(figsize=(18,10))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.tight_layout(pad=10)
                    plt.show()
                    st.pyplot(fig)
                else:
                    ## Wordcloud for pos reviews nouns ##
                    wordcloud.generate(str(set(df[pos_tag.lower()][df['sentiment'] == 1])))
                    fig = plt.figure(figsize=(18,10))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.tight_layout(pad=10)
                    plt.show()
                    st.pyplot(fig)
            
            ## summary of data ##
            st.subheader("Summary of Data")
            st.write("The table below describes the numerical columns of the data.")
            with st.expander("See explanation"):
                st.write("Count will show you the number of rows within the data.")
                st.write("Mean offers you the average of the values in its column.")
                st.write("std stands for standard deviation which is how far the values of the column deviates from each other. ")
                st.write("Min and max are the minimum and maximum values of the given column.")
                st.write("25%, 50%, and 75% give you the percentile where 50% is the median and 25% is the lower percentile and 75% is the upper percentile.")
            st.dataframe(df.describe(), use_container_width=True)
            
            ## reviews per month ##
            df['date'] = pd.to_datetime(df['date'])
            df['month_review'] = df['date'].dt.strftime('%Y-%m')
            gb_month = df.groupby(['month_review'])['review_id'].count().to_frame()
            gb_month.reset_index(inplace=True)
            gb_month.rename(columns = {'month_review':'Month', 'review_id': 'Count of Reviews'}, inplace = True)
            
            st.header("Reviews per month")
            st.line_chart(gb_month, x="Month", y="Count of Reviews")
        
        
            ## reviews by weekday ##
            df['Weekday'] = df['date'].dt.day_name()
            gb_weekday = df.groupby(['Weekday'])['review_id'].count().to_frame()
            gb_weekday.reset_index(inplace=True)
            
            row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
            with row3_1:
                st.header("Reviews by Weekday")
            row4_spacer1, row4_1, row4_spacer2, row4_2, row4_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
            with row4_1:
                gb_weekday['Weekday'] = pd.Categorical(gb_weekday['Weekday'], ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
                gb_weekday.sort_values("Weekday", inplace=True)
                gb_weekday.rename(columns = {'review_id': 'Count of Reviews'}, inplace = True)
                st.dataframe(gb_weekday)
                st.caption("A drilldown of the graph")
            with row4_2:
                st.altair_chart(alt.Chart(gb_weekday).mark_bar().encode(x=alt.X('Weekday', sort=None), y='Count of Reviews', tooltip = ['Weekday', 'Count of Reviews']), use_container_width=True)
        
        
            ## most reviewed businesses ##
            row5_spacer1, row5_1, row5_spacer2 = st.columns((.2, 7.1, .2))
            with row5_1:
                st.header("Most Reviewed Businesses")
            row6_spacer1, row6_1, row6_spacer2, row6_2, row6_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
            with row6_1:
                n = st.slider('Choose a number to see Top N Reviewed Businesses', 1, 30, 10)
            with row6_2:
                st.subheader("Top " + str(n) + " Reviewed Businesses")
                business_cnt = pd.DataFrame(df['name'].value_counts())[:n]
                business_cnt.reset_index(inplace=True)
                business_cnt.rename(columns = {'index': 'Business Name' ,'name': 'Count of Reviews'}, inplace=True)
                st.altair_chart(alt.Chart(business_cnt).mark_bar().encode(x=alt.X('Business Name', sort=None), y='Count of Reviews', tooltip = ['Business Name', 'Count of Reviews']), use_container_width=True)
        
                
                
            ## map, location of businesses ##
            row7_spacer1, row7_1, row7_spacer2 = st.columns((.2, 7.1, .2))
            with row7_1:
                st.header("Location of Businesses Reviewed")
            row8_spacer1, row8_1, row8_spacer2, row8_2, row8_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
            with row8_1:
                state_select = st.selectbox('Option to Filter Reviews by State', (np.append(['All'], np.unique(df['state']).tolist())), 0)
                if state_select == 'All':
                    st.dataframe(df[['city', 'state', 'name']].drop_duplicates().sort_values(['state']))
                else:
                    st.dataframe(df[['city', 'state', 'name']][df['state'] == state_select].drop_duplicates())
            with row8_2:
                if state_select == 'All':
                    st.map(df[['latitude', 'longitude']].drop_duplicates())
                else: 
                    st.map(df[['latitude', 'longitude']][df['state'] == state_select].drop_duplicates())
            
            
            ## show state or city ##
            st.header("Most Reviewed States/Cities")
            row9_spacer1, row9_1, row9_spacer2, row9_2, row9_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))   
            with row9_1:
                location = st.radio( "Distribution by: ", ('State', 'City'))
                if location == 'State':
                    n = st.slider('Choose a number to see Top N States with Most Reviewed.', 1, len(np.unique(df['state'])), 5)
                else: 
                    n = st.slider('Choose a number to see Top N Cities with Most Reviews.', 1, 30, 20)
            with row9_2:    
                if location == 'State':
                    state_cnt = pd.DataFrame(df['state'].value_counts())[:n]
                    state_cnt.reset_index(inplace=True)
                    state_cnt.rename(columns = {'index': 'State' ,'state': 'Count of Reviews'}, inplace = True)
                    st.subheader("Top " + str(n) + " States with Most Reviews")
                    st.altair_chart(alt.Chart(state_cnt).mark_bar().encode(x=alt.X('State', sort=None), y='Count of Reviews', tooltip = ['State', 'Count of Reviews']), use_container_width=True)
                else:
                    city_cnt = pd.DataFrame(df['city'].value_counts())[:n]
                    city_cnt.reset_index(inplace=True)
                    city_cnt.rename(columns = {'index': 'City' ,'city': 'Count of Reviews'}, inplace = True)
                    st.subheader("Top " + str(n) + " Cities with Most Reviews")
                    st.altair_chart(alt.Chart(city_cnt).mark_bar().encode(x=alt.X('City', sort=None), y='Count of Reviews', tooltip = ['City', 'Count of Reviews']), use_container_width=True)
                    
            
            ## top categories ## 
            category_list = df['categories'].str.split(', ').tolist()
            def flatten(listsoflists):
                category_list = []
                for i in listsoflists:
                    if isinstance(i,list): category_list.extend(flatten(i))
                    else: category_list.append(i)
                return category_list
            d = Counter(flatten(category_list))
            category_df = pd.DataFrame.from_dict(d, orient='index').reset_index()
            category_df = category_df.rename(columns={'index':'Category', 0:'Count of Reviews'}).sort_values(by='Count of Reviews', ascending=False)
            st.header("Top Business Categories Reviewed")
            row10_spacer1, row10_1, row10_spacer2, row10_2, row10_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))   
            with row10_1:
                n_categories = st.slider('Choose a number to see Top N Business Categories Reviewed.', 1, 30, 5)
            with row10_2:
                top_n_categories = category_df[:n_categories]
                st.subheader("Top " + str(n_categories) + " Business Categories")
                st.altair_chart(alt.Chart(top_n_categories).mark_bar().encode(x=alt.X('Category', sort=None), y='Count of Reviews', tooltip = ['Category', 'Count of Reviews']), use_container_width=True)
                
                
            ## Distribution of Stars ##
            star_list = list(df.stars)
            star_count = {
                '1': star_list.count(1.0),
                '2': star_list.count(2.0),
                '3': star_list.count(3.0),
                '4': star_list.count(4.0),
                '5': star_list.count(5.0)
            }
            star_names = list(star_count.keys())
            star_values = list(star_count.values())
            stars = pd.DataFrame({'Ratings': star_names, 'Count of Stars': star_values})
            row11_spacer1, row11_1, row11_spacer2, row11_2, row11_spacer3  = st.columns((.2, 2.3, .4, 2.3, .2)) 
            with row11_1:
                st.header("Distribution of Stars Rating")
                st.bar_chart(stars, x='Ratings', y='Count of Stars',use_container_width=True)
                st.caption("Note that there are no 3 star ratings, for the sake of our model we removed rows with a rating of 3 to prevent sentiment ambiguity.")
            
            
            ## Distribution of Sentiment ##
            sentiment_dist = pd.DataFrame(df['sentiment'].value_counts())
            sentiment_dist.reset_index(inplace=True)
            sentiment_dist['index'].replace([1, 0], ['Positive', 'Negative'], inplace=True)
            sentiment_dist.rename(columns={'index': 'Sentiment', 'sentiment': 'Count of Reviews'}, inplace=True)
            with row11_2: 
                st.header("Distribution of Reviews by Sentiment")
                st.bar_chart(sentiment_dist, x='Sentiment', y='Count of Reviews',use_container_width=True)
                st.caption("We filtered the original dataset to obtain a datset with a relatively even ratio of positive to negative reviews.")
                
                
            ## Sentiment vs length of review ##
            def num_words(tokenized_string):
                return len(tokenized_string[2:][:-2].split("', '"))
            df['word_count'] = df['tokenized'].apply(num_words)
            review_length_df = df[['word_count','sentiment']]
            review_length_df0 = review_length_df[review_length_df['sentiment']==0]
            review_length_df1 = review_length_df[review_length_df['sentiment']==1]
            
            word_count0 = review_length_df0['word_count'].sort_values(ascending = True)
            word_count1 = review_length_df1['word_count'].sort_values(ascending = True)

            review_length_0 = dict(Counter(word_count0))
            review_length_1 = dict(Counter(word_count1))
            
            max_review_length0 = max(review_length_0.keys())
            for i in range(1,max_review_length0+1):
                  if i not in review_length_0.keys():
                        review_length_0[i] = 0
            
            max_review_length1 = max(review_length_1.keys())
            for i in range(1,max_review_length1+1):
                  if i not in review_length_1.keys():
                        review_length_1[i] = 0
            mean_review_0 = np.mean(word_count0)
            mean_review_1 = np.mean(word_count1)
            stdev_review_0 = np.std(word_count0, ddof=1)
            stdev_review_1 = np.std(word_count1, ddof=1)
            
            st.header("Sentiment VS Length of Review")
            st.write("This graph shows you the common amount of words used in a review for a particular sentiment.")
            with st.expander("See explanation"):
                st.write("At the apex of the graph, the x-axis will show you the common amount of words that were written for reviews with the choosen sentiment. The y-axis will show you the amount of reviews that have a word count associated with the x-axis.")
            row12_spacer1, row12_1, row12_spacer2, row12_2, row12_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2)) 
            with row12_1:
                sentiment_radio = st.radio( "Choose a sentiment to analyze.", ('Positive', 'Negative'))
            with row12_2:
                sent_vs_len_0 = pd.DataFrame({'Word Count': review_length_0.keys(), 'Count of Reviews': review_length_0.values()})
                sent_vs_len_1 = pd.DataFrame({'Word Count': review_length_1.keys(), 'Count of Reviews': review_length_1.values()})
                if sentiment_radio == 'Negative':
                    st.subheader("Negative Sentiment VS Length of Review")
                    st.bar_chart(sent_vs_len_0, x='Word Count', y='Count of Reviews', use_container_width=True)
                else:
                    st.subheader("Positive Sentiment VS Length of Review")
                    st.bar_chart(sent_vs_len_1, x='Word Count', y='Count of Reviews', use_container_width=True)
            if sentiment_radio == 'Negative':
                st.text('Mean word count for "negative" reviews:                         '+ str(mean_review_0))
                st.text('Standard deviation of word count for "negative" reviews:        '+ str(stdev_review_0))
            else:
                st.text('Mean word count for "positive" reviews:                         '+ str(mean_review_1))
                st.text('Standard deviation of word count for "positive" reviews:        '+ str(stdev_review_1))
                    
            ## boxplot ##
            st.subheader('Statistical Distribution of Review Lengths by Sentiment in Yelp Dataset')
            col1, col2= st.columns([1, 1.5])
            with col1:
                fig, ax = plt.subplots(figsize=(4,4))
                ax.boxplot([word_count0,word_count1], vert=True, patch_artist=True, labels=['Negative','Positive'])
                plt.xlabel('Sentiment')
                plt.ylabel('Word Count')
                st.pyplot(fig)
            with col2: 
                st.write('The box represents the range where most of the data would lie, with the edges of the box being where the lower 25% and 75% of data lie and the middle line being the mean (50%). The lines sticking out the boxes represent where the rest of the data are, with the ends of the plot representing the minimum and maximum and the points on the end lines being outliers.')
                
    ## if not checked then do analysis for 2 businesses ##
    else: 
                
        st.subheader("Business(es) select")
        business_select = st.multiselect("Select your business or select yours along with another business you'd like to compare with.", pd.unique(df['name']).tolist(), max_selections=2)
        
        two = False
        if len(business_select) != 0:
            if len(business_select) > 1: 
                two=True
            st.subheader("Filter Business Locations")
            one_b = df[df['name'] == business_select[0]]
            if two == True:
                two_b = df[df['name'] == business_select[1]]
                both = df[(df['name'] == business_select[0]) | (df['name'] == business_select[1])]
                state_select_2 = st.multiselect("Narrow locations of all " + business_select[1] + "s.", (np.append(['All'], np.unique(two_b['state']).tolist())), default=['All'])
                
            state_select_1 = st.multiselect("Narrow locations of all " + business_select[0] + "s.", (np.append(['All'], np.unique(one_b['state']).tolist())), default=['All'])
            if ('All' in state_select_1 and len(state_select_1)>1) or ('All' in state_select_2 and len(state_select_2)>1):
                st.error('Please deselect "All" then continue', icon="🚨")
            
            if two==True:
                if 'All' not in state_select_1:
                    one_b = one_b[one_b['state'].isin(state_select_1)]
                elif 'All' not in state_select_2:
                    two_b = two_b[two_b['state'].isin(state_select_2)]
            else:
                if 'All' not in state_select_1:
                    df = df[df['state'].isin(state_select_1)]
                
            with tab1: 
                if two==True:
                    if 'All' in state_select_1 and 'All' in state_select_2:
                        st.caption("Showing visualizations for all locations.")
                    elif 'All' in state_select_1 and 'All' not in state_select_2:
                        st.caption("Showing visualizations for all locations of " + business_select[0] + " and " +
                                    ', '.join(state_select_2) + " locations of " + business_select[1])
                    elif 'All' not in state_select_1 and 'All' not in state_select_2:
                        st.caption("Showing visualizations for " + ', '.join(state_select_1) + " locations of " + business_select[0] + " and " + ', '.join(state_select_2) + " locations of " + business_select[1])
                    elif 'All' not in state_select_1 and 'All' in state_select_2:
                        st.caption("Showing visualizations for all locations of " + business_select[1] + " and " +
                                      ', '.join(state_select_1) + " locations of " + business_select[0])
                else:
                    if 'All' in state_select_1:
                        st.caption("Showing visualizations for all locations.")
                    else:
                        st.caption("Showing visualizations for " + ', '.join(state_select_1) + " locations of " + business_select[0])
                row14_spacer1, row14_1, row14_spacer2, row14_2, row14_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
                with row14_1:
                    st.header("Word Cloud For Reviews")
                    st.write("A word cloud is a collection of words where the bolder and bigger the word is witin the image, the more frequent the word is used within the given text.")
                    wordcloud = WordCloud(width=1600, height=800, random_state=0, max_words=500, background_color='white')
                    pos_tag_select_2 = st.selectbox("Please select the part of speech you'd like to analyze", ('Nouns', 'Adjectives'), key=2)
                    sentiment_select_2 = st.selectbox('Please select the type of sentiment for reviews.', ('Positive', 'Negative', 'Both'), key=3)
                    if sentiment_select_2 == 'Positive' or sentiment_select_2 == 'Negative':
                        sentiment2 = sentiment_select_2
                    else: 
                        sentiment2 = ''
                    
                    if pos_tag_select_2 == 'Nouns':
                        pos_tag2 = pos_tag_select_2.lower()
                    else:
                        pos_tag2 = 'adjs'
                with row14_2:
                    st.header('')
                    st.subheader("Top " + pos_tag_select_2 + " in " + sentiment2 + " Reviews")
                    if sentiment_select_2  == 'Both':
                        wordcloud.generate(str(set(one_b[pos_tag2.lower()])))
                        fig = plt.figure(figsize=(18,10))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        plt.tight_layout(pad=10)
                        st.subheader(business_select[0])
                        st.pyplot(fig)
                        
                        if two==True:
                            wordcloud.generate(str(set(two_b[pos_tag2.lower()])))
                            fig = plt.figure(figsize=(18,10))
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis('off')
                            plt.tight_layout(pad=10)
                            st.subheader(business_select[1])
                            st.pyplot(fig)
                    elif sentiment_select_2 == 'Negative': 
                        ## Wordcloud for neg reviews nouns ##
                        wordcloud.generate(str(set(one_b[pos_tag2.lower()][one_b['sentiment'] == 0])))
                        fig = plt.figure(figsize=(18,10))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        plt.tight_layout(pad=10)
                        st.subheader(business_select[0])
                        st.pyplot(fig)
                        
                        if two==True:
                            wordcloud.generate(str(set(two_b[pos_tag2.lower()][two_b['sentiment'] == 0])))
                            fig = plt.figure(figsize=(18,10))
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis('off')
                            plt.tight_layout(pad=10)
                            st.subheader(business_select[1])
                            st.pyplot(fig)
                    else:
                        ## Wordcloud for pos reviews nouns ##
                        wordcloud.generate(str(set(one_b[pos_tag2.lower()][one_b['sentiment'] == 1])))
                        fig = plt.figure(figsize=(18,10))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        plt.tight_layout(pad=10)
                        st.subheader(business_select[0])
                        st.pyplot(fig)
                        
                        if two==True:
                            wordcloud.generate(str(set(two_b[pos_tag2.lower()][two_b['sentiment'] == 1])))
                            fig = plt.figure(figsize=(18,10))
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis('off')
                            plt.tight_layout(pad=10)
                            st.subheader(business_select[1])
                            st.pyplot(fig)
                            
                ## summary of data ##
                st.header("Summary of Data")
                st.write("The table below describes the numerical columns of the data.")
                with st.expander("See explanation"):
                    st.write("Count will show you the number of rows within the data.")
                    st.write("Mean offers you the average of the values in its column.")
                    st.write("std stands for standard deviation which is how far the values of the column deviates from each other. ")
                    st.write("Min and max are the minimum and maximum values of the given column.")
                    st.write("25%, 50%, and 75% give you the percentile where 50% is the median and 25% is the lower percentile and 75% is the upper percentile.")
                if two==True:
                    row15_1, row15_2 = st.columns(2)
                    with row15_2:
                        st.subheader(business_select[1])
                        st.dataframe(two_b.describe()) 
                else: 
                    row15_1 , row15_2 = st.columns([1,0.1])
                with row15_1:
                    st.subheader(business_select[0])
                    st.dataframe(one_b.describe(), use_container_width=True)
                                    

                    
                ## reviews per month ##                
                st.header("Reviews per month")
                if two == True:
                    date_filter = False
                    with st.expander("Expand to filter reviews by date"):
                        date_filter = True
                        min_date = max(min(two_b.date.dt.date), min(one_b.date.dt.date))
                        max_date = min(max(two_b.date.dt.date), max(one_b.date.dt.date))
                        min_date_range = st.date_input("Minimum Date", min_date, min_value = min_date)
                        max_date_range = st.date_input("Maximum Date", max_date, max_value = max_date)
                
                if two==True:
                    row16_1, row16_2 = st.columns(2)
                    with row16_2:
                        if date_filter == True:
                            gb_month_2 = two_b[(two_b['date'].dt.date < max_date_range) & (two_b['date'].dt.date > min_date_range)].groupby(['month_review'])['review_id'].count().to_frame()
                        else:
                            gb_month_2 = two_b.groupby(['month_review'])['review_id'].count().to_frame()
                        gb_month_2.reset_index(inplace=True)
                        gb_month_2.rename(columns = {'month_review':'Month', 'review_id': 'Count of Reviews'}, inplace = True)
                        
                        st.subheader(business_select[1])
                        st.line_chart(gb_month_2, x="Month", y="Count of Reviews")
                else:
                    row16_1, row16_2 = st.columns([1,0.1])
                with row16_1:
                    if date_filter == True:
                        gb_month_1 = one_b[(one_b['date'].dt.date < max_date_range) & (one_b['date'].dt.date > min_date_range)].groupby(['month_review'])['review_id'].count().to_frame()
                    else:
                        gb_month_1 = one_b.groupby(['month_review'])['review_id'].count().to_frame()
                    gb_month_1.reset_index(inplace=True)
                    gb_month_1.rename(columns = {'month_review':'Month', 'review_id': 'Count of Reviews'}, inplace = True)
                    
                    st.subheader(business_select[0])
                    st.line_chart(gb_month_1, x="Month", y="Count of Reviews")
                
                ## reviews by weekday ##
                st.header("Reviews by Weekday")
                row17_spacer1, row17_1, row17_spacer2, row17_2, row17_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
                
                gb_weekday_1 = one_b.groupby(['Weekday'])['review_id'].count().to_frame()
                gb_weekday_1.reset_index(inplace=True)
                gb_weekday_1['Weekday'] = pd.Categorical(gb_weekday_1['Weekday'], ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
                gb_weekday_1.sort_values("Weekday", inplace=True)
                gb_weekday_1.rename(columns = {'review_id': 'Count of Reviews'}, inplace = True)
                
                if two==True:
                    gb_weekday_2 = two_b.groupby(['Weekday'])['review_id'].count().to_frame()
                    gb_weekday_2.reset_index(inplace=True)
                    gb_weekday_2['Weekday'] = pd.Categorical(gb_weekday_2['Weekday'], ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
                    gb_weekday_2.sort_values("Weekday", inplace=True)
                    gb_weekday_2.rename(columns = {'review_id': 'Count of Reviews'}, inplace = True)
                    
                with row17_1:
                    st.subheader(business_select[0])
                    st.dataframe(gb_weekday_1)
                    st.caption("A drilldown of the graph")
                    if two==True:
                        st.subheader(business_select[1])
                        st.dataframe(gb_weekday_2)
                        st.caption("A drilldown of the graph")
                with row17_2:
                    st.header('')
                    st.header('')
                    st.altair_chart(alt.Chart(gb_weekday_1).mark_bar().encode(x=alt.X('Weekday', sort=None), y='Count of Reviews', tooltip = ['Weekday', 'Count of Reviews']), use_container_width=True)
                    if two==True:
                        st.header('')
                        st.altair_chart(alt.Chart(gb_weekday_2).mark_bar().encode(x=alt.X('Weekday', sort=None), y='Count of Reviews', tooltip = ['Weekday', 'Count of Reviews']), use_container_width=True)
                        
                ## map, location of businesses ##
                st.header("Location of Businesses Reviewed")
                row18_spacer1, row18_1, row18_spacer2, row18_2, row18_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
                with row18_1:
                    if two == True:
                        if 'All' in state_select_1 and 'All' in state_select_2:
                            st.dataframe(both[['city', 'state', 'name']].drop_duplicates().sort_values(['state']))
# =============================================================================
#                         elif 'All' not in state_select_1 and 'All' in state_select_2 :
#                             st.dataframe(one_b[['city', 'state', 'name']].append(two_b[['city', 'state', 'name']], ignore_index=True).drop_duplicates().sort_values(['state']))
#                         elif 'All' in state_select_1 and 'All' not in state_select_2:
#                             st.dataframe(one_b[['city', 'state', 'name']].append(two_b[['city', 'state', 'name']]).drop_duplicates().sort_values(['state']))
#                         elif 'All' not in state_select_1 and 'All' not in state_select_2:
# =============================================================================
                        else:
                            st.dataframe(one_b[['city', 'state', 'name']].append(two_b[['city', 'state', 'name']], ignore_index=True).drop_duplicates().sort_values(['state']))
                    else:
                        st.dataframe(one_b[['city', 'state', 'name']].drop_duplicates().sort_values(['state']))
                            
                with row18_2:
                    if two == True:
                        if 'All' in state_select_1 and 'All' in state_select_2:
                            st.map(both[['latitude', 'longitude']].drop_duplicates())
# =============================================================================
#                         elif 'All' not in state_select_1 and 'All' in state_select_2:
#                             st.map(one_b[['latitude', 'longitude']].append(two_b[['latitude', 'longitude']]).drop_duplicates())
#                         elif 'All' in state_select_1 and 'All' not in state_select_2: 
#                             st.map(one_b[['latitude', 'longitude']].append(two_b[['latitude', 'longitude']]).drop_duplicates())
#                         elif 'All' not in state_select_1 and 'All' not in state_select_2 :
# =============================================================================
                        else:
                            st.map(one_b[['latitude', 'longitude']].append(two_b[['latitude', 'longitude']]).drop_duplicates())
                    else:
                        st.map(one_b[['latitude', 'longitude']].drop_duplicates())
                        
                        
                ## show state or city ##
                st.header("Most Reviewed States/Cities")
                row19_spacer1, row19_1, row19_spacer2, row19_2, row19_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))   
                with row19_1:
                    location_1 = st.radio( "Distribution by: ", ('State', 'City'), key='loc1')
                    if two==True:
                        if location_1 == 'State' and len(np.unique(one_b['state'])) != 1:
                            num_states = max(len(np.unique(one_b['state'])), len(np.unique(two_b['state'])))
                            n = st.slider('Choose a number to see Top N States with Most Reviewed.', 1, num_states, math.ceil(num_states/2))
                        elif location_1 == 'City' and len(np.unique(one_b['city'])) != 1: 
                            num_cities = max(len(np.unique(one_b['city'])), len(np.unique(two_b['city'])))
                            if num_cities > 30:
                                num_cities = 30
                            n = st.slider('Choose a number to see Top N Cities with Most Reviews.', 1, num_cities, math.ceil(num_cities/2))
                        else: 
                            n = 1
                    else:
                        if location_1 == 'State' and len(np.unique(one_b['state'])) != 1:
                            num_states = len(np.unique(one_b['state']))
                            n = st.slider('Choose a number to see Top N States with Most Reviewed.', 1, num_states, math.ceil(num_states/2))
                        elif location_1 == 'City' and len(np.unique(one_b['city'])) != 1: 
                            num_cities = len(np.unique(one_b['city']))
                            if num_cities > 30:
                                num_cities = 30
                            n = st.slider('Choose a number to see Top N Cities with Most Reviews.', 1, num_cities, math.ceil(num_cities/2))
                        else: 
                            n = 1
                with row19_2:    
                    if location_1 == 'State':
                        state_cnt_1 = pd.DataFrame(one_b['state'].value_counts())[:n]
                        state_cnt_1.reset_index(inplace=True)
                        state_cnt_1.rename(columns = {'index': 'State' ,'state': 'Count of Reviews'}, inplace = True)
                        st.subheader("Top " + str(n) + " States with Most Reviews")
                        st.caption(business_select[0])
                        st.altair_chart(alt.Chart(state_cnt_1).mark_bar().encode(x=alt.X('State', sort=None), y='Count of Reviews', tooltip = ['State', 'Count of Reviews']), use_container_width=True)
                        
                        if two==True:
                            state_cnt_2 = pd.DataFrame(two_b['state'].value_counts())[:n]
                            state_cnt_2.reset_index(inplace=True)
                            state_cnt_2.rename(columns = {'index': 'State' ,'state': 'Count of Reviews'}, inplace = True)
                            st.subheader("Top " + str(n) + " States with Most Reviews")
                            st.caption(business_select[1])
                            st.altair_chart(alt.Chart(state_cnt_2).mark_bar().encode(x=alt.X('State', sort=None), y='Count of Reviews', tooltip = ['State', 'Count of Reviews']), use_container_width=True)
                    else:
                        city_cnt_1 = pd.DataFrame(one_b['city'].value_counts())[:n]
                        city_cnt_1.reset_index(inplace=True)
                        city_cnt_1.rename(columns = {'index': 'City' ,'city': 'Count of Reviews'}, inplace = True)
                        st.caption(business_select[0])
                        st.altair_chart(alt.Chart(city_cnt_1).mark_bar().encode(x=alt.X('City', sort=None), y='Count of Reviews', tooltip = ['City', 'Count of Reviews']), use_container_width=True)
                        
                        if two==True:
                            city_cnt_2 = pd.DataFrame(two_b['city'].value_counts())[:n]
                            city_cnt_2.reset_index(inplace=True)
                            city_cnt_2.rename(columns = {'index': 'City' ,'city': 'Count of Reviews'}, inplace = True)
                            st.subheader("Top " + str(n) + " Cities with Most Reviews")
                            st.caption(business_select[1])
                            st.altair_chart(alt.Chart(city_cnt_2).mark_bar().encode(x=alt.X('City', sort=None), y='Count of Reviews', tooltip = ['City', 'Count of Reviews']), use_container_width=True)
                        
                        
                ## Distribution of Stars ##
                st.header("Distribution of Stars Rating")
                if two==True:
                    row20_1, row20_2 = st.columns(2)
                    
                    star_list_2 = list(two_b.stars)
                    star_count_2 = {
                        '1': star_list_2.count(1.0),
                        '2': star_list_2.count(2.0),
                        '3': star_list_2.count(3.0),
                        '4': star_list_2.count(4.0),
                        '5': star_list_2.count(5.0)
                    }
                    star_names_2 = list(star_count_2.keys())
                    star_values_2 = list(star_count_2.values())
                    stars_2 = pd.DataFrame({'Ratings': star_names_2, 'Count of Stars': star_values_2})
                    with row20_2:
                        st.subheader(business_select[1])
                        st.bar_chart(stars_2, x='Ratings', y='Count of Stars')
                else:
                    row20_1, row20_2 = st.columns([1,0.1])    
                star_list_1 = list(one_b.stars)
                star_count_1 = {
                    '1': star_list_1.count(1.0),
                    '2': star_list_1.count(2.0),
                    '3': star_list_1.count(3.0),
                    '4': star_list_1.count(4.0),
                    '5': star_list_1.count(5.0)
                }
                star_names_1 = list(star_count_1.keys())
                star_values_1 = list(star_count_1.values())
                stars_1 = pd.DataFrame({'Ratings': star_names_1, 'Count of Stars': star_values_1})
                with row20_1:
                    st.subheader(business_select[0])
                    st.bar_chart(stars_1, x='Ratings', y='Count of Stars',use_container_width=True)
                    st.caption("Note that there are no 3 star ratings, for the sake of our model we removed rows with a rating of 3 to prevent sentiment ambiguity.")

                    
                ## Distribution of Sentiment ##     
                st.header("Distribution of Reviews by Sentiment")
                if two==True:
                    sentiment_dist_2 = pd.DataFrame(two_b['sentiment'].value_counts())
                    sentiment_dist_2.reset_index(inplace=True)
                    sentiment_dist_2['index'].replace([1, 0], ['Positive', 'Negative'], inplace=True)
                    sentiment_dist_2.rename(columns={'index': 'Sentiment', 'sentiment': 'Count of Reviews'}, inplace=True)
                    
                    row21_1, row21_2 = st.columns(2)
                    with row21_2:
                        st.subheader(business_select[1])
                        st.bar_chart(sentiment_dist_2, x='Sentiment', y='Count of Reviews',use_container_width=True)
                else:
                    row21_1, row21_2 = st.columns([1,0.1])    
                sentiment_dist_1 = pd.DataFrame(one_b['sentiment'].value_counts())
                sentiment_dist_1.reset_index(inplace=True)
                sentiment_dist_1['index'].replace([1, 0], ['Positive', 'Negative'], inplace=True)
                sentiment_dist_1.rename(columns={'index': 'Sentiment', 'sentiment': 'Count of Reviews'}, inplace=True)
                
                with row21_1:
                    st.subheader(business_select[0])
                    st.bar_chart(sentiment_dist_1, x='Sentiment', y='Count of Reviews',use_container_width=True)
                    st.caption("We filtered the original dataset to obtain a datset with a relatively even ratio of positive to negative reviews.")
