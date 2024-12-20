#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import openai
from openai import AzureOpenAI
from PIL import Image
from streamlit_option_menu import option_menu
import altair as alt
from datetime import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Set your OpenAI API key
openai_client = AzureOpenAI(
  api_key = "f21c398476b146988044b391bd5256df", # use your key here
  api_version = "2024-06-01", # apparently HKUST uses a deprecated version
  azure_endpoint = "https://hkust.azure-api.net" # per HKUST instructions
)


# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to get tweets based on keywords
REQUIRE_COLUMNS = ['conversation_id_str', 'user_id_str', 'created_at', 'lang', 'full_text', 
                   'bookmark_count', 'favorite_count', 'quote_count', 'reply_count', 'retweet_count']

def get_tweets_by_explore(keywords="finance", get_comment=True):
    headers = {
        "x-rapidapi-key": "625a7d077amshfb106d4c100ee6ap1a8c14jsn87cd99088de2",
        "x-rapidapi-host": "twitter241.p.rapidapi.com"
    }

    url = "https://twitter241.p.rapidapi.com/search-v2"
    querystring = {"type": "Top", "count": "50", "query": keywords}
    response = requests.get(url, headers=headers, params=querystring)
    data_dict = response.json()

    result = [r['content'] for r in data_dict['result']['timeline']['instructions'][0]['entries']]

    csv_data = []
    for i, r in enumerate(result):
        if 'itemContent' in r:
            if 'tweet_results' in r['itemContent']:
                if 'legacy' in r['itemContent']['tweet_results']['result']:
                    tweet = r['itemContent']['tweet_results']['result']['legacy']
                    if get_comment:
                        csv_data.append(tweet)
                        try:
                            csv_data += get_tweets_by_post_id(tweet['conversation_id_str'])
                        except:
                            pass
                    else:
                        csv_data.append(tweet)

                if 'tweet' in r['itemContent']['tweet_results']['result']:
                    tweet = r['itemContent']['tweet_results']['result']['tweet']['legacy']
                    if get_comment:
                        csv_data.append(tweet)
                        try:
                            csv_data += get_tweets_by_post_id(tweet['conversation_id_str'])
                        except:
                            pass
                    else:
                        csv_data.append(tweet)
        if 'items' in r:
            items = r['items']
            for item in items:
                if 'tweet_results' in item['item']['itemContent']:
                    if 'legacy' in item['item']['itemContent']['tweet_results']['result']:
                        tweet = item['item']['itemContent']['tweet_results']['result']['legacy']
                        if get_comment:
                            csv_data.append(tweet)
                            try:
                                csv_data += get_tweets_by_post_id(tweet['conversation_id_str'])
                            except:
                                pass
                        else:
                            csv_data.append(tweet)

                    if 'tweet' in item['item']['itemContent']['tweet_results']['result']:
                        tweet = item['item']['itemContent']['tweet_results']['result']['tweet']['legacy']
                        if get_comment:
                            csv_data.append(tweet)
                            try:
                                csv_data += get_tweets_by_post_id(tweet['conversation_id_str'])
                            except:
                                pass
                        else:
                            csv_data.append(tweet)

    df = pd.DataFrame(csv_data)
    df = df.loc[:, REQUIRE_COLUMNS]
    df = df.drop_duplicates(subset=['conversation_id_str', 'user_id_str', 'full_text', 
                                      'bookmark_count', 'favorite_count', 'quote_count', 
                                      'reply_count', 'retweet_count'], keep='first')

    df = df.reset_index()
    return df

# Function to analyze tweets and create a recommendation
def analyze_tweets(df):
    if df.empty:
        return df, "No tweets found for the given keywords."

    # Combine all tweet texts into a single string for recommendation generation
    combined_text = " ".join(df['full_text'].tolist())

    # Generate recommendation using OpenAI's model
    recommendation = generate_recommendation(combined_text)

    return df, recommendation


# Function to generate recommendation using OpenAI's model
def generate_recommendation(combined_text):
    prompt = f"You are a marketing campaign expert who works in a bank. Based on the tweets (i.e. combined_text), your goal is to suggest possible marketing campaign strategies for a bank that can create value for clients while bringing in revenue to the bank. Prioritize credit card services and tech-savvy. Please think step-by-step and be realistic. Structure your response into two sections: 1) Summary of tweets and 2) Recommendations.:\n\n{combined_text}"
    response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# Display Latest 3 Tweets based on user input
def display_latest_3_tweets(user_input):
    # Fetch the latest tweets based on user input
    df = get_tweets_by_explore(keywords=user_input)  # Pass user input to fetch tweets
    if df.empty:
        st.write("No latest tweets found.")
        return

    # Filter to show only tweets in English
    df = df[df['lang'] == 'en']

    # Check if there are any English tweets
    if df.empty:
        st.write("No latest tweets found in English.")
        return

    # Sort tweets by favorite_count and then by created_at to show most liked and latest first
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    df = df.sort_values(by=['favorite_count', 'created_at'], ascending=[False, False])  # Sort by likes first, then by latest

    # Limit to the latest 3 tweets
    latest_tweets = df.head(3)

    # Display tweets in boxes
    for index, row in latest_tweets.iterrows():
        tweet_time = row['created_at']
        days_ago = (datetime.now() - tweet_time).days

        # Extract the link to the original tweet from the tweet text
        full_text = row['full_text']
        if 'https://t.co/' in full_text:
            link_start = full_text.index('https://t.co/')
            tweet_text = full_text[:link_start].strip()  # Get the text before the link
            tweet_link = full_text[link_start:].strip()  # Get the link part
        else:
            tweet_text = full_text  # If no link, use full text
            tweet_link = ''  # No link available

        # Create the clickable link if it exists
        if tweet_link:
            st.markdown(
                f"""
                <div style="background-color: #D9D9D9; border: 1px solid #ccc; border-radius: 10px; padding: 10px; margin: 10px 0;">
                    <p>{tweet_text} <a href="{tweet_link}" target="_blank">[View Tweet]</a></p>
                    <p style="font-size: 12px; color: grey;">{days_ago} days ago</p>
                    <hr style="margin: 5px 0;">  <!-- Horizontal line -->
                    <p style="font-size: 12px; color: black;">Number of retweets: {format_number(row['retweet_count'])}</p>
                    <p style="font-size: 12px; color: black;">Number of replies: {format_number(row['reply_count'])}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color: #D9D9D9; border: 1px solid #ccc; border-radius: 10px; padding: 10px; margin: 10px 0;">
                    <p>{tweet_text}</p>
                    <p style="font-size: 12px; color: grey;">{days_ago} days ago</p>
                    <hr style="margin: 5px 0;">  <!-- Horizontal line -->
                    <p style="font-size: 12px; color: black;">Number of retweets: {format_number(row['retweet_count'])}</p>
                    <p style="font-size: 12px; color: black;">Number of replies: {format_number(row['reply_count'])}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


# Load FAQ content
def load_faq():
    faq_content = """
    ## Frequently Asked Questions
    **Last updated: December 16, 2024**

    **Why are we named HSBC-hat?**  
    The name "HSBC-hat" reflects our vision for the app: we want everyone in HSBC to engage with us (HSBC-hat). Additionally, the name symbolizes the versatility of the AI, which can wear multiple hats to assist professionals in finding the right answers by adopting the appropriate perspective for each query.\r\n\

    **What is HSBC-hat?**  
    HSBC-hat is an AI-powered tool designed specifically for HSBC professionals to gain insights from the latest social trends on X (previously known as Twitter). It helps users stay informed and inspired, enabling them to design effective marketing campaigns that resonate with current conversations.

    **How does HSBC-hat work?**  
    Simply enter a topic or keyword related to your marketing interests, and HSBC-hat will fetch the latest relevant tweets. This allows you to quickly access real-time insights and trends, streamlining the research process for your campaigns.

    **What topics can HSBC-hat cover?**  
    HSBC-hat is not limited to any specific area; it can pull in tweets on a wide range of topics mentioned on X (previously known as Twitter). This flexibility allows users to explore various themes and trends relevant to their marketing strategies.

    **How many tweets can I fetch at a time?**  
    The app is configured to fetch up to 50 tweets per query. This capability ensures that you receive a substantial amount of data to analyze. The system is also scalable, allowing for increased tweet retrieval if necessary through our RapidAPI subscription.

    **How accurate is HSBC-hat's AI?**  
    While HSBC-hat leverages advanced AI to provide insights, it is important to note that AI can sometimes produce inaccuracies or "hallucinations." We recommend asking questions step-by-step and employing chain-of-thought techniques to refine your queries. For example, instead of asking, "What are the trends?", consider asking, "What are the recent trends in social media marketing?" This approach can yield more focused and useful results.

    **How can I provide feedback on HSBC-hat?**  
    We value your input! Please send any feedback or suggestions to our professor, Joon Nak Choi, at joonnakchoi@ust.hk. Your insights are crucial for enhancing the HSBC-hat experience for all users.
    """
    return faq_content

# Function to display the latest tweets
def format_number(num):
    """Format number to #1,00,000 format."""
    return f"{num:,}"

def display_latest_tweets():
    # Show loading message
    st.write("Getting tweets may take a few seconds. HSBC-hat is working at its full speed...")

    # Fetch the latest tweets
    df = get_tweets_by_explore()  # Fetch latest tweets with keyword 'Banking and financials'
    if df.empty:
        st.write("No latest tweets found.")
        return

    # Filter to show only tweets in English
    df = df[df['lang'] == 'en']

    # Check if there are any English tweets
    if df.empty:
        st.write("No latest tweets found in English.")
        return

    # Sort tweets by favorite_count and then by created_at to show most liked and latest first
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    df = df.sort_values(by=['favorite_count', 'created_at'], ascending=[False, False])  # Sort by likes first, then by latest

    # Display tweets in boxes
    for index, row in df.iterrows():
        tweet_time = row['created_at']
        days_ago = (datetime.now() - tweet_time).days

        # Extract the link to the original tweet from the tweet text
        full_text = row['full_text']
        if 'https://t.co/' in full_text:
            link_start = full_text.index('https://t.co/')
            tweet_text = full_text[:link_start].strip()  # Get the text before the link
            tweet_link = full_text[link_start:].strip()  # Get the link part
        else:
            tweet_text = full_text  # If no link, use full text
            tweet_link = ''  # No link available

        # Create the clickable link if it exists
        if tweet_link:
            st.markdown(
                f"""
                <div style="background-color: #D9D9D9; border: 1px solid #ccc; border-radius: 10px; padding: 10px; margin: 10px 0;">
                    <p>{tweet_text} <a href="{tweet_link}" target="_blank">[View Tweet]</a></p>
                    <p style="font-size: 12px; color: grey;">{days_ago} days ago</p>
                    <hr style="margin: 5px 0;">  <!-- Horizontal line -->
                    <p style="font-size: 12px; color: black;">Number of retweets: {format_number(row['retweet_count'])}</p>
                    <p style="font-size: 12px; color: black;">Number of replies: {format_number(row['reply_count'])}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color: #D9D9D9; border: 1px solid #ccc; border-radius: 10px; padding: 10px; margin: 10px 0;">
                    <p>{tweet_text}</p>
                    <p style="font-size: 12px; color: grey;">{days_ago} days ago</p>
                    <hr style="margin: 5px 0;">  <!-- Horizontal line -->
                    <p style="font-size: 12px; color: black;">Number of retweets: {format_number(row['retweet_count'])}</p>
                    <p style="font-size: 12px; color: black;">Number of replies: {format_number(row['reply_count'])}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
# Streamlit UI
st.set_page_config(page_title="HSBC-hat, Your reliable information agency!", layout="wide")

# Set background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://img.freepik.com/premium-photo/futuristic-white-gold-color-flowing-waving-background-hd-wallpaper_1000823-92883.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;
    }
    .faq-container {
        background-color: rgba(255, 255, 255, 0.9); /* White background with some transparency */
        border-radius: 10px;
        padding: 20px;
        margin: 20px;
        max-height: 600px; /* Limit height */
        overflow-y: auto; /* Enable vertical scrolling */
        user-select: none; /* Disable text selection */
    }
    .sidebar .sidebar-content {
        background-color: lightgrey; /* Light grey background for sidebar */
        border-radius: 10px; /* Rounded corners */
        padding: 10px;
        height: 100vh; /* Full height */
    }
    .footer {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 12px;
        color: grey;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add logo to the top right corner
st.markdown(
    """
    <div style="text-align: right;">
        <img src="https://businessbrainz.com/wp-content/uploads/2020/09/HSBC-Logo.png" alt="HSBC Logo" width="100" />
    </div>
    """,
    unsafe_allow_html=True
)
# Navigation menu using streamlit-option-menu
with st.sidebar:
    selection = option_menu("HSBC-hat", 
                           ["New Chat", "Chat History", "Discovery", "FAQ"],
                           icons=['plus-lg', 'envelope','search', 'question-circle'],
                           menu_icon="box-fill",  # Optional: icon for the menu
                           default_index=0,
                           styles={
                               "container": {"padding": "5!important", "background-color": "#ffffff", "height": "100vh"},  # Make sidebar cover full height
                               "icon": {"color": "black", "font-size": "20px"},
                               "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                               "nav-link-selected": {"background-color": "#d4bfa0"},  # Change selected option background color
                               "nav-link": {"color": "#800000"},  # Set text color to burgundy
                           })     

# Add footer text
st.markdown(
    "<div class='footer'>All Rights Reserved. @HKUST Business School MSc in Business Analytics</div>",
    unsafe_allow_html=True
)
                        

# New Chat Page
if selection == "New Chat":
    st.markdown("<h1 style='color: #800000; font-size: 48px;'>HSBC-hat is here for you!</h1>", unsafe_allow_html=True)

    # Create a container for the input box and button
    container = st.container()

    # Chat input box
    with container:
        user_input = st.text_input(" ", placeholder="Ask HSBC-hat any Keywords on Twitter", key="input", max_chars=100, label_visibility="collapsed")

        # Analyze button positioned at the bottom right
        analyze_button_container = st.container()
        with analyze_button_container:
            if st.button("▶️ Analyze", key="analyze_button"):
                with st.spinner("Fetching tweets..."):
                    df, recommendation = analyze_tweets(get_tweets_by_explore(user_input))

                    if not df.empty:
                        # Check if the maximum number of chat rooms has been reached
                        if len(st.session_state.chat_history) < 5:
                            # Add a new chat entry
                            st.session_state.chat_history.append({"name": user_input, "recommendation": recommendation, "tweets": df})
                        else:
                            st.warning("You have reached the maximum number of chat rooms. Please remove at least one chat to create a new chat.")

                        # Styled recommendation box
                        st.markdown(
                            """
                            <div style="
                                background-color: white; 
                                border-radius: 10px; 
                                padding: 20px; 
                                margin: auto; 
                                max-width: 90%;  
                                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                            ">
                                <h3 style="text-align: left; color: #800000;">Advices from your AI-Powered Advisor:</h3>
                                <p style="text-align: left;">{recommendation}</p>
                            </div>
                            """.format(recommendation=recommendation),
                            unsafe_allow_html=True
                        )
                        
                        st.write("Three Latest Related Tweets:")
                        display_latest_3_tweets(user_input)
                        
                        
# Chat History Page
elif selection == "Chat History":
    st.title("Chat History")
    if st.session_state.chat_history:
        with st.container():
            st.markdown("<div class='button-container'>", unsafe_allow_html=True)  # Start button container
            for index, chat in enumerate(st.session_state.chat_history):
                # Create a container for each chat entry
                chat_container = st.container()
                with chat_container:
                    st.write(f"Chat Room: {chat['name']}")

                    # Button to retrieve the recommendation
                    if st.button(f"🔄 Retrieve {chat['name']}", key=f"retrieve_{index}"):
                        # Display the styled recommendation box
                        st.markdown(
                            """
                            <div style="
                                background-color: white; 
                                border-radius: 10px; 
                                padding: 20px; 
                                margin: auto; 
                                max-width: 90%;  
                                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                            ">
                                <h3 style="text-align: left; color: #800000;">Advices from your AI-Powered Advisor:</h3>
                                <p style="text-align: left;">{recommendation}</p>
                            </div>
                            """.format(recommendation=chat['recommendation']),
                            unsafe_allow_html=True
                        )

                    # Separate Remove button for each chat entry
                    if st.button(f"🗑️ Remove {chat['name']}", key=f"remove_{index}"):  # Unique key for each button
                        st.session_state.chat_history.pop(index)  # Remove the chat entry
                        # No need for st.experimental_rerun() here
            st.markdown("</div>", unsafe_allow_html=True)  # End button container
    else:
        st.write("No chat history available. Talk to HSBC-hat now!")
        
# FAQ Page
elif selection == "FAQ":
    st.title("Frequently Asked Questions")

    # Load FAQ content
    faq_content = load_faq()

    # Update the FAQ container style
    st.markdown(
        f"""
        <div style="max-width: 800px; margin: auto; padding: 20px; background-color: rgba(255, 255, 255, 0.9); border-radius: 10px; overflow-y: auto;">
            {faq_content}
        """,
        unsafe_allow_html=True
    )

    # Discovery Page
elif selection == "Discovery":
    st.title("Guess what's hot in the World?")
    display_latest_tweets()


# In[ ]:




