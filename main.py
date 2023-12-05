import streamlit as st
import pandas as pd
import openai
import random
import time
from tempfile import NamedTemporaryFile

# Title of the web application
st.title('Flying Fox Message Generator 💬')
st.write('made with ❤️ by raava')

# Input field for the user to enter their API key
api_key = st.text_input("Enter your OpenAI API key")

# File uploader allows user to add their own Excel file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 20,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper

@retry_with_exponential_backoff
def chat_with_gpt4(prompt, model="gpt-3.5-turbo", max_tokens=300):
    openai.api_key = api_key
    
    response = openai.chat.completions.create(
        model=model,
        messages=[
                {"role": "system", "content": "You are a camp manager responsible for updating parents on their childs experience. To do so, you will use instructions and unique camper and guide infromation provided for each text.."},
                {"role": "user", "content": prompt},
                ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def generate_description(camper_details):
    desc_prompt = f"""
    - Curate a fun and informal text to parents of a camper to inform them of how their day was.  
    - At the end ask 'Please let us know if there's anything else in particular you'd like to be updated on'. 
    - Only use realistic language. 
    - Make less formal. 
    - Add in two emojis.
    - Sign off the message from Bianca every time

    Use the example below:

    Hey "insert parent name",

    Hope you're doing well! (ALWAYS INCLUDE THAT) Just wanted to give you a quick update on "insert camper name" day at camp today. 

    The highlight of "insert camper name" day was definitely the basketball activity. She had a blast playing and really enjoyed it. She also mentioned that she had a great time during the cooking class and made some delicious hummus. 

    In the afternoon, they went to the beach and had a lot of fun playing in the water. It was a great way to cool off and enjoy the summer weather.

    "insert camper name" got along well with "insert friends name" today. They spent a lot of time hanging out together and seemed to have a great time. 

    In terms of challenging moments, everything went smoothly. "insert camper name" didn't eat breakfast, but she ate lunch and dinner just fine. So no worries there.

    If there's anything else in particular you'd like to be updated on, please let us know. 

    Looking forward to another exciting day at camp tomorrow!

    Bianca

    --------

    Please generate a messagefor the following details. :
    {camper_details}
    """

    return chat_with_gpt4(desc_prompt)


def process_dataframe(df):
    descriptions = []

    for index, row in df.iterrows():
        # Skip the row if 'CAMPER' cell is blank or 'GENERATED MESSAGE' cell is not blank
        if pd.isna(row['CAMPER']) or row['CAMPER'] == '' or (not pd.isna(row['GENERATED MESSAGE']) and row['GENERATED MESSAGE'] != ''):
            continue

        camper_details = row['MERGED MESSAGE']
        st.write('MERGED MESSAGE')
        st.write('--------------------------')
        st.write(camper_details)
        st.write('--------------------------')
        description = generate_description(camper_details)
        st.write(description)
        descriptions.append(description)

    return descriptions

if uploaded_file is not None:
    workbook = pd.ExcelFile(uploaded_file)

    with st.form(key='form_select'):
        sheet_name = st.selectbox("Select a sheet", workbook.sheet_names)
        submit_button = st.form_submit_button(label='Generate messages yay')

    if submit_button:
        df = pd.read_excel(workbook, sheet_name=sheet_name, na_filter=False)
        descriptions = process_dataframe(df)

        data = {'Generated Descriptions': descriptions}
        new_df = pd.DataFrame(data)

        with NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            new_df.to_excel(tmp.name, index=False)
            tmp.seek(0)
            data = tmp.read()
            st.sidebar.download_button(
                label="Download Sheet with descriptions",
                data=data,
                file_name='db_with_descriptions.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )


 