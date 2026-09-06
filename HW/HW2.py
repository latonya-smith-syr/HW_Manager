import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

# Show title and description.
st.title("📄 Document Summarizer")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! ")

secret_key = st.secrets.OPEN_API_KEY

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
#openai_api_key = st.text("Good for you! I will use my own API Key")


    # Create an OpenAI client.
client = OpenAI(api_key=secret_key)


def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.paser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None
    
    # Let the user upload a file via `st.file_uploader`.
attached_url = st.text_input(
    "Please enter a URL", type=("url")
    )

    # Ask the user for a question via `st.text_area`.
#question = st.text_area(
#    "Now ask a question about the document!",
#    placeholder="Can you give me a short summary?",
#    disabled=not uploaded_file,
#)


st.sidebar.title('Choose a Summarization Method')

summary_option = add_selectbox = st.sidebar.selectbox(
    'Options', (
        'Summarize the document in 100 words',
        'Summarize the document in 2 connecting paragraphs',
        'Summarize the document in 5 bullet points'
    )
)

#st.write(summary_option, "Summarization Options")

#mini is advanced and nano is less advanced
advanced_model = st.checkbox('Advanced Model')

if advanced_model:
    model_type = "gpt-5-mini"
else:
    model_type = "gpt-5-nano"

if uploaded_file:

        # Process the uploaded file and question.
    document = uploaded_file.read().decode()
    messages = [
        {
            "role":"system",
            "content": f"Use this instruction of for the document: {summary_option} \n\n---\n\n"

        },
        {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n",
        }
    ]

        # Generate an answer using the OpenAI API.
    stream = client.chat.completions.create(
        model= model_type,
        messages=messages,
        stream=True,
    )

        # Stream the response to the app using `st.write_stream`.
    st.write_stream(stream)