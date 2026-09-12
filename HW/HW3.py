import streamlit as st
from openai import OpenAI
import tiktoken
import requests
from bs4 import BeautifulSoup

st.title("My Lab 3 question answering chatbot")

st.write("Chatbot Demo")

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

#model = "gpt-4o-mini"


if 'client' not in st.session_state:
    api_key = st.secrets["OPENAI_SECRET_KEY"]
    st.session_state.client= OpenAI(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

llm_option = st.sidebar.selectbox(
    'LLMs', (
        'Chat-GPT',
        'Claude'
    )
)

if llm_option == 'Chat-GPT':
    model = "gpt-6-astra"
else:
    model = "claude-fable-5-1"


if st.checkbox('Add URLs'):
    url_number = st.selectbox('How many URLS?',
                          ('1', '2')) 
    attached_url = []
    if url_number == '1':
        attached_url.append(st.text_input('Please enter a URL', type=("url")))
    elif url_number == '2':
        url1 = st.text_input('URL 1', type =("url"))
        url2 = st.text_input('URL 2', type=("url"))
        attached_url.append(url1)
        attached_url.append(url2)
else:
    attached_url= []

def read_urls(urls):
    url_text = ""
    if urls == None:
        return url_text
    else:
        for url in urls:
            if read_url_content(url) == None:
                url_text += ""
            else:
                url_text += read_url_content(url)
        return url_text



system_prompt = {"role": "system", "content": "Explain all answers simply enough for a 10-year-old to understand.After the user's first response ask them this:Do you want more information?. "
            "If they say yes, give them more information and then ask them specifically: Do you want more information?. If they say no, ask them specifically How can I help you?"+ read_urls(attached_url)}

max_tokens = 2000

buffer_type = st.sidebar.selectbox('Buffer type', ('Last 2 responses', 'Token-based'))

def count_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def msg_buffer(messages, system_prompt):
    return [system_prompt] + messages[-4:]

#Note to Grader:  I used AI to strategize how to calculate the tokens and for the coding logic 
# on looking at the last messages
def token_buffer(messages, system_prompt, max_tokens, model=model):
    system_tokens = count_tokens(system_prompt["content"], model)
    budget = max_tokens - system_tokens
    kept = []
    total = 0
    for msg in reversed(messages):
        t = count_tokens(msg["content"], model)
        if total + t > budget:
            break
        kept.insert(0, msg)
        total += t
    return [system_prompt] + kept   




if prompt := st.chat_input("What is up?"):   

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    if buffer_type == "Last 2 responses":
        api_msg = msg_buffer(st.session_state.messages, system_prompt)
    else:
        api_msg = token_buffer(st.session_state.messages, system_prompt, max_tokens, model)

    client = st.session_state.client
    stream = client.chat.completions.create(
        model= model,
        messages = api_msg,
        stream=True
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

    