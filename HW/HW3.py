import streamlit as st
from openai import OpenAI
import tiktoken
import requests
import anthropic
from bs4 import BeautifulSoup

st.title("My Chatbot Demo")

st.write("Chatbot Demo")
st.write(" This chatbot functions as a normal chatbot. However there are some limitations.\n" \
"You have the option of inputting up to two URLs (only at the start of the chat) and the chat will forget the conversation after approximately 2000 tokens. \n"
)

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

llm_option = st.sidebar.selectbox(
    'LLMs', (
        'Chat-GPT',
        'Claude'
    )
)

if llm_option == 'Chat-GPT':
    model = "gpt-6-astra"
    api_key = st.secrets["OPENAI_SECRET_KEY"]
    client = OpenAI(api_key=api_key)

else:
    model = "claude-fable-5-1"
    api_key = st.secrets["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic(api_key = api_key)

if 'client' not in st.session_state:
    api_key = api_key
    st.session_state.client= client


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])



if len(st.session_state.messages) <= 1:
    if st.sidebar.checkbox('Add URLs'):
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
    st.session_state["attached_url"] = attached_url
else:
    attached_url = st.session_state.get("attached_url", [])

def read_urls(urls):
    url_text = ""
    if urls == None:
        return url_text
    else:
        for url in urls:
            if not url:
                continue
            content = read_url_content(url)
            if content is None:
                url_text += ""
            else:
                url_text += content[:4000] 
        return url_text


system_prompt = {"role": "system", "content": "Explain all answers simply enough for a 10-year-old to understand.After the user asks you to do something ask them this:Do you want more information?. "
            "If they say yes, give them more information and then ask them specifically: Do you want more information?. If they say no, ask them specifically How can I help you?"+ read_urls(attached_url)}



def count_tokens(text, model=model):

    encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(text))


#Note to Grader:  I used AI to strategize how to calculate the tokens and for the coding logic 
# on looking at the last messages

max_tokens = 2000
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

    api_msg = token_buffer(st.session_state.messages, system_prompt, max_tokens, model)

    if llm_option == 'Chat-GPT':
        try:
            client.models.list()
        except Exception as e:
            st.info(
                "This OpenAI key is no longer valid. "
                "Please contact the owner of this website."
                )
            st.stop()  
        stream = client.chat.completions.create(
            model= model,
            messages = api_msg,
            stream=True
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        try:
            client.messages.create(
            model = model,
            max_tokens=1,
            messages=[{"role": "user", "content": "Are you working?"}]
    )
        except Exception as e:
            st.info(
            "This Claude key is not valid. "
            "Please contact the owner of this website."
        )
            st.stop()
    
        message = client.messages.create(
                model = model,
                max_tokens=1500,
                system= system_prompt["content"],
                #Here I used AI to understand what anthropic is being passed and how to take out the system prompt
                messages=api_msg[1:]
            )
        data = next(block.text for block in message.content if block.type == "text")

        st.session_state.messages.append({"role": "assistant", "content": data})
        st.write(data)