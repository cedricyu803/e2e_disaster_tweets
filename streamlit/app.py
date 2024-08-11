import yaml
from backend_client import BackendClient

import streamlit as st

# get endpoint_base
APP_CONFIG_PATH = './app_config.yml'
with open(APP_CONFIG_PATH, 'r') as f:
    app_configs = yaml.load(f, yaml.SafeLoader)
global endpoint_base
endpoint_base = app_configs['endpoint_base']

st.set_page_config(
    page_title="Disaster tweet classification",
    layout="centered", initial_sidebar_state="auto", menu_items=None)


st.title("Disaster tweet classification")
st.info("Disaster tweet classification")


st.session_state.messages = []
# initialise backend
backend_client = BackendClient(endpoint_base=endpoint_base)

if st.session_state.messages is None:
    st.session_state.messages = []

if ("messages" not in st.session_state.keys() or
        st.session_state.messages == []):
    st.session_state.messages = \
        [{"role": "assistant",
            "content": "Please enter your tweet to query"}]

# Prompt for user input and display on streamlit
if prompt := st.chat_input("Please enter your tweet to query"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Formulating response"):
            response = backend_client.get_query(prompt)
            answer = ''
            if response == 1:
                answer = "It is a disaster!"
            elif response == 0:
                answer = 'Not a disaster'
            else:
                answer = 'Answer unknown'
            # new chat messages are automatically appended to database
            st.write(answer)
            message = {"role": "assistant", "content": answer}
            # Add response to message history
            st.session_state.messages.append(message)
