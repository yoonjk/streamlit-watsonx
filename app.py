import streamlit as st
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# Create LLM
def create_llm(api_key, api_url, project_id):
    """_summary_
        LLM을 생성합니다.
    Args:
        api_key (_type_): ibm cloud api key
        api_url (_type_): ibm cloud region url
        project_id (_type_): watsonx.ai project_id
    """
    credentials = {
        "apikey": api_key,
        "url" : api_url
    }
    
   # Instantiate parameters for text generation
    params = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 300
    }

    # Instantiate a model proxy object to send your requests
    model = Model(
        model_id="meta-llama/llama-3-70b-instruct",
        params=params,
        credentials=credentials,
        project_id=project_id)
    
    return model 
    
def watsonx_ai_api(prompt, api_key, api_url, project_id):
    model = create_llm(api_key, api_url, project_id)
    response = model.generate_text(prompt = prompt)
    print(response) 
    
    return response

with st.sidebar:
    watsonx_api_key = st.text_input('Enter API Key:')
    watsonx_api_url = st.text_input('Enter API Url:', value="https://us-south.ml.cloud.ibm.com")
    watsonx_project_id = st.text_input('Enter PROJECT_ID:')
  
    if not (watsonx_api_key and watsonx_api_url and watsonx_project_id):
        st.warning('Please enter your credentials!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = watsonx_ai_api(prompt, watsonx_api_key, watsonx_api_url, watsonx_project_id) 
            st.write(response) 
    st.session_state.messages.append({"role": "assistant", "content": response})