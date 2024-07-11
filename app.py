import streamlit as st
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

st.title("watsonx.ai를 이용한 챗봇!")

# Create LLM
def create_llm(api_key, api_url, project_id, params):
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
    
    # Instantiate a model proxy object to send your requests
    model = Model(
        model_id="meta-llama/llama-2-70b-chat",
        params=params,
        credentials=credentials,
        project_id=project_id)
    
    return model 
    
def watsonx_ai_api(prompt, api_key, api_url, project_id, params):
    model = create_llm(api_key, api_url, project_id, params)
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
        
    st.subheader("LLM Model Parameters")
    model_id = st.selectbox("Model List", options=["meta-llama/llama-3-70b-instruct", "meta-llama/llama-2-13b-chat", "meta-llama/llama-2-70b", "ibm/granite-13b-instruct-v2", "google/flan-ul2", "google/flan-t5-xxl", "codellama/codellama-34b-instruct"], index=0, key="model_id")
    decoding_col1, decoding_col2 = st.columns([2,8])
  
    with decoding_col1:
        st.markdown("Greedy")

    with decoding_col2:
        sampling_on = st.toggle("Sampling")

    print("decoding_method:", sampling_on)    
    
    if sampling_on == True:
        top_k = st.slider("Top k:", min_value=0, max_value=100, value=50)
        top_p = st.number_input("Top P:", value=1,  min_value=0, max_value=1)
        decoding_method = "sample"
    else:
        decoding_method = "greedy" 
        top_p = None
        top_k = None 
     
    max_new_tokens = st.number_input("Max New Token:", min_value=10, max_value=512, value=10)   
    min_new_tokens = st.number_input("Min New Token:", min_value=0, max_value=512, value=1) 
    temperature = st.number_input("Temperature:", value=0.0, step=.1, format="%.2f", max_value=2.1, min_value=0.0)
    repetition_penalty = st.number_input("Repetition Penalty",value=1.0, step=.1, format="%.1f", max_value=2.1, min_value=0.0) 

    params = {
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.TOP_P : top_p,
        GenParams.TOP_K : top_k,
        GenParams.MIN_NEW_TOKENS: min_new_tokens,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.TEMPERATURE : temperature,
        GenParams.REPETITION_PENALTY : repetition_penalty 
    }

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
            response = watsonx_ai_api(prompt, watsonx_api_key, watsonx_api_url, watsonx_project_id, params) 
            st.write(response) 
    st.session_state.messages.append({"role": "assistant", "content": response})