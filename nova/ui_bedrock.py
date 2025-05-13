import streamlit as st
import boto3
import json

# Initialize Bedrock client
client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
)

# Available models
AVAILABLE_MODELS = {
    "Nova Premier": "us.amazon.nova-premier-v1:0",
    "Nova Pro": "us.amazon.nova-pro-v1:0"

}

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def create_request(input_type, s3_uri, bucket_owner, system_prompt, user_prompt, model_params):
    message_list = [
        {
            "role": "user",
            "content": [
                {
                    input_type: {
                        "format": "mp4" if input_type == "video" else "jpg",
                        "source": {
                            "s3Location": {
                                "uri": s3_uri,
                                "bucketOwner": bucket_owner
                            }
                        }
                    }
                },
                {
                    "text": user_prompt
                }
            ]
        }
    ]
    
    system_list = [{"text": system_prompt}]
    
    native_request = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": model_params,
    }
    
    return native_request

def main():
    st.title("Bedrock Model Interface")
    
    # Sidebar for model selection and parameters
    with st.sidebar:
        st.header("Model Configuration")
        selected_model_name = st.selectbox(
            "Select Model",
            options=list(AVAILABLE_MODELS.keys()),
            index=0
        )
        model_id = AVAILABLE_MODELS[selected_model_name]

        system_prompt = st.text_area("System Prompt", value="Bạn là một chuyên gia phân tích", height=100)
        
        st.subheader("Model Parameters")
        max_tokens = st.slider("Max Tokens", 1, 10000, 1000)
        top_p = st.slider("Top P", 0.0, 1.0, 0.3)
        top_k = st.slider("Top K", 1, 100, 50)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
        
        
        model_params = {
            "maxTokens": max_tokens,
            "topP": top_p,
            "topK": top_k,
            "temperature": temperature
        }
    
    # Input type selection
    input_type = st.radio("Select Input Type", ["image", "video"])
    
    # S3 configuration
    s3_uri = st.text_input("S3 URI", value="s3://test-models-vnm/video/video-test2.mp4")
    bucket_owner = st.text_input("Bucket Owner", value="536697245***")
    
    # User prompt
    user_prompt = st.text_area("User Prompt", value="Đây là video tôi đã cung cấp, hãy phân tích nó.")
    
    if st.button("Generate Response"):
        try:
            # Create request
            request = create_request(
                input_type,
                s3_uri,
                bucket_owner,
                system_prompt,
                user_prompt,
                model_params
            )
            
            # Invoke model
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(request)
            )
            
            # Parse response
            model_response = json.loads(response["body"].read())
            content_text = model_response["output"]["message"]["content"][0]["text"]
            
            # Display response
            st.subheader("Model Response")
            st.write(content_text)
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": content_text
            })
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # # Display chat history
    # if st.session_state.messages:
    #     st.header("Chat History")
    #     for message in st.session_state.messages:
    #         with st.chat_message(message["role"]):
    #             st.write(message["content"])

if __name__ == "__main__":
    main()

