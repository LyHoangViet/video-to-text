import os
import cv2
import random
import streamlit as st
from typing import List, Dict, Union
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from models import ChatModel
from config import config
from role_prompt import role_prompt

# Load environment variables
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "widget_key" not in st.session_state:
    st.session_state["widget_key"] = str(random.randint(1, 1000000))

def set_page_config():
    """Set up the Streamlit page configuration"""
    st.set_page_config(
        page_title="Video Analysis with Bedrock",
        layout="wide",
        page_icon="🎥"
    )
    st.title("🎥 Video Analysis with Bedrock AI")

def extract_frames(video_path: str, output_folder: str, percentage: float) -> Dict:
    """
    Extract frames from video and return processing information
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return {"error": f"Cannot open video file: {video_path}"}
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames_to_extract = int(total_frames * percentage / 100)
    
    if num_frames_to_extract <= 0:
        return {"error": "Percentage too small, no frames extracted"}
    
    step = total_frames / num_frames_to_extract
    count = 0
    frame_index = 0
    
    while count < num_frames_to_extract:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = video.read()
        
        if not success:
            break
        
        output_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(output_path, frame)
        
        count += 1
        frame_index += step
    
    video.release()
    
    return {
        "total_frames": total_frames,
        "fps": fps,
        "extracted_frames": count,
        "output_folder": output_folder
    }

def init_chat_model(model_name: str, model_kwargs: Dict) -> RunnableWithMessageHistory:
    """Initialize the chat model with Bedrock"""
    chat_model = ChatModel(model_name, model_kwargs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that analyzes video frames and provides insights."),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="query"),
    ])

    chain = prompt | chat_model.llm
    msgs = StreamlitChatMessageHistory()
    
    conversation = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="query",
        history_messages_key="chat_history"
    ) | StrOutputParser()
    
    return conversation

def render_sidebar() -> tuple:
    """Render sidebar with model selection and parameters"""
    with st.sidebar:
        st.header("Model Settings")
        
        model_name = st.selectbox(
            'Select Model',
            ['us.anthropic.claude-3-7-sonnet-20250219-v1:0', 
             'us.amazon.nova-pro-v1:0',
             'us.amazon.nova-premier-v1:0'],
            key=f"{st.session_state['widget_key']}_model"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            key=f"{st.session_state['widget_key']}_temperature"
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4096,
            value=2048,
            key=f"{st.session_state['widget_key']}_max_tokens"
        )
        
        model_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        return model_name, model_kwargs

def main():
    set_page_config()
    
    # Sidebar configuration
    model_name, model_kwargs = render_sidebar()
    
    # Main content area
    st.header("Video Processing")
    
    # File uploader for video
    uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
    
    # Frame extraction parameters
    col1, col2 = st.columns(2)
    with col1:
        percentage = st.slider(
            "Frame extraction percentage",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=1.0
        )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_video_path = f"temp_{uploaded_file.name}"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Create output folder for frames
        output_folder = "extracted_frames"
        
        # Process video
        if st.button("Process Video"):
            with st.spinner("Processing video..."):
                result = extract_frames(temp_video_path, output_folder, percentage)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"""
                    Video Processing Complete:
                    - Total frames: {result['total_frames']}
                    - FPS: {result['fps']}
                    - Extracted frames: {result['extracted_frames']}
                    """)
                    
                    # Display first few frames
                    st.subheader("Sample Frames")
                    cols = st.columns(4)
                    for i, frame_file in enumerate(sorted(os.listdir(output_folder))[:4]):
                        if frame_file.endswith('.jpg'):
                            with cols[i % 4]:
                                st.image(os.path.join(output_folder, frame_file), 
                                       caption=f"Frame {i+1}")
                    
                    # Initialize chat model for analysis
                    chat_model = init_chat_model(model_name, model_kwargs)
                    
                    # Analysis prompt
                    analysis_prompt = f"""
                    Analyze the video frames extracted from {uploaded_file.name}:
                    - Total frames: {result['total_frames']}
                    - FPS: {result['fps']}
                    - Extracted frames: {result['extracted_frames']}
                    
                    Please provide insights about:
                    1. The video's characteristics
                    2. Potential content analysis
                    3. Any notable patterns or features
                    """
                    
                    # Generate analysis
                    with st.spinner("Generating analysis..."):
                        response = chat_model.invoke(
                            {"query": [{"role": "user", "content": analysis_prompt}]},
                            config={"configurable": {"session_id": "streamlit_chat"}}
                        )
                        
                        st.subheader("AI Analysis")
                        st.write(response)
        
        # Cleanup
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

if __name__ == "__main__":
    main()
