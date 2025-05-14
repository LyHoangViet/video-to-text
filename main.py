import os
import boto3
import base64
import json
import streamlit as st
from typing import List
from botocore.exceptions import ClientError
import tempfile
import cv2
from PIL import Image
import io
import enum

# Import từ module video_processor
from video_processor import VideoProcessor

st.set_page_config(
    page_title="Phân Tích Video Sản Phẩm Với Bedrock",
    page_icon="🔍",
    layout="wide"
)

# Khai báo mặc định cho prompt và các thông số model
DEFAULT_PROMPT = """Dựa vào list hình từ video này, bạn hãy:

1. Xác định tất cả các loại sản phẩm khác nhau xuất hiện trong hình (dựa vào màu sắc bao bì, thiết kế và tên sản phẩm).

2. Đếm số lượng sản phẩm của mỗi loại.

3. Tổng hợp thông tin với:
   - Số loại sản phẩm khác nhau
   - Số lượng của mỗi loại
   - Tổng số sản phẩm
   
4. Mô tả ngắn gọn đặc điểm nhận dạng chính của mỗi loại sản phẩm để phân biệt."""

DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.2
DEFAULT_TOP_K = 50
DEFAULT_MAX_TOKENS = 4000
DEFAULT_REGION = "us-east-1"  # Sử dụng region mặc định

# Định nghĩa enum cho các model
class ModelType(str, enum.Enum):
    NOVA = "Amazon Nova"
    CLAUDE = "Claude 3.7 Sonnet"

def detect_image_type(file_name: str) -> str:
    """
    Detects the image MIME type based on file extension.
    
    Args:
        file_name: Name of the image file
        
    Returns:
        MIME type string for the image
    """
    extension = os.path.splitext(file_name)[1].lower()
    
    mime_types = {
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".png": "png",
        ".gif": "gif",
        ".webp": "webp",
        ".bmp": "bmp"
    }
    
    return mime_types.get(extension, "jpeg")

def image_to_base64(image, image_name="frame.jpg"):
    """
    Chuyển đổi ảnh (PIL Image hoặc numpy array) thành chuỗi base64.
    
    Args:
        image: Ảnh dạng PIL Image hoặc numpy array
        image_name: Tên ảnh để xác định kiểu MIME
        
    Returns:
        Tuple (chuỗi base64, kiểu MIME)
    """
    # Nếu là numpy array (từ OpenCV), chuyển thành PIL Image
    if isinstance(image, (list, tuple, bytes, bytearray)):
        # Đã là bytes hoặc bytearray
        img_bytes = image
    elif hasattr(image, 'shape'):  # Numpy array
        img_pil = Image.fromarray(image)
        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG')
        img_bytes = buf.getvalue()
    else:  # PIL Image
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        img_bytes = buf.getvalue()
        
    # Encode base64
    base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
    mime_type = detect_image_type(image_name)
    
    return base64_encoded, mime_type

def analyze_frames_with_nova(frames, prompt: str, temperature: float, top_p: float, top_k: int, max_tokens: int):
    """
    Uses Amazon Nova on AWS Bedrock to analyze video frames and count different types of products.
    
    Args:
        frames: List of video frames to analyze
        prompt: Custom prompt for Nova
        temperature: Temperature setting (0-1)
        top_p: Top-p setting (0-1)
        top_k: Top-k setting
        max_tokens: Maximum tokens in response
        
    Returns:
        Nova's analysis of the video frame content
    """
    
    # Create placeholder for loading indicator
    with st.spinner("Nova đang phân tích frames từ video..."):
        # System prompt
        system_list = [
            {
                "text": "You are an expert at analyzing product images and videos. You can identify different types of products, count them, and describe their key features."
            }
        ]
        
        # Prepare user message with images and text prompt
        user_content = []
        
        # Add images to user content
        for i, (_, frame) in enumerate(frames):
            # Encode to base64
            image_data, mime_type = image_to_base64(frame, f"frame_{i}.jpg")
            
            # Add image to user content
            user_content.append({
                "image": {
                    "format": mime_type,
                    "source": {"bytes": image_data}
                }
            })
        
        # Add text prompt to user content
        user_content.append({"text": prompt})
        
        # Create message list
        message_list = [
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        # Configure inference parameters
        inf_params = {
            "maxTokens": max_tokens,
            "topP": top_p,
            "topK": top_k,
            "temperature": temperature
        }
        
        # Prepare the request payload for Nova on Bedrock
        payload = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            "system": system_list,
            "inferenceConfig": inf_params
        }
        
        try:
            # Create a Bedrock Runtime client
            # Sử dụng region mặc định và credentials từ môi trường
            bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=DEFAULT_REGION
            )
            
            # Invoke the model on Bedrock
            response = bedrock_runtime.invoke_model(
                modelId="us.amazon.nova-premier-v1:0",
                body=json.dumps(payload)
            )
            
            # Parse and return the response
            response_body = json.loads(response["body"].read())
            
            # Extract text content from Nova's response
            content_text = response_body["output"]["message"]["content"][0]["text"]
            return content_text
        
        except ClientError as e:
            return f"Lỗi khi gọi Bedrock model: {str(e)}"
        except Exception as e:
            return f"Lỗi không xác định: {str(e)}"

def analyze_frames_with_claude(frames, prompt: str, temperature: float, top_p: float, top_k: int, max_tokens: int):
    """
    Uses Claude 3.7 Sonnet on AWS Bedrock to analyze video frames and count different types of products.
    
    Args:
        frames: List of video frames to analyze
        prompt: Custom prompt for Claude
        temperature: Temperature setting (0-1)
        top_p: Top-p setting (0-1)
        top_k: Top-k setting
        max_tokens: Maximum tokens in response
        
    Returns:
        Claude's analysis of the video frame content
    """
    
    # Create placeholder for loading indicator
    with st.spinner("Claude đang phân tích frames từ video..."):
        # Read and encode frames
        images = []
        for i, (_, frame) in enumerate(frames):
            # Encode to base64
            image_data, mime_type = image_to_base64(frame, f"frame_{i}.jpg")
                
            images.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": f"image/{mime_type}",
                    "data": image_data
                }
            })
        
        # Prepare the message content combining images and prompt
        message_content = images + [{"type": "text", "text": prompt}]
        
        # Prepare the request payload for Bedrock
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "messages": [
                {
                    "role": "user",
                    "content": message_content
                }
            ]
        }
        
        try:
            # Create a Bedrock Runtime client
            # Use AWS credentials from environment variables or AWS configuration
            bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=DEFAULT_REGION
            )
            
            # Invoke the model on Bedrock
            response = bedrock_runtime.invoke_model(
                modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                body=json.dumps(payload)
            )
            
            # Parse and return the response
            response_body = json.loads(response["body"].read())
            return response_body["content"][0]["text"]
        
        except ClientError as e:
            return f"Lỗi khi gọi Bedrock model: {str(e)}"
        except Exception as e:
            return f"Lỗi không xác định: {str(e)}"

def main():
    # App title
    st.title("Phân Tích Video Sản Phẩm Với Bedrock")
    
    # Information block
    st.info("""
    Ứng dụng này sử dụng Amazon Nova và Claude trên AWS Bedrock để phân tích video sản phẩm.
    Tải lên video và chọn phương pháp trích xuất frames để Nova hoặc Claude phân tích sản phẩm.
    """)
    
    # Sidebar for model parameters only (removed AWS configuration)
    st.sidebar.title("Cấu hình")
    
    # Model selection
    st.sidebar.subheader("Chọn Model")
    selected_model = st.sidebar.radio(
        "Model:",
        options=[ModelType.NOVA, ModelType.CLAUDE],
        format_func=lambda x: x.value
    )
    
    # Model Parameters section
    st.sidebar.subheader("Tham Số Model")
    
    temperature = st.sidebar.slider(
        "Temperature:", 
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.05,
        help="Kiểm soát mức độ ngẫu nhiên trong đầu ra. Giá trị 0 sẽ cho kết quả nhất quán, giá trị cao hơn tạo nhiều biến thể."
    )
    
    top_p = st.sidebar.slider(
        "Top P:",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TOP_P,
        step=0.05,
        help="Kiểm soát đa dạng thông qua nucleus sampling. 1.0 = không hạn chế, 0.5 = chỉ xem xét tokens trong top 50% xác suất."
    )
    
    top_k = st.sidebar.slider(
        "Top K:",
        min_value=0,
        max_value=500,
        value=DEFAULT_TOP_K,
        step=10,
        help="Giới hạn số tokens được xem xét khi tạo đầu ra. 0 = không sử dụng top_k."
    )
    
    max_tokens = st.sidebar.slider(
        "Max Tokens:",
        min_value=100,
        max_value=10000,
        value=DEFAULT_MAX_TOKENS,
        step=100,
        help="Độ dài tối đa của phản hồi được tạo ra."
    )
    
    # Prompt configuration section
    st.sidebar.subheader("Tùy Chỉnh Prompt")
    
    # Option to use custom prompt
    use_custom_prompt = st.sidebar.checkbox("Sử dụng prompt tùy chỉnh", value=False)
    
    if use_custom_prompt:
        prompt = st.sidebar.text_area("Nhập prompt của bạn:", DEFAULT_PROMPT, height=300)
    else:
        prompt = DEFAULT_PROMPT
        st.sidebar.markdown("*Đang sử dụng prompt mặc định*")
    
    # Main content area
    # File uploader for video
    st.subheader("Tải lên video sản phẩm")
    uploaded_video = st.file_uploader(
        "Chọn file video (MP4, MOV, AVI, etc.)",
        type=["mp4", "mov", "avi", "mkv", "wmv"],
    )
    
    # Video frame extraction settings
    st.subheader("Cài đặt trích xuất frames")
    
    extraction_method = st.radio(
        "Phương pháp trích xuất frames:",
        ["Đều đặn theo số lượng", "Theo khoảng thời gian", "Tự động phát hiện keyframes"]
    )
    
    # Show different settings based on extraction method
    if extraction_method == "Đều đặn theo số lượng":
        num_frames = st.slider(
            "Số lượng frames cần trích xuất:",
            min_value=5,
            max_value=100,
            value=10,
            step=1
        )
    elif extraction_method == "Theo khoảng thời gian":
        interval_seconds = st.slider(
            "Khoảng thời gian giữa các frames (giây):",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5
        )
    else:  # Tự động phát hiện keyframes
        threshold = st.slider(
            "Ngưỡng phát hiện keyframes:",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Giá trị càng thấp, càng nhiều frames được phát hiện"
        )
        max_keyframes = st.slider(
            "Số lượng keyframes tối đa:",
            min_value=5,
            max_value=30,
            value=15,
            step=1
        )
    
    # Show video and process it
    if uploaded_video:
        st.video(uploaded_video)
        
        # Button to extract frames and analyze
        if st.button("Trích xuất frames và phân tích", type="primary"):
            # Process the video to extract frames
            video_processor = VideoProcessor(uploaded_video)
            video_info = video_processor.get_video_info()
            
            # Display video info
            st.subheader("Thông tin video")
            st.write(f"Tên file: {video_info['filename']}")
            st.write(f"Độ phân giải: {video_info['resolution'][0]} x {video_info['resolution'][1]}")
            st.write(f"FPS: {video_info['fps']:.2f}")
            st.write(f"Thời lượng: {video_info['duration']:.2f} giây")
            st.write(f"Tổng số frames: {video_info['frame_count']}")
            
            # Extract frames based on selected method
            with st.spinner("Đang trích xuất frames từ video..."):
                if extraction_method == "Đều đặn theo số lượng":
                    frames = video_processor.extract_frames_uniform(num_frames)
                    st.write(f"Đã trích xuất {len(frames)} frames phân bố đều")
                elif extraction_method == "Theo khoảng thời gian":
                    frames = video_processor.extract_frames_interval(interval_seconds)
                    st.write(f"Đã trích xuất {len(frames)} frames (mỗi {interval_seconds} giây)")
                else:  # Tự động phát hiện keyframes
                    frames = video_processor.extract_frames_keyframes(threshold, max_keyframes)
                    st.write(f"Đã phát hiện và trích xuất {len(frames)} keyframes")
            
            # Display extracted frames with expander for show/hide
            st.subheader("Frames đã trích xuất")
            
            # Lưu frames để phân tích (luôn thực hiện)
            extracted_frames = [(i, frame.image) for i, frame in enumerate(frames)]
            
            # Hiển thị thông tin tổng quan về frames
            st.info(f"Đã trích xuất {len(frames)} frames.")
            
            # Sử dụng expander để hiển thị/ẩn frames (không gây reload toàn bộ trang)
            with st.expander("Xem chi tiết các frames", expanded=False):
                # Thêm tùy chọn xem kiểu lưới hoặc danh sách
                view_mode = st.radio(
                    "Chế độ hiển thị:",
                    ["Lưới", "Danh sách"]
                )
                
                if view_mode == "Lưới":
                    # Create columns to display frames in grid
                    num_cols = 4  # Number of columns in the grid
                    cols = st.columns(num_cols)
                    
                    # Hiển thị các frames đã trích xuất theo lưới
                    for i, frame in enumerate(frames):
                        col_idx = i % num_cols
                        with cols[col_idx]:
                            # Hiển thị frame
                            st.image(
                                frame.image, 
                                caption=f"Frame {frame.frame_number} (t={frame.timestamp:.2f}s)", 
                                use_column_width=True
                            )
                else:  # Danh sách
                    # Tạo container để đặt tất cả các expander bên trong
                    frame_container = st.container()
                    # Hiển thị theo danh sách từng frame một
                    for i, frame in enumerate(frames):
                        with frame_container.expander(f"Frame {frame.frame_number} (t={frame.timestamp:.2f}s)"):
                            st.image(frame.image, use_column_width=True)
            
            # Phân tích frames bằng Nova hoặc Claude
            if extracted_frames:
                if selected_model == ModelType.NOVA:
                    # Call Nova via AWS Bedrock
                    result = analyze_frames_with_nova(
                        extracted_frames, 
                        prompt, 
                        temperature, 
                        top_p, 
                        top_k, 
                        max_tokens
                    )
                else:  # Claude
                    # Call Claude via AWS Bedrock
                    result = analyze_frames_with_claude(
                        extracted_frames, 
                        prompt, 
                        temperature, 
                        top_p, 
                        top_k, 
                        max_tokens
                    )
                
                # Display results
                st.subheader(f"Kết quả phân tích từ {selected_model.value}")
                st.markdown(result)
                
                # Option to download as text file
                if result:
                    # Create download button for the result
                    st.download_button(
                        label="Tải kết quả về (TXT)",
                        data=result,
                        file_name="ket_qua_phan_tich.txt",
                        mime="text/plain"
                    )
                    
                    # Save prompt used for reference
                    st.download_button(
                        label="Tải prompt đã sử dụng (TXT)",
                        data=prompt,
                        file_name="prompt_da_su_dung.txt",
                        mime="text/plain"
                    )
    else:
        st.write("👆 Hãy tải lên video để phân tích")

if __name__ == "__main__":
    main()