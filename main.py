import os
import boto3
import base64
import json
import streamlit as st
from botocore.exceptions import ClientError
from PIL import Image
import io
import enum
import pandas as pd

# Import từ module video_processor
from video_processor import VideoProcessor

st.set_page_config(
    page_title="Phân Tích Video/Ảnh Sản Phẩm Với Bedrock",
    page_icon="🔍",
    layout="wide"
)

# Khai báo mặc định cho prompt và các thông số model
DEFAULT_PROMPT = """# 🧠 Hướng Dẫn Phân Tích Sản Phẩm Vinamilk Từ Các Frame Ảnh Cắt Từ Video Quay Kệ Hàng

## 🎯 Mục Tiêu

Bạn là một chuyên gia phân tích hình ảnh từ video. Nhiệm vụ của bạn là nhận diện, phân loại và trích xuất thông tin chi tiết về các sản phẩm Vinamilk xuất hiện trong các hình ảnh (frame) được cắt từ một video quay kệ trưng bày sản phẩm (chai, hộp, khay...).

❗️Lưu ý đặc biệt:  
Do dữ liệu đầu vào là các frame ảnh liên tiếp từ video, có khả năng cao các sản phẩm sẽ xuất hiện lặp lại ở nhiều frame. Vì vậy bạn cần áp dụng kỹ thuật nhận diện trùng lặp (duplicate detection) để:

- ✅ Không đếm trùng sản phẩm đã xuất hiện ở các frame trước.
- ✅ Chỉ đếm mỗi sản phẩm Vinamilk một lần nếu mặt trước đã được nhìn rõ từ trước đó.

---

## 🔧 Tiền Xử Lý Ảnh (BẮT BUỘC THỰC HIỆN TRƯỚC)

Trước khi bắt đầu phân tích, phải thực hiện các bước xử lý ảnh sau:

1. 🌀 Xoay ảnh đúng chiều: mặt trước sản phẩm hướng ra ngoài.
2. 🌗 Cân bằng độ sáng và tương phản để cải thiện khả năng đọc nhãn.
3. 🔍 Phóng to vùng chứa chữ trên sản phẩm để dễ phân tích.
4. ⚙️ Tăng độ sắc nét nếu ảnh bị mờ hoặc out nét.

---

## 📌 Nhiệm Vụ Phân Tích Nội Dung

1. ✅ Đếm tổng số sản phẩm Vinamilk có mặt trước rõ ràng.  
   - ❌ Không đếm sản phẩm chỉ thấy mặt bên hoặc mặt sau.  
   - ❗️Không đếm trùng sản phẩm đã thấy ở các frame trước.

2. 🏷️ Phân loại dòng sản phẩm Vinamilk  
   - Ví dụ: sữa tươi, sữa chua, sữa bột, sữa đặc...  
   - ⚠️ Nếu cùng loại nhưng khác màu bao bì → tách thành dòng riêng biệt  

3. 📦 Đếm số stack (hàng ngang dưới cùng của mỗi dòng sản phẩm)  
   - Không tính sản phẩm ở phía sau hoặc bị khuất.

4. 💰 Trích xuất giá gốc  
   - Giá thường nhỏ hơn, bị gạch ngang.

5. 💵 Trích xuất giá khuyến mãi  
   - Giá thường to hơn, in đậm hơn giá gốc.

6. 🗓️ Trích xuất thời gian khuyến mãi  
   - Dạng: `DD/MM–DD/MM/YYYY`, nằm gần khu vực giá.

7. 🎨 Mô tả màu sắc bao bì  
   - Bao gồm: Màu chính (primary) và màu phụ (secondary).

---

## 📤 Kết Quả Trả Về (CHỈ DƯỚI DẠNG JSON)

> Mỗi sản phẩm là một phần tử trong mảng JSON.

Cấu trúc mẫu:

```json
[
  {
    "label_name": "Vinamilk Flex đỏ trắng",
    "total_stacks": 4,
    "product_detail": "1L, Vinamilk, sữa tươi tiệt trùng, ít béo",
    "origin_price": "34.000đ",
    "discount_price": "29.500đ",
    "promotion_period": "10/05–15/05/2025"
  },
  {
    "label_name": "Vinamilk ADM xanh trắng",
    "total_stacks": 3,
    "product_detail": "110ml, Vinamilk, sữa chua uống, bổ sung vitamin A+D+M",
    "origin_price": "18.000đ",
    "discount_price": "15.000đ",
    "promotion_period": "01/05–10/05/2025"
  }
]
```

## ⚠️ Lưu Ý Bắt Buộc

- ❗️ Bạn cần xử lý và loại bỏ các sản phẩm bị trùng lặp giữa các frame do video cắt thành nhiều ảnh liên tiếp.
  - Sử dụng cơ chế nhận diện hình ảnh hoặc vị trí (hoặc hashing ảnh) để phát hiện trùng lặp.
  - Ưu tiên đếm sản phẩm trong frame đầu tiên mà sản phẩm đó xuất hiện rõ ràng.
- ✅ Chỉ đếm các sản phẩm Vinamilk có mặt trước được nhìn thấy rõ ràng.
- ✅ Cùng một sản phẩm nhưng khác màu bao bì → tính là dòng sản phẩm riêng biệt.
- ✅ Nếu một sản phẩm xuất hiện ở nhiều frame → chỉ đếm 1 lần duy nhất.
- ❌ Không đếm sản phẩm bị khuất, mờ, hoặc chỉ thấy một phần bên hông hay mặt sau.
- 🚫 Không thêm bất kỳ nội dung giải thích nào ngoài JSON đầu ra.
- ⚙️ JSON trả về phải hợp lệ hoàn toàn: đúng cấu trúc, đúng định dạng kiểu dữ liệu, không thiếu trường.
"""

DEFAULT_TEMPERATURE = 0.01
DEFAULT_TOP_P = 0.2
DEFAULT_TOP_K = 100
DEFAULT_MAX_TOKENS = 10000
DEFAULT_REGION = "us-east-1" 

class ModelType(str, enum.Enum):
    NOVA = "Nova Premier"
    CLAUDE_3_7_SONNET = "Claude 3.7 Sonnet"
    CLAUDE_4_SONNET = "Claude 4 Sonnet"
    CLAUDE_4_OPUS = "Claude 4 Opus"

class InputType(str, enum.Enum):
    VIDEO = "Video"
    IMAGE = "Hình ảnh"

def get_model_id(model_type: ModelType) -> str:
    """
    Trả về model ID tương ứng với loại model được chọn
    
    Args:
        model_type: Loại model được chọn
        
    Returns:
        Model ID string cho AWS Bedrock
    """
    model_ids = {
        ModelType.NOVA: "us.amazon.nova-premier-v1:0",
        ModelType.CLAUDE_3_7_SONNET: "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        ModelType.CLAUDE_4_SONNET: "us.anthropic.claude-sonnet-4-20250514-v1:0",
        ModelType.CLAUDE_4_OPUS: "us.anthropic.claude-opus-4-20250514-v1:0"
    }
    return model_ids[model_type]

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

def image_to_base64(image, image_name="image.jpg"):
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
    with st.spinner("Nova đang phân tích hình ảnh..."):
        # System prompt
        system_list = [
            {
                "text": "You are an expert at analyzing product images and videos. You can identify different types of products, count them, and describe their key features."
            }
        ]
        
        # Prepare user message with images and text prompt
        user_content = []
        
        # Add images to user content
        for i, frame in enumerate(frames):
            # Encode to base64
            if isinstance(frame, tuple):
                # Từ video frames (tuple của index và frame)
                _, frame_img = frame
            else:
                # Trực tiếp từ hình ảnh
                frame_img = frame
                
            # Encode to base64
            image_data, mime_type = image_to_base64(frame_img, f"image_{i}.jpg")
            
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
                modelId=get_model_id(ModelType.NOVA),
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

def analyze_frames_with_claude(frames, prompt: str, temperature: float, top_p: float, top_k: int, max_tokens: int, model_type: ModelType):
    """
    Uses Claude models on AWS Bedrock to analyze video frames and count different types of products.
    
    Args:
        frames: List of video frames or images to analyze
        prompt: Custom prompt for Claude
        temperature: Temperature setting (0-1)
        top_p: Top-p setting (0-1)
        top_k: Top-k setting
        max_tokens: Maximum tokens in response
        model_type: Type of Claude model to use
        
    Returns:
        Claude's analysis of the video frame content
    """
    
    model_name = model_type.value
    with st.spinner(f"{model_name} đang phân tích hình ảnh..."):
        images = []
        for i, frame in enumerate(frames):
            if isinstance(frame, tuple):
                _, frame_img = frame
            else:
                frame_img = frame
            
            image_data, mime_type = image_to_base64(frame_img, f"image_{i}.jpg")
                
            images.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": f"image/{mime_type}",
                    "data": image_data
                }
            })
        
        message_content = images + [{"type": "text", "text": prompt}]
        
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
            bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=DEFAULT_REGION
            )
            
            response = bedrock_runtime.invoke_model(
                modelId=get_model_id(model_type),
                body=json.dumps(payload)
            )
            
            response_body = json.loads(response["body"].read())
            return response_body["content"][0]["text"]
        
        except ClientError as e:
            return f"Lỗi khi gọi Bedrock model: {str(e)}"
        except Exception as e:
            return f"Lỗi không xác định: {str(e)}"
        
def display_results_as_table(result_text):
    """
    Chuyển đổi kết quả JSON thành bảng Pandas và hiển thị trong Streamlit
    
    Args:
        result_text: Văn bản kết quả từ model (dạng JSON)
    
    Returns:
        DataFrame: Pandas DataFrame được tạo từ JSON nếu thành công, None nếu thất bại
    """
    try:
        json_start = result_text.find('[')
        json_end = result_text.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = result_text[json_start:json_end]
            data = json.loads(json_str)
            
            df = pd.DataFrame(data)
            
            return df
        else:
            st.error("Không tìm thấy JSON hợp lệ trong kết quả")
            return None
    except json.JSONDecodeError:
        st.error("Không thể parse JSON từ kết quả. Định dạng không hợp lệ.")
        return None
    except Exception as e:
        st.error(f"Lỗi khi chuyển đổi kết quả thành bảng: {str(e)}")
        return None

def main():
    st.title("Phân Tích Video/Ảnh Sản Phẩm Với Bedrock")
    
    st.info("""
    Ứng dụng này sử dụng Amazon Nova và Claude trên AWS Bedrock để phân tích video hoặc hình ảnh sản phẩm.
    Tải lên video/hình ảnh và chọn phương pháp phân tích để nhận kết quả.
    """)
    
    st.sidebar.title("Cấu hình")
    
    # Model selection
    st.sidebar.subheader("Chọn Model")
    selected_model = st.sidebar.radio(
        "Model:",
        options=[ModelType.NOVA, ModelType.CLAUDE_3_7_SONNET, ModelType.CLAUDE_4_SONNET, ModelType.CLAUDE_4_OPUS],
        format_func=lambda x: x.value
    )
    
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
    
    st.sidebar.subheader("Tùy Chỉnh Prompt")
    
    use_custom_prompt = st.sidebar.checkbox("Sử dụng prompt tùy chỉnh", value=False)
    
    if use_custom_prompt:
        prompt = st.sidebar.text_area("Nhập prompt của bạn:", DEFAULT_PROMPT, height=300)
    else:
        prompt = DEFAULT_PROMPT
        st.sidebar.markdown("*Đang sử dụng prompt mặc định*")
    
    input_type = st.radio(
        "Chọn loại đầu vào:",
        [InputType.VIDEO, InputType.IMAGE]
    )
    
    if input_type == InputType.VIDEO:
        st.subheader("Tải lên video sản phẩm")
        uploaded_video = st.file_uploader(
            "Chọn file video (MP4, MOV, AVI, etc.)",
            type=["mp4", "mov", "avi", "mkv", "wmv"],
            key="video_uploader"
        )
        
        st.subheader("Cài đặt trích xuất frames")
        
        extraction_method = st.radio(
            "Phương pháp trích xuất frames:",
            ["Đều đặn theo số lượng", "Theo khoảng thời gian", "Tự động phát hiện keyframes"]
        )
        
        if extraction_method == "Đều đặn theo số lượng":
            num_frames = st.slider(
                "Số lượng frames cần trích xuất:",
                min_value=5,
                max_value=100,
                value=20,
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
        else:  
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
        
        if uploaded_video:
            st.video(uploaded_video)
            
            if st.button("Trích xuất frames và phân tích", type="primary", key="analyze_video_btn"):
                video_processor = VideoProcessor(uploaded_video)
                video_info = video_processor.get_video_info()
                
                st.subheader("Thông tin video")
                st.write(f"Tên file: {video_info['filename']}")
                st.write(f"Độ phân giải: {video_info['resolution'][0]} x {video_info['resolution'][1]}")
                st.write(f"FPS: {video_info['fps']:.2f}")
                st.write(f"Thời lượng: {video_info['duration']:.2f} giây")
                st.write(f"Tổng số frames: {video_info['frame_count']}")
                
                with st.spinner("Đang trích xuất frames từ video..."):
                    if extraction_method == "Đều đặn theo số lượng":
                        frames = video_processor.extract_frames_uniform(num_frames)
                        st.write(f"Đã trích xuất {len(frames)} frames phân bố đều")
                    elif extraction_method == "Theo khoảng thời gian":
                        frames = video_processor.extract_frames_interval(interval_seconds)
                        st.write(f"Đã trích xuất {len(frames)} frames (mỗi {interval_seconds} giây)")
                    else:  
                        frames = video_processor.extract_frames_keyframes(threshold, max_keyframes)
                        st.write(f"Đã phát hiện và trích xuất {len(frames)} keyframes")
                
                st.subheader("Frames đã trích xuất")
                
                extracted_frames = [(i, frame.image) for i, frame in enumerate(frames)]
                
                st.info(f"Đã trích xuất {len(frames)} frames.")
                
                with st.expander("Xem chi tiết các frames", expanded=False):
                    view_mode = st.radio(
                        "Chế độ hiển thị:",
                        ["Lưới", "Danh sách"],
                        key="video_view_mode"
                    )
                    
                    if view_mode == "Lưới":
                        num_cols = 4  
                        cols = st.columns(num_cols)
                        
                        for i, frame in enumerate(frames):
                            col_idx = i % num_cols
                            with cols[col_idx]:
                                st.image(
                                    frame.image, 
                                    caption=f"Frame {frame.frame_number} (t={frame.timestamp:.2f}s)", 
                                    use_column_width=True
                                )
                    else: 
                        frame_container = st.container()
                        for i, frame in enumerate(frames):
                            with frame_container.expander(f"Frame {frame.frame_number} (t={frame.timestamp:.2f}s)"):
                                st.image(frame.image, use_column_width=True)
                
                if extracted_frames:
                    if selected_model == ModelType.NOVA:
                        result = analyze_frames_with_nova(
                            extracted_frames, 
                            prompt, 
                            temperature, 
                            top_p, 
                            top_k, 
                            max_tokens
                        )
                    else: 
                        result = analyze_frames_with_claude(
                            extracted_frames, 
                            prompt, 
                            temperature, 
                            top_p, 
                            top_k, 
                            max_tokens,
                            selected_model
                        )
                    
                    st.subheader(f"Kết quả phân tích từ {selected_model.value}")
                    with st.expander("Xem kết quả phân tích", expanded=False):
                        st.markdown(result)  
                                            
                    df = display_results_as_table(result)
                    if df is not None:
                        st.subheader("Kết quả dạng bảng")
                        
                        edited_df = st.data_editor(
                            df,
                            num_rows="dynamic",
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        csv = edited_df.to_csv(index=False)
                        st.download_button(
                            label="Tải bảng về (CSV)",
                            data=csv,
                            file_name="ket_qua_phan_tich.csv",
                            mime="text/csv"
                        )
                        
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            edited_df.to_excel(writer, sheet_name='Kết quả phân tích', index=False)
                        buffer.seek(0)
                        
                        st.download_button(
                            label="Tải bảng về (Excel)",
                            data=buffer,
                            file_name="ket_qua_phan_tich.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    st.download_button(
                        label="Tải kết quả về (TXT)",
                        data=result,
                        file_name="ket_qua_phan_tich.txt",
                        mime="text/plain"
                    )
                    
                    st.download_button(
                        label="Tải prompt đã sử dụng (TXT)",
                        data=prompt,
                        file_name="prompt_da_su_dung.txt",
                        mime="text/plain"
                    )
        else:
            st.write("👆 Hãy tải lên video để phân tích")
            
    else:
        st.subheader("Tải lên hình ảnh sản phẩm")
        uploaded_images = st.file_uploader(
            "Chọn file hình ảnh (JPG, PNG, etc.)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
            key="image_uploader"
        )
        
        if uploaded_images:
            st.info(f"Đã tải lên {len(uploaded_images)} hình ảnh.")
            
            st.subheader("Hình ảnh đã tải lên")
            
            view_mode = st.radio(
                "Chế độ hiển thị:",
                ["Lưới", "Danh sách"],
                key="image_view_mode"
            )
            
            processed_images = []
            
            if view_mode == "Lưới":
                num_cols = 3 
                cols = st.columns(num_cols)
                
                for i, img_file in enumerate(uploaded_images):
                    col_idx = i % num_cols
                    with cols[col_idx]:
                        img = Image.open(img_file)
                        # Đọc hình ảnh
                        img = Image.open(img_file)
                        # Hiển thị hình ảnh
                        st.image(img, caption=f"Ảnh {i+1}: {img_file.name}", use_column_width=True)
                        # Thêm hình ảnh vào danh sách để phân tích
                        processed_images.append(img)
            else:  # Danh sách
                # Tạo container để đặt tất cả các expander bên trong
                image_container = st.container()
                # Hiển thị theo danh sách từng ảnh một
                for i, img_file in enumerate(uploaded_images):
                    with image_container.expander(f"Ảnh {i+1}: {img_file.name}"):
                        img = Image.open(img_file)
                        st.image(img, use_column_width=True)
                        # Thêm hình ảnh vào danh sách để phân tích
                        processed_images.append(img)
            
            # Button để phân tích các hình ảnh đã tải lên
            if st.button("Phân tích hình ảnh", type="primary", key="analyze_image_btn"):
                # Phân tích hình ảnh bằng Nova hoặc Claude
                if processed_images:
                    if selected_model == ModelType.NOVA:
                        # Call Nova via AWS Bedrock
                        result = analyze_frames_with_nova(
                            processed_images, 
                            prompt, 
                            temperature, 
                            top_p, 
                            top_k, 
                            max_tokens
                        )
                    else:  # Claude
                        # Call Claude via AWS Bedrock
                        result = analyze_frames_with_claude(
                            processed_images, 
                            prompt, 
                            temperature, 
                            top_p, 
                            top_k, 
                            max_tokens
                        )
                    
                    # Display results
                    st.subheader(f"Kết quả phân tích từ {selected_model.value}")
                    # Sử dụng expander để ẩn/hiện kết quả
                    with st.expander("Xem kết quả phân tích", expanded=False):
                        st.markdown(result)    

                    # Option to download as text file
                    df = display_results_as_table(result)
                    if df is not None:
                        st.subheader("Kết quả dạng bảng")
                        
                        # Thêm tùy chọn chỉnh sửa bảng nếu cần
                        edited_df = st.data_editor(
                            df,
                            num_rows="dynamic",
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Tạo nút tải xuống cho CSV
                        csv = edited_df.to_csv(index=False)
                        st.download_button(
                            label="Tải bảng về (CSV)",
                            data=csv,
                            file_name="ket_qua_phan_tich.csv",
                            mime="text/csv"
                        )
                        
                        # Tùy chọn tạo Excel
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            edited_df.to_excel(writer, sheet_name='Kết quả phân tích', index=False)
                        buffer.seek(0)
                        
                        st.download_button(
                            label="Tải bảng về (Excel)",
                            data=buffer,
                            file_name="ket_qua_phan_tich.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    # Option to download as text file vẫn giữ lại
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
            st.write("👆 Hãy tải lên hình ảnh để phân tích")

if __name__ == "__main__":
    main()