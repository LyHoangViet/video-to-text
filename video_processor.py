import cv2
import os
import numpy as np
import tempfile
from typing import List, Tuple
import streamlit as st
from dataclasses import dataclass

@dataclass
class VideoFrame:
    """Lớp đại diện cho một frame được trích xuất từ video."""
    image: np.ndarray  # Dữ liệu ảnh dạng numpy array
    timestamp: float   # Thời điểm của frame trong video (giây)
    frame_number: int  # Số thứ tự của frame trong video

class VideoProcessor:
    """
    Lớp xử lý video để trích xuất các frame theo các phương thức khác nhau.
    """
    
    def __init__(self, video_file):
        """
        Khởi tạo bộ xử lý video với file video đã được tải lên.
        
        Args:
            video_file: File video đã tải lên từ Streamlit
        """
        self.video_file = video_file
        self.temp_file_path = None
        self.cap = None
        
        # Lưu video vào file tạm thời để OpenCV có thể đọc được
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            self.temp_file_path = temp_file.name
            temp_file.write(video_file.read())
        
        # Mở video bằng OpenCV
        self.cap = cv2.VideoCapture(self.temp_file_path)
        
        # Lấy thông tin cơ bản của video
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def get_video_info(self) -> dict:
        """
        Trả về thông tin cơ bản của video.
        
        Returns:
            Dictionary chứa thông tin của video
        """
        return {
            "filename": self.video_file.name,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "resolution": (self.width, self.height)
        }
    
    def extract_frames_uniform(self, num_frames: int) -> List[VideoFrame]:
        """
        Trích xuất số lượng frames định trước, được phân bố đều trên toàn bộ video.
        
        Args:
            num_frames: Số lượng frames muốn trích xuất
            
        Returns:
            Danh sách các VideoFrame đã trích xuất
        """
        if not self.cap or not self.cap.isOpened():
            return []
        
        # Đảm bảo số frame không vượt quá tổng số frame của video
        num_frames = min(num_frames, self.frame_count)
        
        frames = []
        # Tính khoảng cách giữa các frames sẽ lấy
        if num_frames > 1:
            step = (self.frame_count - 1) / (num_frames - 1)
        else:
            step = 0
            
        # Reset video capture về đầu video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for i in range(num_frames):
            # Tính vị trí frame cần lấy
            frame_pos = int(i * step) if step > 0 else 0
            
            # Di chuyển đến vị trí frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            
            # Đọc frame
            ret, frame = self.cap.read()
            
            if ret:
                # Tính timestamp
                timestamp = frame_pos / self.fps if self.fps > 0 else 0
                
                # Chuyển từ BGR sang RGB vì OpenCV đọc ảnh dạng BGR
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Thêm frame vào danh sách kết quả
                frames.append(VideoFrame(
                    image=frame_rgb,
                    timestamp=timestamp,
                    frame_number=frame_pos
                ))
        
        return frames
    
    def extract_frames_interval(self, interval_seconds: float) -> List[VideoFrame]:
        """
        Trích xuất frames theo một khoảng thời gian cố định.
        
        Args:
            interval_seconds: Khoảng thời gian giữa các frames (giây)
            
        Returns:
            Danh sách các VideoFrame đã trích xuất
        """
        if not self.cap or not self.cap.isOpened() or interval_seconds <= 0:
            return []
        
        frames = []
        # Tính số frame cần bỏ qua để đạt được khoảng thời gian mong muốn
        frame_interval = int(interval_seconds * self.fps)
        
        # Reset video capture về đầu video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_pos = 0
        while frame_pos < self.frame_count:
            # Di chuyển đến vị trí frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            
            # Đọc frame
            ret, frame = self.cap.read()
            
            if ret:
                # Tính timestamp
                timestamp = frame_pos / self.fps if self.fps > 0 else 0
                
                # Chuyển từ BGR sang RGB vì OpenCV đọc ảnh dạng BGR
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Thêm frame vào danh sách kết quả
                frames.append(VideoFrame(
                    image=frame_rgb,
                    timestamp=timestamp,
                    frame_number=frame_pos
                ))
            
            # Cập nhật vị trí frame tiếp theo cần lấy
            frame_pos += frame_interval
        
        return frames
    
    def extract_frames_keyframes(self, threshold: float = 0.1, max_frames: int = 20) -> List[VideoFrame]:
        """
        Trích xuất các keyframes dựa trên sự thay đổi giữa các frames.
        
        Args:
            threshold: Ngưỡng khác biệt để xác định keyframe (0-1)
            max_frames: Số lượng frames tối đa muốn trích xuất
            
        Returns:
            Danh sách các VideoFrame đã trích xuất
        """
        if not self.cap or not self.cap.isOpened():
            return []
        
        keyframes = []
        prev_frame = None
        frame_number = 0
        
        # Reset video capture về đầu video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
                
            frame_number += 1
            
            # Chuyển thành ảnh xám để so sánh
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Tính độ khác biệt giữa frame hiện tại và frame trước đó
                diff = cv2.absdiff(gray, prev_frame)
                score = np.sum(diff) / (self.width * self.height * 255)
                
                # Nếu độ khác biệt lớn hơn ngưỡng, đây là keyframe
                if score > threshold:
                    # Tính timestamp
                    timestamp = (frame_number - 1) / self.fps if self.fps > 0 else 0
                    
                    # Chuyển từ BGR sang RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Thêm vào danh sách keyframes
                    keyframes.append(VideoFrame(
                        image=frame_rgb,
                        timestamp=timestamp,
                        frame_number=frame_number - 1
                    ))
            
            # Cập nhật frame trước đó
            prev_frame = gray
            
            # Nếu đã đủ số lượng keyframes cần thiết, dừng lại
            if len(keyframes) >= max_frames:
                break
        
        # Nếu không tìm thấy đủ keyframes, bổ sung bằng cách lấy frames đều
        if len(keyframes) == 0:
            return self.extract_frames_uniform(max_frames)
        elif len(keyframes) < max_frames:
            # Tính số frames cần bổ sung
            remaining = max_frames - len(keyframes)
            # Lấy thêm frames phân bố đều
            uniform_frames = self.extract_frames_uniform(remaining)
            # Kết hợp hai danh sách
            keyframes.extend(uniform_frames)
        
        return keyframes
    
    def save_frames_to_temp(self, frames: List[VideoFrame]) -> List[Tuple[str, float, int]]:
        """
        Lưu danh sách frames thành các file ảnh tạm thời.
        
        Args:
            frames: Danh sách VideoFrame cần lưu
            
        Returns:
            Danh sách các tuple (đường dẫn file, timestamp, frame_number)
        """
        temp_dir = tempfile.mkdtemp()
        saved_files = []
        
        for i, frame in enumerate(frames):
            # Tạo tên file
            file_path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
            
            # Chuyển lại từ RGB sang BGR để lưu
            frame_bgr = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
            
            # Lưu ảnh
            cv2.imwrite(file_path, frame_bgr)
            
            # Thêm vào danh sách kết quả
            saved_files.append((file_path, frame.timestamp, frame.frame_number))
        
        return saved_files
    
    def __del__(self):
        """
        Hàm hủy để giải phóng tài nguyên.
        """
        # Đóng video capture
        if self.cap:
            self.cap.release()
        
        # Xóa file tạm nếu còn tồn tại
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            try:
                os.unlink(self.temp_file_path)
            except:
                pass