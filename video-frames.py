import cv2
import os
import argparse

def extract_frames(video_path, output_folder, percentage):
    """
    Trích xuất các frame từ video theo tỷ lệ phần trăm
    
    Args:
        video_path (str): Đường dẫn đến file video
        output_folder (str): Thư mục lưu các frame
        percentage (float): Tỷ lệ phần trăm frame cần trích xuất (0-100)
    """
    # Tạo thư mục output nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Đọc video
    video = cv2.VideoCapture(video_path)
    
    # Kiểm tra xem video có mở thành công không
    if not video.isOpened():
        print(f"Không thể mở file video: {video_path}")
        return
    
    # Lấy thông tin video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"Tổng số frame: {total_frames}")
    print(f"FPS: {fps}")
    
    # Tính số frame cần trích xuất
    num_frames_to_extract = int(total_frames * percentage / 100)
    print(f"Số frame sẽ trích xuất: {num_frames_to_extract}")
    
    if num_frames_to_extract <= 0:
        print("Tỷ lệ phần trăm quá nhỏ, không có frame nào được trích xuất")
        return
    
    # Tính khoảng cách giữa các frame cần trích xuất
    step = total_frames / num_frames_to_extract
    
    count = 0
    frame_index = 0
    
    while count < num_frames_to_extract:
        # Đặt vị trí đọc frame
        video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        
        # Đọc frame
        success, frame = video.read()
        
        if not success:
            break
        
        # Lưu frame
        output_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(output_path, frame)
        
        count += 1
        frame_index += step
        
        # Hiển thị tiến trình
        if count % 10 == 0:
            print(f"Đã trích xuất {count}/{num_frames_to_extract} frames")
    
    video.release()
    print(f"Hoàn thành! Đã trích xuất {count} frames vào thư mục {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất frame từ video theo tỷ lệ phần trăm")
    parser.add_argument("video_path", help="Đường dẫn đến file video")
    parser.add_argument("output_folder", help="Thư mục lưu các frame")
    parser.add_argument("percentage", type=float, help="Tỷ lệ phần trăm frame cần trích xuất (0-100)")
    
    args = parser.parse_args()
    
    extract_frames(args.video_path, args.output_folder, args.percentage)