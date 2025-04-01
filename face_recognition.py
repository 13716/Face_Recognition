import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Library import FaceRecognitionLibrary

# Biến toàn cục
root = None
camera_active = False
current_camera = None
face_lib = FaceRecognitionLibrary()

def stop_camera():
    global camera_active, current_camera
    camera_active = False
    if current_camera is not None:
        current_camera.release()
        current_camera = None
    if root:
        root.update_status("Camera đã dừng")
        root.video_label.config(image='')

def register_face(name):
    global camera_active, current_camera, root
    if not name:
        return
    
    stop_camera()
    camera_active = True
    current_camera = cv2.VideoCapture(0)
    encodings = []
    
    # Tạo frame chứa nút chụp
    button_frame = tk.Frame(root.display_frame, bg="white")
    button_frame.pack(side=tk.BOTTOM, pady=10)
    
    capture_button = tk.Button(
        button_frame, 
        text="Chụp ảnh", 
        command=lambda: capture_image(),
        width=15, height=1,
        bg="#4CAF50", fg="white",
        font=("Arial", 10, "bold")
    )
    capture_button.pack(side=tk.LEFT, padx=5)
    
    finish_button = tk.Button(
        button_frame, 
        text="Hoàn thành", 
        command=lambda: finish_capture(),
        width=15, height=1,
        bg="#f44336", fg="white",
        font=("Arial", 10, "bold")
    )
    finish_button.pack(side=tk.LEFT, padx=5)
    
    def capture_image():
        nonlocal encodings
        ret, frame = current_camera.read()
        if ret:
            encoding = face_lib.get_face_encoding(frame)
            if encoding is not None:
                encodings.append(encoding)
                root.update_status(f"Đã chụp: {len(encodings)}/8 ảnh")
                if len(encodings) >= 8:
                    finish_capture()
            else:
                root.update_status("Không tìm thấy khuôn mặt!")
    
    def finish_capture():
        nonlocal encodings
        if len(encodings) > 0:
            avg_encoding = np.mean(encodings, axis=0)
            face_lib.save_face(name, avg_encoding)
            messagebox.showinfo("Thành công", f"Đã lưu khuôn mặt của {name}")
        else:
            messagebox.showerror("Lỗi", "Chưa có ảnh nào được chụp!")
        button_frame.destroy()
        stop_camera()
    
    def update_frame():
        if not camera_active:
            button_frame.destroy()
            return
        
        ret, frame = current_camera.read()
        if not ret:
            stop_camera()
            button_frame.destroy()
            messagebox.showerror("Lỗi", "Không thể kết nối camera")
            return
        
        # Vẽ khung nhận diện khuôn mặt
        faces = face_lib.detect_faces(frame)
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Da chup: {len(encodings)}/8 anh", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        root.update_display(frame)
        root.update_status(f"Đã chụp: {len(encodings)}/8 ảnh")
        
        if camera_active:
            root.after(10, update_frame)
    
    update_frame()

def register_face_from_images(name, folder_path):
    if not name or not folder_path:
        return
    
    encodings = face_lib.process_image_folder(folder_path)
    
    if encodings:
        avg_encoding = np.mean(encodings, axis=0)
        face_lib.save_face(name, avg_encoding)
        messagebox.showinfo("Thành công", 
                          f"Đã lưu khuôn mặt của {name} từ {len(encodings)} ảnh")
    else:
        messagebox.showerror("Lỗi", "Không tìm thấy khuôn mặt trong ảnh!")
    
    root.update_status("Sẵn sàng")

def train_knn():
    knn, cm = face_lib.train_model()
    if knn is None:
        messagebox.showwarning("Cảnh báo", "Không có dữ liệu để train!")
        return
    
    # Vẽ Confusion Matrix
    names, _ = face_lib.load_faces()
    unique_names = list(set(names))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=unique_names, yticklabels=unique_names)
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.title("Confusion Matrix")
    plt.show()
    
    root.update_status("Train mô hình hoàn tất")
    messagebox.showinfo("Thành công", "Đã train xong mô hình KNN!")

def recognize_face():
    global camera_active, current_camera
    
    if not os.path.exists("knn_model.pkl"):
        messagebox.showerror("Lỗi", "Chưa train mô hình! Vui lòng train KNN trước.")
        return
    
    stop_camera()
    camera_active = True
    current_camera = cv2.VideoCapture(0)
    
    # Giảm độ phân giải để tăng tốc độ xử lý
    current_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    current_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    current_camera.set(cv2.CAP_PROP_FPS, 15)
    
    frame_count = 0
    
    def update_frame():
        if not camera_active:
            return
        
        nonlocal frame_count
        ret, frame = current_camera.read()
        if not ret:
            stop_camera()
            return
        
        # Chỉ xử lý nhận diện mỗi 5 frame để tăng FPS
        frame_count += 1
        if frame_count % 5 == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            encoding = face_lib.get_face_encoding(small_frame)
            if encoding is not None:
                label, confidence = face_lib.predict_face(encoding)
                faces = face_lib.detect_faces(small_frame)
                for (x1, y1, x2, y2) in faces:
                    x1, y1, x2, y2 = x1*2, y1*2, x2*2, y2*2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f}%)", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                root.update_status(f"Nhận diện: {label} ({confidence:.2f}%)")
            else:
                root.update_status("Đang tìm kiếm khuôn mặt...")
        
        root.update_display(frame)
        
        if camera_active:
            root.after(5, update_frame)
    
    update_frame()

def recognize_face_from_image(image_path):
    if not image_path:
        return
    
    if not os.path.exists("knn_model.pkl"):
        messagebox.showerror("Lỗi", "Chưa train mô hình! Vui lòng train KNN trước.")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Lỗi", "Không thể đọc file ảnh")
        return
    
    root.update_status("Đang nhận diện...")
    
    encoding = face_lib.get_face_encoding(image)
    if encoding is None:
        root.update_status("Không tìm thấy khuôn mặt trong ảnh!")
        return
    
    label, confidence = face_lib.predict_face(encoding)
    
    # Vẽ kết quả
    faces = face_lib.detect_faces(image)
    for (x1, y1, x2, y2) in faces:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image,  f"{label} ({confidence:.2f}%)", (x1, y1-10), 
                   cv2.FONT_ITALIC, 0.9, (0, 255, 0), 2)
    
    root.update_display(image)
    root.update_status(f"Nhận diện: {label}({confidence:.2f}%)")

def delete_face_data():
    # Lấy danh sách tên
    names = face_lib.list_all_names()
    if not names:
        messagebox.showwarning("Cảnh báo", "Không có dữ liệu khuôn mặt nào!")
        return
    
    # Tạo cửa sổ dialog để chọn tên cần xóa
    dialog = tk.Toplevel(root)
    dialog.title("Xóa dữ liệu khuôn mặt")
    dialog.geometry("300x400")
    
    # Tạo Listbox để hiển thị danh sách tên
    listbox = tk.Listbox(dialog, selectmode=tk.SINGLE)
    listbox.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
    
    # Thêm các tên vào listbox
    for name in names:
        listbox.insert(tk.END, name)
    
    def confirm_delete():
        if not listbox.curselection():
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một tên để xóa!")
            return
        
        selected_name = listbox.get(listbox.curselection())
        if messagebox.askyesno("Xác nhận", f"Bạn có chắc muốn xóa dữ liệu của {selected_name}?"):
            success, message = face_lib.delete_face(selected_name)
            messagebox.showinfo("Kết quả", message)
            dialog.destroy()
            root.update_status(message)
    
    # Thêm nút xóa
    delete_btn = tk.Button(dialog, text="Xóa", command=confirm_delete,
                          bg="#f44336", fg="white", font=("Arial", 10, "bold"))
    delete_btn.pack(pady=10)

def main_menu():
    global root
    
    # Kiểm tra file model
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        messagebox.showwarning(
            "Cảnh báo", 
            "Không tìm thấy file shape_predictor_68_face_landmarks.dat. "
            "Vui lòng tải về từ https://github.com/davisking/dlib-models"
        )
    
    root = tk.Tk()
    root.title("Hệ thống nhận diện khuôn mặt")
    root.geometry("1200x700")
    root.config(bg="#f5f5f5")
    
    # Frame chính bên trái (điều khiển)
    control_frame = tk.Frame(root, bg="#f5f5f5", width=400)
    control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)
    control_frame.pack_propagate(False)
    
    # Tiêu đề
    tk.Label(control_frame, text="HỆ THỐNG NHẬN DIỆN KHUÔN MẶT", 
             font=("Arial", 16, "bold"), bg="#f5f5f5", fg="#333333").pack(pady=10)
    
    tk.Label(control_frame, text="Sử dụng dlib và OpenCV", 
             font=("Arial", 10), bg="#f5f5f5", fg="#666666").pack()
    
    # Frame chức năng chính
    main_functions = tk.LabelFrame(control_frame, text="Chức năng chính", 
                                 bg="#f5f5f5", font=("Arial", 12, "bold"))
    main_functions.pack(pady=20, fill=tk.X)
    
    tk.Button(main_functions, text="Đăng ký khuôn mặt mới", 
             command=lambda: register_face(simpledialog.askstring("Đăng ký", "Nhập tên người dùng:")),
             width=30, height=2, bg="#4CAF50", fg="white", 
             font=("Arial", 10, "bold")).pack(pady=10)
    
    tk.Button(main_functions, text="Đăng ký từ ảnh có sẵn", 
             command=lambda: register_face_from_images(
                 simpledialog.askstring("Đăng ký", "Nhập tên người dùng:"),
                 filedialog.askdirectory(title="Chọn thư mục chứa ảnh")
             ),
             width=30, height=2, bg="#2196F3", fg="white",
             font=("Arial", 10, "bold")).pack(pady=10)
    
    tk.Button(main_functions, text="Xóa dữ liệu khuôn mặt", 
             command=delete_face_data,
             width=30, height=2, bg="#FF5722", fg="white", 
             font=("Arial", 10, "bold")).pack(pady=10)
    
    # Frame nhận diện
    recognition_frame = tk.LabelFrame(control_frame, text="Nhận diện", 
                                    bg="#f5f5f5", font=("Arial", 12, "bold"))
    recognition_frame.pack(pady=20, fill=tk.X)
    
    tk.Button(recognition_frame, text="Train KNN", command=train_knn,
             width=30, height=1, bg="#FFC107", fg="black",
             font=("Arial", 10)).pack(pady=5)
    
    tk.Button(recognition_frame, text="Nhận diện từ camera", command=recognize_face,
             width=30, height=1, bg="#FF9800", fg="black",
             font=("Arial", 10)).pack(pady=5)
    
    tk.Button(recognition_frame, text="Nhận diện từ ảnh", 
             command=lambda: recognize_face_from_image(
                 filedialog.askopenfilename(
                     title="Chọn ảnh",
                     filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*"))
                 )
             ),
             width=30, height=1, bg="#9C27B0", fg="white",
             font=("Arial", 10)).pack(pady=5)
    
    tk.Button(recognition_frame, text="Dừng camera", command=stop_camera,
             width=30, height=1, bg="#f44336", fg="white",
             font=("Arial", 10)).pack(pady=5)
    
    # Frame hiển thị bên phải
    display_frame = tk.Frame(root, bg="white", width=700, height=600)
    display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)
    display_frame.pack_propagate(False)
    
    # Thêm display_frame vào root
    root.display_frame = display_frame
    
    # Label hiển thị camera/ảnh
    video_label = tk.Label(display_frame)
    video_label.pack(expand=True)
    
    # Label hiển thị trạng thái
    status_label = tk.Label(display_frame, text="Sẵn sàng", 
                           bg="white", font=("Arial", 10))
    status_label.pack(pady=10)
    
    # Thông tin
    tk.Label(control_frame, text="Lưu ý: Cần tải file shape_predictor_68_face_landmarks.dat", 
            font=("Arial", 8), bg="#f5f5f5", fg="#666666").pack(pady=20)
    
    def update_display(frame):
        if frame is not None:
            # Tối ưu việc chuyển đổi màu và resize
            frame = cv2.resize(frame, (500, 400))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
    
    def update_status(text):
        status_label.config(text=text)
    
    # Thêm các phương thức vào root
    root.video_label = video_label
    root.status_label = status_label
    root.update_display = update_display
    root.update_status = update_status
    
    root.mainloop()

if __name__ == "__main__":
    face_lib.init_db()
    main_menu()
