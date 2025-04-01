import cv2
import dlib
import numpy as np
import sqlite3
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from PIL import Image

# Tải bộ phát hiện khuôn mặt và dự đoán điểm đặc trưng của Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

class FaceRecognitionLibrary:
    def __init__(self):
        self.init_db()
    
    def init_db(self):
        """Khởi tạo cơ sở dữ liệu."""
        with sqlite3.connect("faces.db") as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
                              id INTEGER PRIMARY KEY AUTOINCREMENT,
                              name TEXT,
                              encoding BLOB)''')
            conn.commit()

    def save_face(self, name, encoding):
        """Lưu mã hóa khuôn mặt vào cơ sở dữ liệu."""
        with sqlite3.connect("faces.db") as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", 
                           (name, pickle.dumps(encoding)))
            conn.commit()

    def load_faces(self):
        """Tải tất cả khuôn mặt từ cơ sở dữ liệu."""
        with sqlite3.connect("faces.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, encoding FROM faces")
            data = cursor.fetchall()
        
        names = [name for name, _ in data]
        encodings = [pickle.loads(encoding) for _, encoding in data]
        return names, np.array(encodings)

    def get_face_encoding(self, image):
        """Lấy mã hóa khuôn mặt từ một ảnh."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if not faces:
            return None
        shape = predictor(gray, faces[0])
        return np.array(face_rec_model.compute_face_descriptor(image, shape))

    def process_image_folder(self, folder_path):
        """Xử lý một thư mục ảnh và trả về danh sách mã hóa."""
        encodings = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                img_path = os.path.join(folder_path, file)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                encoding = self.get_face_encoding(image)
                if encoding is not None:
                    encodings.append(encoding)
        return encodings

    def train_model(self):
        """Huấn luyện mô hình KNN và trả về ma trận nhầm lẫn."""
        names, encodings = self.load_faces()
        if not names:
            return None, None
        
        unique_names = list(set(names))
        n_neighbors = min(3, len(unique_names))
        
        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric='euclidean',
            weights='distance'
        )
        knn.fit(encodings, names)
        
        # Lưu mô hình
        with open("knn_model.pkl", "wb") as f:
            pickle.dump(knn, f)
        
        # Tính ma trận nhầm lẫn
        predictions = knn.predict(encodings)
        cm = confusion_matrix(names, predictions, labels=unique_names)
        
        return knn, cm

    def predict_face(self, encoding):
        """Dự đoán khuôn mặt từ mã hóa sử dụng mô hình đã huấn luyện."""
        if not os.path.exists("knn_model.pkl"):
            return None, 0.0  # Return 0% confidence if model is not available
        
        with open("knn_model.pkl", "rb") as f:
            knn = pickle.load(f)
        
        distances, indices = knn.kneighbors([encoding])
        nearest_distance = distances[0][0]
        
        # Trả về "Unknown" nếu khoảng cách gần nhất vượt ngưỡng
        threshold = 0.5
        if nearest_distance > threshold:
            return "Unknown", 0.0  # Return 0% confidence for unknown faces
        
        # Calculate confidence as a percentage
        confidence = max(0, 1 - nearest_distance / threshold) * 100
        
        return knn.predict([encoding])[0], confidence
     
    def detect_faces(self, image):
        """Phát hiện khuôn mặt trong một ảnh và trả về tọa độ của chúng."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        #############################
        for face in faces:
            face_features = predictor(image=gray, box=face)
            for i in range(0,68):
                x = face_features.part(i).x
                y = face_features.part(i).y
                cv2.circle(img=image, center=(x,y), radius=1, color=(0,0,255), thickness=1)
        #############################
        return [(face.left(), face.top(), face.right(), face.bottom()) for face in faces]

    def delete_face(self, name):
        """Xóa dữ liệu khuôn mặt khỏi cơ sở dữ liệu theo tên."""
        try:
            with sqlite3.connect("faces.db") as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM faces WHERE name = ?", (name,))
                if cursor.fetchone()[0] == 0:
                    return False, f"Không tìm thấy dữ liệu của {name}"
                
                cursor.execute("DELETE FROM faces WHERE name = ?", (name,))
                conn.commit()
                
                # Huấn luyện lại mô hình sau khi xóa
                self.train_model()
                return True, f"Đã xóa dữ liệu của {name} thành công"
        
        except Exception as e:
            return False, f"Lỗi khi xóa dữ liệu: {str(e)}"

    def list_all_names(self):
        """Liệt kê tất cả các tên trong cơ sở dữ liệu."""
        with sqlite3.connect("faces.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT name FROM faces")
            names = [row[0] for row in cursor.fetchall()]
        return names
