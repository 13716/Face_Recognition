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

# Load Dlib face detector và facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

class FaceRecognitionLibrary:
    def __init__(self):
        self.init_db()
    
    def init_db(self):
        """Khởi tạo database"""
        conn = sqlite3.connect("faces.db")
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT,
                      encoding BLOB)''')
        conn.commit()
        conn.close()

    def save_face(self, name, encoding):
        """Lưu khuôn mặt vào database"""
        conn = sqlite3.connect("faces.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", 
                      (name, pickle.dumps(encoding)))
        conn.commit()
        conn.close()

    def load_faces(self):
        """Load tất cả khuôn mặt từ database"""
        conn = sqlite3.connect("faces.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name, encoding FROM faces")
        data = cursor.fetchall()
        conn.close()
        
        names = []
        encodings = []
        for name, encoding in data:
            names.append(name)
            encodings.append(pickle.loads(encoding))
        return names, np.array(encodings)

    def get_face_encoding(self, image):
        """Lấy encoding của khuôn mặt từ ảnh"""
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
        if len(faces) == 0:
            return None
        shape = predictor(gray, faces[0])
        return np.array(face_rec_model.compute_face_descriptor(image, shape))

    def process_image_folder(self, folder_path):
        """Xử lý thư mục ảnh và trả về danh sách encodings"""
        encodings = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg','.webp')):
                img_path = os.path.join(folder_path, file)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                encoding = self.get_face_encoding(image)
                if encoding is not None:
                    encodings.append(encoding)
        return encodings

    def train_model(self):
        """Train mô hình KNN và trả về confusion matrix"""
        names, encodings = self.load_faces()
        if len(names) == 0:
            return None, None
        
        unique_names = list(set(names))
        n_neighbors = min(3, len(unique_names))
        
        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric='euclidean',
            weights='distance'
        )
        knn.fit(encodings, names)
        
        # Lưu model
        with open("knn_model.pkl", "wb") as f:
            pickle.dump(knn, f)
        
        # Tính confusion matrix
        predictions = knn.predict(encodings)
        cm = confusion_matrix(names, predictions, labels=unique_names)
        
        return knn, cm

    def predict_face(self, encoding):
        """Dự đoán khuôn mặt từ encoding sử dụng model đã train"""
        if not os.path.exists("knn_model.pkl"):
            return None
            
        with open("knn_model.pkl", "rb") as f:
            knn = pickle.load(f)
        # list_distance=[]
        # Thêm logic để kiểm tra khuôn mặt không xác định
        distances, indices = knn.kneighbors([encoding])
        nearest_distance = distances[0][0]
        
        print(nearest_distance)
        # Nếu khoảng cách đến neighbor gần nhất lớn hơn ngưỡng, trả về "Unknown"
        threshold = 0.5  # Có thể điều chỉnh ngưỡng này
        # if len(nearest_distance)
        if nearest_distance > threshold:
            return "Unknown"
        
        return knn.predict([encoding])[0]

    def detect_faces(self, image):
        """Phát hiện khuôn mặt trong ảnh và trả về tọa độ"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        return [(face.left(), face.top(), face.right(), face.bottom()) for face in faces]

    def delete_face(self, name):
        """Xóa khuôn mặt khỏi database theo tên"""
        try:
            conn = sqlite3.connect("faces.db")
            cursor = conn.cursor()
            
            # Kiểm tra xem tên có tồn tại không
            cursor.execute("SELECT COUNT(*) FROM faces WHERE name = ?", (name,))
            if cursor.fetchone()[0] == 0:
                conn.close()
                return False, f"Không tìm thấy dữ liệu của {name}"
            
            # Thực hiện xóa
            cursor.execute("DELETE FROM faces WHERE name = ?", (name,))
            conn.commit()
            conn.close()
            
            # Retrain model sau khi xóa
            self.train_model()
            return True, f"Đã xóa dữ liệu của {name} thành công"
        
        except Exception as e:
            return False, f"Lỗi khi xóa dữ liệu: {str(e)}"

    def list_all_names(self):
        """Liệt kê tất cả các tên trong database"""
        conn = sqlite3.connect("faces.db")
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT name FROM faces")
        names = [row[0] for row in cursor.fetchall()]
        conn.close()
        return names
