import cv2
import os
import sys

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

class FaceDetector:
    def __init__(self, face_model_path, face_proto_path):
        if not os.path.exists(face_model_path) or not os.path.exists(face_proto_path):
            raise FileNotFoundError("Файлы модели не найдены.")

        self.faceNet = cv2.dnn.readNet(face_model_path, face_proto_path)
    
    def detect_faces(self, frame, conf_threshold=0.7):
        """Функция определения лиц на изображении"""
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        
        faceBoxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 
                            int(round(frameHeight/150)), 8)
        
        return frameOpencvDnn, faceBoxes
    
    def process_image(self, image_path, output_path=None, conf_threshold=0.7):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удается загрузить изображение: {image_path}")
        
        result_img, face_boxes = self.detect_faces(image, conf_threshold)
        
        if face_boxes:
            print(f"Найдено лиц: {len(face_boxes)}")
            for i, box in enumerate(face_boxes, 1):
                print(f"Лицо {i}: координаты {box}")
        else:
            print("Лица не распознаны")
        
        if output_path:
            cv2.imwrite(output_path, result_img)
            print(f"Результат сохранен: {output_path}")
        
        return result_img, face_boxes
    
    def process_video_stream(self, conf_threshold=0.7):
        video = cv2.VideoCapture(0)
        
        if not video.isOpened():
            raise RuntimeError("Не удается открыть камеру")
        
        print("Нажмите любую клавишу для выхода...")
        
        while cv2.waitKey(1) < 0:
            hasFrame, frame = video.read()
            if not hasFrame:
                cv2.waitKey()
                break
            
            resultImg, faceBoxes = self.detect_faces(frame, conf_threshold)
            
            if not faceBoxes:
                print("Лица не распознаны")
            else:
                print(f"Найдено лиц: {len(faceBoxes)}")
            
            cv2.imshow("Face detection", resultImg)
        
        video.release()
        cv2.destroyAllWindows()


def main():
    try:
        detector = FaceDetector(faceModel, faceProto)
        
        print("Обработка изображения")
        image_filename = "image.jpg"
        
        if os.path.exists(image_filename):
            result_img, face_boxes = detector.process_image(
                image_filename, 
                output_path="result_" + image_filename
            )
            
            cv2.imshow("Face Detection Result", result_img)
            print("Нажмите любую клавишу для продолжения...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Файл {image_filename} не найден. Пропускаем обработку изображения.")
        
        print("\n Видеокамера")
        user_input = input("Запустить видеокамеру? (y/n): ")
        if user_input.lower() == 'y':
            detector.process_video_stream()
    
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


def detect_faces_in_image(image_filename, output_filename=None, conf_threshold=0.7):
    detector = FaceDetector(faceModel, faceProto)
    
    if output_filename is None:
        name, ext = os.path.splitext(image_filename)
        output_filename = f"{name}_faces_detected{ext}"
    
    return detector.process_image(image_filename, output_filename, conf_threshold)


if __name__ == "__main__":
    main()