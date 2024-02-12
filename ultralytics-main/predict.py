from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('C:\\Users\\z004uvjn\\Desktop\\Vision\\ultralytics-main\\runs\\detect\\train\\weights\\best.pt')

# Define path to the image file
detection_output =model.predict(source='C:\\Users\\z004uvjn\\Desktop\\Vision\\ultralytics-main\\1.jpg', save=True, conf=0.2)



#print(detection_output)

