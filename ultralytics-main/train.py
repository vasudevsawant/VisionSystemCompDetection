from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
#model = YOLO("yolov8n.pt")  load a pretrained model (recommended for training)

# Use the model
model.train(data="C:\\Users\\z004uvjn\\Desktop\\Vision\\ultralytics-main\\yolov8_custom_model\\data.yaml", imgsz=640 , epochs=100)  # train the model
#metrics = model.val()  # evaluate model performance on the validation set
#results = model("C:\\Users\\z004uvjn\\Desktop\\Vision")  # predict on an image
# path = model.export(format="sx") export the model to ONNX format
