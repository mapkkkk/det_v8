from ultralytics import YOLO
import cv2

class yolov8:
    pre_trained_model = './models/yolov8n.pt'
    train_model_yaml = 'yolov8n.yaml'
    train_data_yaml = './data/target.yaml'
    detect_model = 'model.pt'
    wait_export_model = 'yolov8n.pt'
    val_model_yaml = 'yolov8n.yaml'
    val_dataset_yaml = 'coco128.yaml'
    train_epoch = 50
    val_epoch = 50
    
    def __init__(self):
        pass
    
    def train(self):
        model = YOLO(self.train_model_yaml)
        results = model.train(data=self.train_data_yaml, epochs=5)

    def val(self):
        model = YOLO(self.val_model_yaml)
        model.train(data=self.val_dataset_yaml, epochs=self.val_epoch)
        model.val()

    def detect(self, img):
        model = YOLO(self.detect_model)
        results = model.predict(source=img, save=True, save_txt=True)
        for result in results:
        # Detection
            result.boxes.xyxy   # box with xyxy format, (N, 4)
            result.boxes.xywh   # box with xywh format, (N, 4)
            result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
            result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
            result.boxes.conf   # confidence score, (N, 1)
            result.boxes.cls    # cls, (N, 1)
            # Each result is composed of torch.Tensor by default,
            # in which you can easily use following functionality:
            # result = result.cuda()
            # result = result.cpu()
            # result = result.to("cpu")
            result = result.numpy()
        return results

    def export_onnx(self):
        model = YOLO(self.wait_export_model)
        model.export(format='onnx', dynamic=True)