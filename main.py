import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import threading
import numpy as np

# Absolute path to image file
image_path = 'E:/Project/ObjectDetected/image/pass.jpg'

# Paths to YOLO configuration and weights files
yolo_cfg_path = 'E:/Project/ObjectDetected/yolo/yolov3.cfg'
yolo_weights_path = 'E:/Project/ObjectDetected/yolo/yolov3.weights'
yolo_labels_path = 'E:/Project/ObjectDetected/yolo/coco.names'


def run1():
    cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    while True:
        im = cv2.imread(image_path)
        if im is None:
            print(f"Failed to load image from {image_path}")
            break

        cv2.imshow('live transmission', im)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def run2():
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    while True:
        im = cv2.imread(image_path)
        if im is None:
            print(f"Failed to load image from {image_path}")
            break

        # Load the YOLO model
        net = cv2.dnn.readNet(yolo_weights_path, yolo_cfg_path)
        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except TypeError:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        height, width, channels = im.shape
        blob = cv2.dnn.blobFromImage(im, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)
                cv2.putText(im, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('detection', im)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("started")

    # Load class labels
    with open(yolo_labels_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    thread1 = threading.Thread(target=run1)
    thread2 = threading.Thread(target=run2)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
