#!/usr/bin/env python3
"""
Defines class Yolo that uses the Yolo v3 algorithm to perform object detection
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    Class that uses Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Yolo class constructor
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            lines = f.readlines()
            self.class_names = []
            for name in lines:
                self.class_names.append(name[:-1])
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """
        Returns the output after passing through Sigmoid function
        output will be between 0 and 1
        """
        return (1. / (1. + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        for i, output in enumerate(outputs):
            anchors = self.anchors[i]
            grid_height, grid_width = output.shape[:2]

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]

            sigmoid_conf = self.sigmoid(output[..., 4])
            sigmoid_prob = self.sigmoid(output[..., 5:])

            box_conf = np.expand_dims(sigmoid_conf, axis=-1)
            box_class_prob = sigmoid_prob

            box_confidences.append(box_conf)
            box_class_probs.append(box_class_prob)

            b_wh = anchors * np.exp(t_wh)
            b_wh /= self.model.inputs[0].shape.as_list()[1:3]

            grid = np.tile(np.indices((grid_width, grid_height)).T,
                           anchors.shape[0]).reshape(
                               (grid_height, grid_width) + anchors.shape)

            b_xy = (self.sigmoid(t_xy) + grid) / [grid_width, grid_height]

            b_xy1 = b_xy - (b_wh / 2)
            b_xy2 = b_xy + (b_wh / 2)
            box = np.concatenate((b_xy1, b_xy2), axis=-1)
            box *= np.tile(np.flip(image_size, axis=0), 2)

            boxes.append(box)
        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Determines filtered bounding boxes from processed outputs
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i, b in enumerate(boxes):
            box_conf = box_confidences[i]
            box_class_prob = box_class_probs[i]

            box_score = box_conf * box_class_prob

            box_class = np.argmax(box_score, axis=-1)
            box_class_score = np.max(box_score, axis=-1)

            index = np.where(box_class_score >= self.class_t)

            filtered_boxes.append(b[index])
            box_classes.append(box_class[index])
            box_scores.append(box_class_score[index])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)
        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Suppresses all non-max filter boxes to return predicted bounding box
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for i, b in enumerate(filtered_boxes):
            box_class = box_classes[i]
            box_score = box_scores[i]

            index = np.where(box_score >= self.nms_t)

            box_predictions.append(b[index])
            predicted_box_classes.append()

    @staticmethod
    def load_images(folder_path):
        """
        Loads images
        """
        image_paths = glob.glob(folder_path + "/*")
        images = []
        for image in image_paths:
            images.append(cv2.imread(image))
        return (images, image_paths)
