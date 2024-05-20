import numpy as np
import argparse
import tensorflow as tf
import cv2

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

import numpy as np
import argparse
import tensorflow as tf
import cv2

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


import os
import glob

# Функция для обработки видео из директории
def process_videos_from_directory(model, category_index, directory):
    output_directory = "outputs/"
    os.makedirs(output_directory, exist_ok=True)

    video_files = glob.glob(os.path.join(directory, "*.mp4"))

    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        output_file = os.path.join(output_directory, os.path.basename(video_file))
        process_video(model, category_index, cap, output_file)
        cap.release()



def process_video(model, category_index, cap, output_file):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    objects_detected = False

    while cap.isOpened():
        ret, image_np = cap.read()
        if not ret:
            break

        output_dict = run_inference_for_single_image(model, image_np)


        if output_dict['num_detections'] > 0:
            objects_detected = True
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8)
            out.write(image_np)

    cap.release()
    out.release()


    if objects_detected:
        os.rename(output_file, os.path.join("outputs", os.path.basename(output_file)))
    else:
        os.remove(output_file)


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)

    input_tensor = tf.convert_to_tensor(image)

    input_tensor = input_tensor[tf.newaxis,...]


    output_dict = model(input_tensor)




    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections


    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:

        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                    output_dict['detection_masks'], output_dict['detection_boxes'],
                                    image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def run_inference(model, category_index, cap):
    while cap.isOpened():
        ret, image_np = cap.read()
        if not ret:
            break


        output_dict = run_inference_for_single_image(model, image_np)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside videos in a directory')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Path to directory containing videos.')
    args = parser.parse_args()

    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)

    process_videos_from_directory(detection_model, category_index, args.directory)


