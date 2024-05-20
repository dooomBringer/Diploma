import numpy as np
import argparse
import os
import tensorflow as tf
from PIL import Image
import glob
from concurrent.futures import ThreadPoolExecutor

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


def load_image(path):
    return Image.open(path)


def load_images(image_paths):
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_image, image_paths))
    return images


def load_image_into_numpy_array(image):
    return np.array(image)


def run_inference_for_image(model, category_index, image_path, output_dir):
    image = load_image(image_path)
    image_np = load_image_into_numpy_array(image)
    output_dict = run_inference_for_single_image(model, image_np)
    if np.any(output_dict['detection_scores'] > 0.5):
        visualize_and_save(image_np, output_dict, category_index, output_dir, image_path)


def visualize_and_save(image_np, output_dict, category_index, output_dir, image_path):
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    image_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"detection_output_{image_name}")
    Image.fromarray(image_np).save(save_path)


def run_inference_for_single_image(model, image):
    input_tensor = tf.convert_to_tensor(image)[tf.newaxis, ...]
    output_dict = model(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks(
            tf.convert_to_tensor(output_dict['detection_masks']),
            tf.convert_to_tensor(output_dict['detection_boxes']),
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    return output_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside images')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to model')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to labelmap')
    parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to image (or folder)')
    args = parser.parse_args()

    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)

    if os.path.isdir(args.image_path):
        image_paths = []
        for file_extension in ('*.png', '*.jpg', '*.jpeg'):
            image_paths.extend(glob.glob(os.path.join(args.image_path, file_extension)))

        with ThreadPoolExecutor() as executor:
            for image_path in image_paths:
                executor.submit(run_inference_for_image, detection_model, category_index, image_path, "outputs")
    else:
        run_inference_for_image(detection_model, category_index, args.image_path, "outputs")
