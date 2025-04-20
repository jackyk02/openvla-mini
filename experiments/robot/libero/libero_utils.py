"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)

from PIL import Image
import os

def process_image(image_path, output_dir="./transfer_images/", crop_scale=0.9, target_size=(224, 224), batch_size=1):
    """
    Process an image by center-cropping and resizing using TensorFlow.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save the processed image
        crop_scale (float): The area of the center crop with respect to the original image
        target_size (tuple): Target size for the processed image (height, width)
        batch_size (int): Batch size for processing
        
    Returns:
        str: Path to the processed image
    """
    def crop_and_resize(image, crop_scale, batch_size, target_size):
        """
        Center-crops an image and resizes it back to target size.
        
        Args:
            image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C)
            crop_scale: The area of the center crop with respect to the original image
            batch_size: Batch size
            target_size: Tuple of (height, width) for the output image
        """
        # Handle input dimensions
        if image.shape.ndims == 3:
            image = tf.expand_dims(image, axis=0)
            expanded_dims = True
        else:
            expanded_dims = False

        # Calculate crop dimensions
        new_scale = tf.reshape(
            tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), 
            shape=(batch_size,)
        )
        
        # Calculate bounding box
        offsets = (1 - new_scale) / 2
        bounding_boxes = tf.stack(
            [
                offsets,          # height offset
                offsets,          # width offset
                offsets + new_scale,  # height + offset
                offsets + new_scale   # width + offset
            ],
            axis=1
        )

        # Perform crop and resize
        image = tf.image.crop_and_resize(
            image, 
            bounding_boxes, 
            tf.range(batch_size), 
            target_size
        )

        # Remove batch dimension if input was 3D
        if expanded_dims:
            image = image[0]

        return image

    try:
        # Load and convert image to tensor
        image = Image.open(image_path)
        image = image.convert("RGB")

        current_size = image.size  # Returns (width, height)
        
        # Check if current size matches target size (accounting for PIL's width,height order)
        if current_size == (target_size[1], target_size[0]):
            return image_path
            
        image = tf.convert_to_tensor(np.array(image))
        
        # Store original dtype
        original_dtype = image.dtype

        # Convert to float32 [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Apply transformations
        image = crop_and_resize(image, crop_scale, batch_size, target_size)

        # Convert back to original dtype
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, original_dtype, saturate=True)

        # Convert to PIL Image and save
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

        image.save(image_path)

        return None

    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")


def save_reward_img(image):
    # raw_to_tf
    image = tf.image.encode_jpeg(image)  # Encode as JPEG, as done in RLDS dataset builder
    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    image = tf.image.resize(
        image, (256, 256), method="lanczos3", antialias=True
    )
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    image = tf.io.encode_jpeg(image, quality=95)

    # susie
    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    image = tf.image.resize(
        image, (256, 256), method="lanczos3", antialias=True
    )
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    image = image.numpy()

    import os
    os.makedirs("/root/openvla-mini/transfer_images/", exist_ok=True)
    Image.fromarray(image).save(f"/root/openvla-mini/transfer_images/reward_img.jpg")

    # resize down to 224x224
    process_image(
        "/root/openvla-mini/transfer_images/reward_img.jpg",
        output_dir="./output/",
        crop_scale=0.9,
        target_size=(224, 224),
        batch_size=1
    )


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    save_reward_img(img)
    img = resize_image(img, resize_size)
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
