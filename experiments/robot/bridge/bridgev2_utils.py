"""Utils for evaluating policies in real-world BridgeData V2 environments."""

import os
import sys
import time

import imageio
import numpy as np
import tensorflow as tf
import torch
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs
from PIL import Image

sys.path.append(".")
from experiments.robot.bridge.widowx_env import WidowXGym

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
BRIDGE_PROPRIO_DIM = 7
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


def get_widowx_env_params(cfg):
    """Gets (mostly default) environment parameters for the WidowX environment."""
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params["override_workspace_boundaries"] = cfg.bounds
    env_params["camera_topics"] = cfg.camera_topics
    env_params["return_full_image"] = True
    return env_params


def get_widowx_env(cfg, model=None):
    """Get WidowX control environment."""
    # Set up the WidowX environment parameters
    env_params = get_widowx_env_params(cfg)
    start_state = np.concatenate([cfg.init_ee_pos, cfg.init_ee_quat])
    env_params["start_state"] = list(start_state)
    # Set up the WidowX client
    widowx_client = WidowXClient(host=cfg.host_ip, port=cfg.port)
    widowx_client.init(env_params)
    env = WidowXGym(
        widowx_client,
        cfg=cfg,
        blocking=cfg.blocking,
    )
    return env


def get_next_task_label(task_label):
    """Prompt the user to input the next task."""
    if task_label == "":
        user_input = ""
        while user_input == "":
            user_input = input("Enter the task name: ")
        task_label = user_input
    else:
        user_input = input("Enter the task name (or leave blank to repeat the previous task): ")
        if user_input == "":
            pass  # Do nothing -> Let task_label be the same
        else:
            task_label = user_input
    print(f"Task: {task_label}")
    return task_label


def save_rollout_video(rollout_images, idx, batch_size, temperature, gaussian):
    """Saves an MP4 replay of an episode."""
    os.makedirs("./rollouts", exist_ok=True)
    mp4_path = f"./rollouts/rollout-{DATE_TIME}-{idx+1}-b{batch_size}-t{temperature}-g{gaussian}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=5)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")


def save_rollout_data(rollout_orig_images, rollout_images, rollout_states, rollout_actions, idx):
    """
    Saves rollout data from an episode.

    Args:
        rollout_orig_images (list): Original rollout images (before preprocessing).
        rollout_images (list): Preprocessed images.
        rollout_states (list): Proprioceptive states.
        rollout_actions (list): Predicted actions.
        idx (int): Episode index.
    """
    os.makedirs("./rollouts", exist_ok=True)
    path = f"./rollouts/rollout-{DATE_TIME}-{idx+1}.npz"
    # Convert lists to numpy arrays
    orig_images_array = np.array(rollout_orig_images)
    images_array = np.array(rollout_images)
    states_array = np.array(rollout_states)
    actions_array = np.array(rollout_actions)
    # Save to a single .npz file
    np.savez(path, orig_images=orig_images_array, images=images_array, states=states_array, actions=actions_array)
    print(f"Saved rollout data at path {path}")


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

def save_reward_img(image):
    # raw_to_tf
    image = tf.image.encode_jpeg(image)  # Encode as JPEG, as done in RLDS dataset builder
    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    image = image.numpy()

    import os
    os.makedirs("/home/jacky/Desktop/openvla-mini/transfer_images/", exist_ok=True)
    Image.fromarray(image).save(f"/home/jacky/Desktop/openvla-mini/transfer_images/reward_img.jpg")

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


def get_preprocessed_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    save_reward_img(obs["full_image"])
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    obs["full_image"] = resize_image(obs["full_image"], resize_size)
    return obs["full_image"]


def refresh_obs(obs, env):
    """Fetches new observations from the environment and updates the current observations."""
    new_obs = env.get_observation()
    obs["full_image"] = new_obs["full_image"]
    obs["image_primary"] = new_obs["image_primary"]
    obs["proprio"] = new_obs["proprio"]
    return obs
