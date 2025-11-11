"""Utils for evaluating the OpenVLA policy."""

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
import imageio

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.load import load_vla

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

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
    
def get_prismatic_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Prepare for model loading.
    print(f"[*] Initializing Generation Playground with `{cfg.model_family}`")
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    # set_seed(cfg.seed)
    # Load VLA checkpoint.
    print(f"Loading VLM from checkpoint: {cfg.pretrained_checkpoint}")
    vla = load_vla(
        cfg.pretrained_checkpoint,
        hf_token=hf_token,
        load_for_training=False,
    )
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"
    # Cast to half precision.
    vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
    vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(DEVICE)
    return vla


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="cuda:0",
    )

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def apply_center_crop(im, t_h, t_w):
    """
    Source: https://github.com/ARISE-Initiative/robomimic/blob/5dee58f9cc1235010d0877142b54d0e82dd23986/robomimic/utils/obs_utils.py#L268

    Takes a center crop of an image.

    Args:
        im (np.array or torch.Tensor): image of shape (..., height, width, channel)
        t_h (int): height of crop
        t_w (int): width of crop

    Returns:
        im (np.array or torch.Tensor): center cropped image
    """
    assert im.shape[-3] >= t_h and im.shape[-2] >= t_w
    assert im.shape[-1] in [1, 3, 6]
    crop_h = int((im.shape[-3] - t_h) / 2)
    crop_w = int((im.shape[-2] - t_w) / 2)
    return im[..., crop_h : crop_h + t_h, crop_w : crop_w + t_w, :]

import requests
import base64
import os
import argparse
import time

def send_image_to_server(server_url, image, instruction, observation_state, timestep, original_instruction=None):
    """
    Send an image, instruction, and observation state to the SIMPLER server to get action.
    
    Args:
        server_url (str): URL of the server endpoint (e.g., "http://localhost:5001/process_action")
        image (np.ndarray or PIL.Image): The image to send (will be converted to base64)
        instruction (str): The task instruction
        observation_state (np.ndarray or dict): Proprioceptive state (7-element array or dict with 'agent' key)
        timestep (int): Current timestep in the episode
        original_instruction (str, optional): Original instruction for rephrased lookup. Defaults to instruction.
        
    Returns:
        dict: Server response containing action and other info
    """
    try:
        # Convert image to PIL Image if it's a numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert PIL Image to base64
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_data = buffer.getvalue()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        # Convert observation_state to list if it's a numpy array
        if isinstance(observation_state, np.ndarray):
            observation_state = observation_state.tolist()
        input_observation = {'agent': {'eef_pos': observation_state}}
        # Prepare the request data
        payload = {
            "instruction": instruction,
            "original_instruction": original_instruction if original_instruction else instruction,
            "image": img_base64,
            "observation_state": input_observation,
            "timestep": timestep
        }
        
        # Send the request to the server
        response = requests.post(
            server_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60  # Add timeout for long-running verifier computations
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Server returned status code {response.status_code}: {response.text}"}
            
    except Exception as e:
        return {"error": str(e)}

def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False, 
                   timestep=0, original_instruction=None, server_url="http://localhost:5001/process_action"):
    """
    Generates an action with the VLA policy by calling the SIMPLER server API.
    
    Args:
        vla: VLA model (not used when calling server, kept for compatibility)
        processor: Processor (not used when calling server, kept for compatibility)
        base_vla_name: Base VLA name (not used when calling server, kept for compatibility)
        obs (dict): Observation dictionary containing 'full_image' and 'proprio' keys
        task_label (str): Task instruction
        unnorm_key: Unnormalization key (not used when calling server, kept for compatibility)
        center_crop (bool): Center crop flag (not used when calling server, kept for compatibility)
        timestep (int): Current timestep in the episode (important for action queue management)
        original_instruction (str, optional): Original instruction for rephrased lookup. Defaults to task_label.
        server_url (str): URL of the SIMPLER server endpoint
        
    Returns:
        np.ndarray: Action array of shape (7,)
    """
    # Extract image and observation state from obs
    image = obs["full_image"]
    
    # Handle list of images (take first one)
    if isinstance(image, list):
        image = image[0]
    
    # Get proprioceptive state
    observation_state = obs.get("proprio", None)
    if observation_state is None:
        raise ValueError("obs must contain 'proprio' key with proprioceptive state")
    
    # Prepare instruction
    instruction = task_label.lower()
    
    # Call server API
    result = send_image_to_server(
        server_url=server_url,
        image=image,
        instruction=instruction,
        observation_state=observation_state,
        timestep=timestep,
        original_instruction=original_instruction if original_instruction else instruction
    )
    
    # Check for errors
    if "error" in result:
        raise RuntimeError(f"Server error: {result['error']}")
    
    # Extract action from response
    action = np.array(result['action'])
    
    # Optionally log additional info
    if result.get('verifier_score') is not None:
        print(f"Verifier score: {result['verifier_score']:.4f}")
    if result.get('selected_instruction') is not None:
        print(f"Selected instruction: {result['selected_instruction']}")
    
    return action


def get_prismatic_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False, **kwargs):
    """Generates an action with the VLA policy."""

    if not isinstance(obs["full_image"], list):
        obs["full_image"] = [obs["full_image"]]

    processed_images = []

    for img in obs["full_image"]:
        image = Image.fromarray(img)
        image = image.convert("RGB")

        # (If trained with image augmentations) Center crop image and then resize back up to original size.
        # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), we must multiply
        #            the original height and width by sqrt(0.9) -- not 0.9!
        if center_crop:
            temp_image = np.array(image)  # (H, W, C)
            crop_scale = 0.9
            sqrt_crop_scale = math.sqrt(crop_scale)
            temp_image_cropped = apply_center_crop(
                temp_image,
                t_h=int(sqrt_crop_scale * temp_image.shape[0]),
                t_w=int(sqrt_crop_scale * temp_image.shape[1]),
            )
            temp_image = Image.fromarray(temp_image_cropped)
            temp_image = temp_image.resize(
                image.size, Image.Resampling.BILINEAR
            )  # IMPORTANT: dlimp uses BILINEAR resize
            image = temp_image

        processed_images.append(image)

    # extract for single image
    if len(processed_images) == 1:
        processed_images = processed_images[0]

    action = vla.predict_action(processed_images, task_label, unnorm_key=unnorm_key, **kwargs)
    return action
