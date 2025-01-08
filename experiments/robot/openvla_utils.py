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

import requests
import json_numpy as json

import numpy as np
import numpy as np
from transformers import AutoConfig


class TokenActionConverter:
    def __init__(self, n_action_bins: int = 256, unnorm_key: str = "bridge_orig"):
        self.bins = np.linspace(-1, 1, n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size = 32000
        self.unnorm_key = unnorm_key
        self.config = AutoConfig.from_pretrained(
            "openvla/openvla-7b", trust_remote_code=True
        ).to_dict()
        self.norm_stats = self.config["norm_stats"]
        assert unnorm_key is not None
        if unnorm_key not in self.norm_stats:
            raise ValueError(
                f"The `unnorm_key` you chose ({unnorm_key = }) is not in the available statistics. "
                f"Please choose from: {self.norm_stats.keys()}"
            )

    def token_to_action(self, output_ids):
        """
        Convert token IDs to actions.

        Args:
            output_ids (list or np.ndarray): Token IDs to convert

        Returns:
            np.ndarray: The corresponding actions
        """
        predicted_action_token_ids = np.array(output_ids)
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.norm_stats[self.unnorm_key]["action"]
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(
            action_norm_stats["q01"]
        )
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) *
            (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions

    def action_to_token(self, actions):
        """
        Convert actions back to token IDs.

        Args:
            actions (np.ndarray): The actions to convert

        Returns:
            np.ndarray: The corresponding token IDs
        """
        # First, normalize the actions back to [-1, 1] range
        action_norm_stats = self.norm_stats[self.unnorm_key]["action"]
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(
            action_norm_stats["q01"]
        )

        # Reverse the unnormalization
        normalized_actions = np.where(
            mask,
            2 * (actions - action_low) / (action_high - action_low) - 1,
            actions
        )

        # Find the closest bin centers to the normalized actions
        discretized_actions = np.array([
            np.abs(self.bin_centers - val).argmin()
            for val in normalized_actions
        ])

        # Convert back to token ids
        output_ids = self.vocab_size - discretized_actions - 1
        output_ids = np.array(output_ids)
        output_ids = np.where(output_ids == 31745, 31744, output_ids)

        return output_ids

converter = TokenActionConverter()

def select_action_index(rewards, temperature=0.1):
    """
    Select an action index based on rewards using a combination of top-k sampling rewards
    and greedy action selection, with temperature-scaled softmax probabilities.
    
    Args:
        rewards: numpy array of rewards where the last element is the greedy reward
        temperature: temperature parameter for softmax (default: 0.1)
                    Lower values make the distribution more peaked (more deterministic)
    
    Returns:
        Selected action index in the original rewards array
    """
    # Convert rewards to numpy array if it isn't already
    rewards = np.array(rewards)
    
    # Separate sampling rewards and greedy reward
    sampling_rewards = rewards[:-1]
    
    # Get indices of top k sampling rewards (ensure integer type)
    k = min(len(sampling_rewards), 2)  # Handle cases with fewer than 3 sampling rewards
    top_k_indices = np.argsort(sampling_rewards)[-k:].astype(np.int64)
    
    # Combine top k indices with the greedy index (last index)
    greedy_index = np.array([len(rewards) - 1], dtype=np.int64)
    candidate_indices = np.concatenate([top_k_indices, greedy_index])
    
    # Get corresponding rewards for candidates
    candidate_rewards = rewards[candidate_indices]
    print(candidate_rewards)
    
    # Apply temperature-scaled softmax to get selection probabilities
    scaled_rewards = candidate_rewards / temperature
    exp_rewards = np.exp(scaled_rewards - np.max(scaled_rewards))  # subtract max for numerical stability
    probabilities = exp_rewards / np.sum(exp_rewards)
    print(probabilities)
    
    # Select index based on probabilities
    selected_candidate_idx = np.random.choice(len(candidate_indices), p=probabilities)
    selected_idx = int(candidate_indices[selected_candidate_idx])  # ensure integer output
    
    return selected_idx

def preprocess_actions(output_ids, action):
    # Convert arrays to numpy arrays if they aren't already
    output_ids = np.array(output_ids)
    output_ids = np.where(output_ids == 31775, 31774, output_ids)
    action = np.array(action)
    
    # Get the majority value for the last dimension of each row
    last_dim_values = output_ids[:, -1]
    majority_value = np.bincount(last_dim_values).argmax()
    
    # Create a mask for rows where the last value matches the majority
    majority_mask = (output_ids[:, -1] == majority_value)
    
    # Filter arrays to keep only rows with majority value in last dimension
    output_ids = output_ids[majority_mask]
    action = action[majority_mask]
    
    # Apply the original range filter
    range_mask = np.all((output_ids >= 31744) & (output_ids <= 32000), axis=1)
    output_ids = output_ids[range_mask]
    action = action[range_mask]
    
    # Get unique rows and their indices
    unique_rows, indices = np.unique(output_ids, axis=0, return_index=True)
    
    # Sort indices to maintain original order
    indices = sorted(indices)
    
    # Return both arrays with only unique rows, maintaining alignment
    return output_ids[indices], action[indices]

def majority_vote_outputs(output_ids):
    """
    Apply majority voting to get the most common token at each position across all sequences.
    
    Args:
        output_ids (numpy.ndarray): 2D array of token ids with shape (n_sequences, sequence_length)
    
    Returns:
        numpy.ndarray: 1D array containing the majority-voted sequence
    """
    # Convert to numpy array if not already
    output_ids = np.array(output_ids)
    
    # For each position, get the most common token
    majority_sequence = []
    for pos in range(output_ids.shape[1]):
        # Get all tokens at this position
        tokens_at_pos = output_ids[:, pos]
        # Find the most common token
        majority_token = np.bincount(tokens_at_pos).argmax()
        majority_sequence.append(majority_token)
    
    return np.array(majority_sequence)

def get_rewards(instruction, image_path, actions):
    # Initialize rewards list
    all_rewards = []
    
    # Process actions in batches of 4
    batch_size = 4
    num_batches = math.ceil(len(actions) / batch_size)
    
    for i in range(num_batches):
        # Get the current batch of actions
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(actions))
        action_batch = actions[start_idx:end_idx]
        
        # Prepare payload for the current batch
        payload = {
            "instruction": instruction,
            "image_path": image_path,
            "action": action_batch
        }
        
        # Send request to server
        response = requests.post("http://127.0.0.1:3100/process", data=json.dumps(payload))
        response_data = json.loads(response.text)
        
        # Extend rewards list with batch results
        all_rewards.extend(response_data["rewards"])
    
    return all_rewards

def get_batch_actions(instruction: str, image_path: str, batch_size: int = 4, temperature: float = 1.0):
    """
    Get batch predictions from the batch processing server.
    
    Args:
        instruction (str): The instruction for the robot
        image_path (str): Path to the input image
        batch_size (int, optional): Size of the batch. Defaults to 4.
        temperature (float, optional): Sampling temperature. Defaults to 1.0.
    
    Returns:
        numpy.ndarray: Array of predicted actions
    """
    # Verify image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Prepare the payload
    payload = {
        "instruction": instruction,
        "image_path": image_path,
        "batch_size": batch_size,
        "temperature": temperature
    }
    
    # Send request to server
    response = requests.post(
        "http://127.0.0.1:3200/batch",
        data=json.dumps(payload),
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code != 200:
        raise Exception(f"Error from server: {response.text}")
    
    response_data = json.loads(response.text)
    return np.array(response_data["output_ids"]), np.array(response_data["actions"])

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
    return None
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
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
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

#
def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""

    # only supports 1 image
    if isinstance(obs["full_image"], list):
        obs["full_image"] = obs["full_image"][0]

    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

        # Save processed image and path for Inference
        transfer_dir = f"./transfer_images/"
        os.makedirs(transfer_dir, exist_ok=True)
        image_path = f"{transfer_dir}/vla_processed_img.jpg"
        image.save(image_path)
    
    # Get action from SGLang
    instruction = task_label.lower()
    image_path = "/root/openvla-mini/transfer_images/vla_processed_img.jpg"
    # print(instruction)
    output_ids, actions = get_batch_actions(
        instruction=instruction,
        image_path=image_path,
        batch_size=32,
        temperature=0.5
    )
    output_ids, actions = preprocess_actions(output_ids, actions)
    mv_id = majority_vote_outputs(output_ids)
    final_action = converter.token_to_action(mv_id)

    # greedy_output_ids, greedy_actions = get_batch_actions(
    #     instruction=instruction,
    #     image_path=image_path,
    #     batch_size=1,
    #     temperature=0
    # )
    # greedy_output_ids, greedy_actions = preprocess_actions(greedy_output_ids, greedy_actions)

    # print(output_ids)

    # if len(output_ids)==1:
    #     return actions[0]
    
    # combine n_samples + greedy
    # output_ids = np.concatenate([output_ids, greedy_output_ids], axis=0)
    # actions = np.concatenate([actions, greedy_actions], axis=0)
    
    # reward_img = "/root/openvla-mini/transfer_images/reward_img.jpg"
    
    # rewards = get_rewards(instruction, reward_img, output_ids)
    
    # selected_index = select_action_index(rewards, temperature=0.5)
    # selected_index = np.argmax(rewards)
    # print("ids: ", output_ids)
    # print("continuous: ", actions)
    # return action
    # print("len: ", len(rewards))
    # print("selected: ", selected_index)
    return final_action


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
