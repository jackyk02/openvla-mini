# SIMPLER Server Integration Guide

## Overview

The `openvla_utils.py` and `robot_utils.py` files have been updated to work with the `simpler_server.py` API, which now uses the PI0 model with action queue management and optional verifier.

## Changes Made

### 1. Updated `send_image_to_server()` in `openvla_utils.py`

**Before:**
- Accepted image file path
- Sent to arbitrary endpoint with `number_samples` and `temperature` parameters

**After:**
- Accepts raw image data (numpy array or PIL Image)
- Converts image to base64 JPEG encoding
- Sends to SIMPLER server's `/process_action` endpoint
- Includes required fields:
  - `instruction`: Task instruction
  - `original_instruction`: For rephrased instruction lookup
  - `image`: Base64 encoded JPEG
  - `observation_state`: Proprioceptive state (7-element array)
  - `timestep`: Current timestep for action queue management

### 2. Updated `get_vla_action()` in `openvla_utils.py`

**Before:**
- Used hardcoded image path
- Called wrong endpoint

**After:**
- Extracts raw image from `obs["full_image"]`
- Extracts proprioceptive state from `obs["proprio"]`
- Passes timestep for proper action queue management
- Supports original_instruction for rephrased lookup
- Returns action from server response
- Logs verifier score and selected instruction

**New Parameters:**
- `timestep` (int): Current timestep in episode (default: 0)
- `original_instruction` (str, optional): Original instruction for rephrased lookup
- `server_url` (str): Server endpoint URL (default: "http://localhost:5001/process_action")

### 3. Updated `get_action()` in `robot_utils.py`

**New Parameters:**
- `timestep` (int): Current timestep (default: 0)
- `original_instruction` (str, optional): Original instruction for rephrased lookup
- `server_url` (str, optional): Server URL override

## Usage Example

### Starting the SIMPLER Server

```bash
# Set environment variables (optional)
export REPHRASED_JSON_PATH="simpler_rephrased_final_eval_vlm.json"
export PRETRAINED_CHECKPOINT="juexzz/INTACT-pi0-finetune-rephrase-bridge"
export USE_VERIFIER="True"
export N_ACTION_STEPS="4"
export ACTION_ENSEMBLE_TEMP="-0.8"
export VERIFIER_PATH="/root/vla-clip/bridge_verifier/ensemble_182123_trainable_only.pt"
export PORT="5001"
export HOST="0.0.0.0"

# Start the server
cd ~/vla-clip/RoboMonkey/openvla-mini/experiments/robot/simpler
python simpler_server.py
```

### Updating Client Code (e.g., run_bridgev2_eval.py)

**Before:**
```python
while t < cfg.max_steps:
    # ... refresh obs ...
    
    action = get_action(
        cfg,
        model,
        obs,
        task_label,
        processor=processor,
    )
    
    obs, _, _, _, _ = env.step(action)
    t += 1
```

**After:**
```python
while t < cfg.max_steps:
    # ... refresh obs ...
    
    action = get_action(
        cfg,
        model,
        obs,
        task_label,
        processor=processor,
        timestep=t,  # IMPORTANT: Pass timestep for action queue management
        original_instruction=None,  # Optional: set if different from task_label
        server_url=None  # Optional: override default server URL
    )
    
    obs, _, _, _, _ = env.step(action)
    t += 1
```

### Resetting Between Episodes

When starting a new episode, you should reset the server-side state:

```python
import requests

def reset_server_session(server_url="http://localhost:5001", timestep=0):
    """Reset server-side action queue and history."""
    response = requests.post(
        f"{server_url.replace('/process_action', '')}/reset_session",
        json={"timestep": timestep},
        headers={"Content-Type": "application/json"}
    )
    return response.json()

# Call before each episode
obs, _ = env.reset()
reset_server_session()  # Reset server state
t = 0
```

## Important Notes

### 1. Observation Dictionary Requirements

The `obs` dictionary must contain:
- `"full_image"`: Raw image as numpy array (H, W, 3) or list of images
- `"proprio"`: Proprioceptive state as numpy array (7,) or list

### 2. Timestep Management

The timestep is critical for the server's action queue management:
- At timestep % n_action_steps == 0, the server generates new actions
- At other timesteps, the server uses cached actions from the queue
- Default n_action_steps = 4

### 3. Action Queue Storage

The server stores action queues server-side, keyed by batch_number = timestep // n_action_steps. Clients don't need to track action queues.

### 4. Action History

The server maintains action history server-side (last 6 actions). Clients don't need to pass or track action history.

### 5. Rephrased Instructions

The server automatically loads rephrased instructions from JSON based on the `original_instruction` field using exact match against the `rephrases_original` field in the JSON.

## Server Response Format

```json
{
    "status": "success",
    "action": [x, y, z, roll, pitch, yaw, gripper],  // 7-element action array
    "selected_instruction": "pick up the cup",  // Selected instruction (may be rephrased)
    "verifier_score": 0.8234,  // Verifier confidence score (or null)
    "action_queue": [...],  // Remaining action queue (for debugging)
    "action_history_update": null  // No longer needed by client
}
```

## Backward Compatibility

The updated functions maintain backward compatibility:
- All new parameters are optional with sensible defaults
- The function signatures remain compatible with existing code
- Unused parameters (vla, processor, etc.) are kept for compatibility

## Testing

To test the integration:

1. Start the SIMPLER server
2. Check health: `curl http://localhost:5001/health`
3. Run your evaluation script with updated `get_action` calls
4. Monitor server logs for action generation and verifier scores

