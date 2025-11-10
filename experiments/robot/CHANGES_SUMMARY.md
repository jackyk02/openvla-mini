# Summary of Changes

## Files Modified

1. **`/root/openvla-mini/experiments/robot/openvla_utils.py`**
   - Updated `send_image_to_server()` function
   - Updated `get_vla_action()` function

2. **`/root/openvla-mini/experiments/robot/robot_utils.py`**
   - Updated `get_action()` function to pass new parameters

3. **Created documentation**
   - `/root/SIMPLER_SERVER_INTEGRATION_GUIDE.md` - Detailed integration guide

---

## Key Changes

### `send_image_to_server()` - New Signature

```python
def send_image_to_server(
    server_url,           # "http://localhost:5001/process_action"
    image,                # numpy array or PIL Image (NOT file path)
    instruction,          # Task instruction string
    observation_state,    # Proprioceptive state (7-element array)
    timestep,             # Current timestep (int)
    original_instruction  # Optional: for rephrased lookup
)
```

**What it does:**
- Converts raw image to base64 JPEG
- Sends all required data to SIMPLER server's `/process_action` endpoint
- Returns server response with action and metadata

---

### `get_vla_action()` - New Parameters

```python
def get_vla_action(
    vla,                    # (kept for compatibility)
    processor,              # (kept for compatibility)
    base_vla_name,          # (kept for compatibility)
    obs,                    # Must contain 'full_image' and 'proprio'
    task_label,             # Task instruction
    unnorm_key,             # (kept for compatibility)
    center_crop=False,      # (kept for compatibility)
    timestep=0,             # NEW: Current timestep
    original_instruction=None,  # NEW: For rephrased lookup
    server_url="http://localhost:5001/process_action"  # NEW: Server URL
)
```

**What it does:**
- Extracts `image` from `obs["full_image"]`
- Extracts `observation_state` from `obs["proprio"]`
- Calls `send_image_to_server()` with proper parameters
- Returns action as numpy array
- Logs verifier score and selected instruction

---

### `get_action()` - New Parameters

```python
def get_action(
    cfg,
    model,
    obs,
    task_label,
    processor=None,
    timestep=0,                # NEW: Current timestep
    original_instruction=None, # NEW: For rephrased lookup
    server_url=None            # NEW: Server URL override
)
```

**What it does:**
- Passes new parameters to `get_vla_action()` when using OpenVLA model
- Maintains backward compatibility with default values

---

## Minimal Code Change Required

### In your evaluation script (e.g., run_bridgev2_eval.py):

**Change this:**
```python
action = get_action(
    cfg, model, obs, task_label, processor=processor
)
```

**To this:**
```python
action = get_action(
    cfg, model, obs, task_label, processor=processor, timestep=t
)
```

That's it! The `timestep=t` parameter is the only required change for basic functionality.

---

## Data Flow

```
Your Script
    ↓
    obs = {
        "full_image": numpy_array,  # Raw image from camera
        "proprio": numpy_array       # 7-element state [x,y,z,roll,pitch,yaw,gripper]
    }
    ↓
get_action(cfg, model, obs, task_label, timestep=t)
    ↓
get_vla_action(...)  # Extracts image and proprio from obs
    ↓
send_image_to_server(...)  # Converts image to base64, sends HTTP POST
    ↓
SIMPLER Server (simpler_server.py)
    - Processes with PI0 model
    - Uses verifier to select best action
    - Manages action queue server-side
    - Tracks action history server-side
    ↓
Response: {"action": [7-element array], "verifier_score": float, ...}
    ↓
Return action to your script
```

---

## Server Requirements

The SIMPLER server expects from `obs`:

1. **Image**: `obs["full_image"]`
   - Raw camera image (numpy array or PIL Image)
   - Will be converted to base64 JPEG

2. **Proprioceptive State**: `obs["proprio"]`
   - 7-element array: [x, y, z, roll, pitch, yaw, gripper]
   - Can be numpy array or list

These are the standard fields already present in Bridge V2 observations!

---

## Server Manages Automatically

The server now handles these internally (no client action needed):

1. **Action Queue**: Generated actions are queued server-side
2. **Action History**: Last 6 actions tracked server-side
3. **Rephrased Instructions**: Loaded from JSON file server-side
4. **Batch Management**: Keyed by timestep // n_action_steps

---

## Testing

1. **Start server:**
   ```bash
   cd ~/vla-clip/RoboMonkey/openvla-mini/experiments/robot/simpler
   python simpler_server.py
   ```

2. **Check health:**
   ```bash
   curl http://localhost:5001/health
   ```

3. **Run evaluation with updated code:**
   ```bash
   python experiments/robot/bridge/run_bridgev2_eval.py \
       --model_family openvla \
       --pretrained_checkpoint <dummy_value>
   ```
   
   (The checkpoint arg is kept for compatibility but not used since server loads its own model)

---

## Error Handling

If you see errors about missing keys:
- ✓ Ensure `obs["full_image"]` exists
- ✓ Ensure `obs["proprio"]` exists  
- ✓ Ensure server is running on correct port
- ✓ Check server logs for detailed error messages

