
import os
import json
import torch
import gradio as gr
import argparse
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image
from torchvision import transforms
import time
import platform
import matplotlib.pyplot as plt
from glob import glob
import threading  # Import the threading module
from contextlib import nullcontext

# --- Assume models are in a 'models' subdirectory ---
try:
    from models.birefnet import BiRefNet  # Try local import
    LOCAL_MODEL_AVAILABLE = True
except ImportError:
    BiRefNet = None
    LOCAL_MODEL_AVAILABLE = False
    print("Warning: Local BiRefNet class not found.  Will attempt to load all models from Hugging Face.")

# --- Global Offline Mode State ---
OFFLINE_MODE_ENABLED = False

def set_offline_mode(enabled: bool):
    """Enable or disable offline mode globally."""
    global OFFLINE_MODE_ENABLED
    OFFLINE_MODE_ENABLED = enabled
    if enabled:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        print("Offline mode ENABLED - No internet connections will be attempted")
    else:
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('TRANSFORMERS_OFFLINE', None)
        print("Offline mode DISABLED - Internet connections allowed")

def parse_args():
    parser = argparse.ArgumentParser(description="Run the image segmentation app")
    parser.add_argument("--share", action="store_true", help="Enable sharing of the Gradio interface")
    parser.add_argument("--allowed_paths", type=str, default="", help="Comma-separated list of additional allowed paths")
    parser.add_argument("--offline", action="store_true", help="Enable 100%% offline mode - no internet connections will be made")
    return parser.parse_args()

torch.set_float32_matmul_precision('high')
os.environ['HOME'] = os.path.expanduser('~')
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Global variable to control batch processing ---
batch_processing_stop_event = threading.Event()  # Use an event for thread-safe stopping

# --- Mixed Precision Configuration ---
MIXED_PRECISION_OPTIONS = {
    'No Mixed Precision (FP32)': 'no',
    'Float16 (FP16)': 'fp16',
    'BFloat16 (BF16 - Recommended)': 'bf16'
}

def get_autocast_context(precision_mode: str):
    """Create appropriate autocast context for mixed precision inference."""
    if precision_mode == 'fp16':
        return torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    elif precision_mode == 'bf16':
        return torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        return nullcontext()

# --- Resolution Presets ---
RESOLUTION_PRESETS = {
    'Original (No Resize)': None,
    '1024x1024': (1024, 1024),
    '2048x2048': (2048, 2048),
    '4096x4096': (4096, 4096),
    'Custom': 'custom'
}

# --- Background Color Options ---
BACKGROUND_OPTIONS = {
    'Transparent (Default)': 'transparent',
    'White Background': 'white',
    'Black Background': 'black'
}

# --- Preset System ---
# Use absolute path based on script location to ensure presets are always found
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRESETS_FOLDER = os.path.join(SCRIPT_DIR, "presets")
LAST_USED_PRESET = "last_used.json"
LAST_SELECTED_PRESET_FILE = "last_selected_preset.txt"

# Default settings for the app
DEFAULT_SETTINGS = {
    'resolution_preset': '2048x2048',
    'resolution_custom': '2048x2048',
    'weights_file': 'BiRefNet_HR (High Resolution 2048x2048)',
    'precision_mode': 'BFloat16 (BF16 - Recommended)',
    'use_local': False,
    'offline_mode': False,
    'add_metadata': True,
    'background_color': 'Transparent (Default)',
    # Batch-specific
    'batch_folder': '',
    'output_folder': 'results',
    'display_images': False
}

def ensure_presets_folder():
    """Create presets folder if it doesn't exist."""
    if not os.path.exists(PRESETS_FOLDER):
        os.makedirs(PRESETS_FOLDER)
        print(f"Created presets folder: {PRESETS_FOLDER}")

def get_preset_list() -> List[str]:
    """Get list of available preset names (without .json extension)."""
    ensure_presets_folder()
    presets = []
    for f in os.listdir(PRESETS_FOLDER):
        if f.endswith('.json') and f != LAST_USED_PRESET:
            presets.append(f[:-5])  # Remove .json extension
    return sorted(presets)

def save_preset(name: str, settings: Dict[str, Any]) -> str:
    """Save settings to a preset file."""
    ensure_presets_folder()
    if not name or name.strip() == '':
        return "Error: Please enter a preset name."
    
    # Sanitize filename
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
    if not safe_name:
        return "Error: Invalid preset name."
    
    preset_path = os.path.join(PRESETS_FOLDER, f"{safe_name}.json")
    try:
        with open(preset_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
        print(f"Saved preset: {preset_path}")
        return f"✓ Preset '{safe_name}' saved successfully!"
    except Exception as e:
        return f"Error saving preset: {e}"

def load_preset(name: str) -> Dict[str, Any]:
    """Load settings from a preset file."""
    ensure_presets_folder()
    if not name or name.strip() == '':
        return DEFAULT_SETTINGS.copy()
    
    preset_path = os.path.join(PRESETS_FOLDER, f"{name}.json")
    if os.path.exists(preset_path):
        try:
            with open(preset_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            # Merge with defaults to handle missing keys
            merged = DEFAULT_SETTINGS.copy()
            merged.update(settings)
            print(f"Loaded preset: {preset_path}")
            return merged
        except Exception as e:
            print(f"Error loading preset {name}: {e}")
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()

def save_last_used(settings: Dict[str, Any]):
    """Save current settings as last used."""
    ensure_presets_folder()
    preset_path = os.path.join(PRESETS_FOLDER, LAST_USED_PRESET)
    try:
        with open(preset_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
        print(f"Saved last used settings to: {preset_path}")
    except Exception as e:
        print(f"Error saving last used settings: {e}")

def load_last_used() -> Dict[str, Any]:
    """Load last used settings, or defaults if not found."""
    ensure_presets_folder()
    preset_path = os.path.join(PRESETS_FOLDER, LAST_USED_PRESET)
    if os.path.exists(preset_path):
        try:
            with open(preset_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            # Merge with defaults to handle missing keys
            merged = DEFAULT_SETTINGS.copy()
            merged.update(settings)
            print(f"✓ Loaded last used settings from: {preset_path}")
            print(f"  - Model: {merged.get('weights_file', 'N/A')}")
            print(f"  - Resolution: {merged.get('resolution_preset', 'N/A')}")
            print(f"  - Background: {merged.get('background_color', 'N/A')}")
            return merged
        except Exception as e:
            print(f"Error loading last used settings: {e}")
    else:
        print(f"No last used settings found at: {preset_path}")
        print("Using default settings.")
    return DEFAULT_SETTINGS.copy()

def delete_preset(name: str) -> Tuple[str, List[str]]:
    """Delete a preset file."""
    ensure_presets_folder()
    if not name or name.strip() == '':
        return "Error: Please select a preset to delete.", get_preset_list()
    
    preset_path = os.path.join(PRESETS_FOLDER, f"{name}.json")
    if os.path.exists(preset_path):
        try:
            os.remove(preset_path)
            # Clear last selected if it was the deleted preset
            if load_last_selected_preset() == name:
                filepath = os.path.join(PRESETS_FOLDER, LAST_SELECTED_PRESET_FILE)
                if os.path.exists(filepath):
                    os.remove(filepath)
            return f"✓ Preset '{name}' deleted successfully!", get_preset_list()
        except Exception as e:
            return f"Error deleting preset: {e}", get_preset_list()
    return f"Error: Preset '{name}' not found.", get_preset_list()

def save_last_selected_preset(preset_name: str):
    """Save the name of the last selected preset."""
    ensure_presets_folder()
    filepath = os.path.join(PRESETS_FOLDER, LAST_SELECTED_PRESET_FILE)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(preset_name)
    except Exception as e:
        print(f"Error saving last selected preset: {e}")

def load_last_selected_preset() -> str:
    """Load the name of the last selected preset."""
    ensure_presets_folder()
    filepath = os.path.join(PRESETS_FOLDER, LAST_SELECTED_PRESET_FILE)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                preset_name = f.read().strip()
            # Verify the preset still exists
            if preset_name in get_preset_list():
                print(f"✓ Last selected preset: {preset_name}")
                return preset_name
        except Exception as e:
            print(f"Error loading last selected preset: {e}")
    return None

def open_folder():
    open_folder_path = os.path.abspath("results")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')

def parse_resolution(resolution_preset: str, resolution_custom: str, image_path: str = None, config_size=(2048, 2048)):
    """
    Parse resolution with multiple modes:
    - 'Original (No Resize)': Use original image resolution (None)
    - Preset values: Use predefined resolution tuples
    - 'Custom': Parse custom resolution string
    """
    try:
        # Check if it's a preset
        if resolution_preset in RESOLUTION_PRESETS:
            preset_value = RESOLUTION_PRESETS[resolution_preset]
            
            if preset_value is None:
                # Original resolution mode
                return None
            elif preset_value == 'custom':
                # Parse custom resolution
                if not resolution_custom or resolution_custom.strip() == '':
                    print("Custom resolution selected but no value provided. Using config size.")
                    return config_size
                parsed = [int(int(reso)//32*32) for reso in resolution_custom.strip().split('x')]
                if len(parsed) != 2:
                    raise ValueError("Resolution must be in format: WIDTHxHEIGHT")
                return tuple(parsed)
            else:
                # Direct preset tuple
                return preset_value
        else:
            # Fallback: try to parse as string
            if resolution_preset in [None, '', 'None', 'original']:
                return None
            parsed = [int(int(reso)//32*32) for reso in resolution_preset.strip().split('x')]
            return tuple(parsed)
    except Exception as e:
        print(f"Resolution parsing error: {e}. Using config size {config_size}.")
        return config_size

def generate_output_path(base_folder: str, image_path: str, resolution: tuple, 
                        model_name: str, add_metadata=True) -> str:
    """Generate organized output path with optional metadata."""
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    if add_metadata:
        # Create subfolder with model and resolution info
        if resolution is None:
            reso_str = "original"
        else:
            reso_str = f"{resolution[0]}x{resolution[1]}"
        
        subfolder = os.path.join(base_folder, f"{model_name}-reso_{reso_str}")
        os.makedirs(subfolder, exist_ok=True)
    else:
        subfolder = base_folder
        os.makedirs(subfolder, exist_ok=True)
    
    output_path = os.path.join(subfolder, f"{base_filename}.png")
    
    # Handle duplicates
    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(subfolder, f"{base_filename}_{counter:04d}.png")
        counter += 1
    
    return output_path

usage_to_weights_file = {
    'General (BiRefNet)': 'BiRefNet',
    'General-Lite (BiRefNet_T)': 'BiRefNet_T',
    'Portrait (BiRefNet-portrait)': 'BiRefNet-portrait',
    'DIS (BiRefNet-DIS5K)': 'BiRefNet-DIS5K',
    'HRSOD (BiRefNet-HRSOD)': 'BiRefNet-HRSOD',
    'COD (BiRefNet-COD)': 'BiRefNet-COD',
    'DIS-TR_TEs (BiRefNet-DIS5K-TR_TEs)': 'BiRefNet-DIS5K-TR_TEs',
    'BiRefNet_HR (High Resolution 2048x2048)': 'BiRefNet_HR',
}

def load_model(weights_file: str, use_local: bool = False, offline_mode: bool = None):
    """
    Load model without precision conversion.
    Precision is now handled by autocast during inference.
    
    Args:
        weights_file: Name of the model weights file
        use_local: If True, try to load from local .pth file first
        offline_mode: If True, use only cached/local files (no internet). 
                      If None, uses global OFFLINE_MODE_ENABLED setting.
    """
    global OFFLINE_MODE_ENABLED
    
    # Use global setting if not explicitly specified
    if offline_mode is None:
        offline_mode = OFFLINE_MODE_ENABLED
    
    # Set environment variables for offline mode
    if offline_mode:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    model_path = os.path.join("models", f"{weights_file}.pth")
    huggingface_path = f'ZhengPeng7/{weights_file}'
    birefnet = None  # Initialize to track loading status

    # Try local loading first if requested
    if use_local and LOCAL_MODEL_AVAILABLE and os.path.exists(model_path):
        try:
            # When loading from local .pth, we need to create model with bb_pretrained=False
            # because the .pth file already contains the full model weights
            birefnet = BiRefNet(bb_pretrained=False)
            birefnet.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from local path: {model_path}")
        except Exception as e:
            print(f"Error loading from local path {model_path}: {e}.  Falling back to Hugging Face.")
            birefnet = None  # Reset on failure

    # Only try HuggingFace if local loading failed or wasn't attempted
    if birefnet is None:
        # Warn user if they expected local loading
        if use_local and not os.path.exists(model_path):
            if offline_mode:
                print(f"ERROR: Offline mode enabled but local model file not found at {model_path}")
                print("Please download the model first while online, or place the .pth file in the models folder.")
                return None
            else:
                print(f"Warning: Local model file not found at {model_path}. Downloading from Hugging Face...")
        
        try:
            # CRITICAL: Always use bb_pretrained=False when loading from pretrained
            # The pretrained weights already include backbone weights, so we don't need
            # to download them separately (which would fail in offline mode)
            hf_kwargs = {
                'bb_pretrained': False,  # Don't try to load backbone weights separately
            }
            
            if offline_mode:
                hf_kwargs['local_files_only'] = True
                print(f"Loading model from HuggingFace cache (offline mode): {huggingface_path}")
            
            if LOCAL_MODEL_AVAILABLE:
                birefnet = BiRefNet.from_pretrained(huggingface_path, **hf_kwargs)
                print(f"Loaded model from Hugging Face (BiRefNet): {huggingface_path}")
            else:
                from transformers import AutoModelForImageSegmentation
                # AutoModel doesn't accept bb_pretrained, remove it
                hf_kwargs_auto = {k: v for k, v in hf_kwargs.items() if k != 'bb_pretrained'}
                hf_kwargs_auto['trust_remote_code'] = True
                birefnet = AutoModelForImageSegmentation.from_pretrained(huggingface_path, **hf_kwargs_auto)
                print(f"Loaded model from Hugging Face (AutoModel): {huggingface_path}")

        except Exception as e:
            error_msg = str(e)
            if offline_mode and ("offline mode" in error_msg.lower() or "local_files_only" in error_msg.lower() or "couldn't find" in error_msg.lower()):
                print(f"ERROR: Model not found in cache. Please run the app once with internet to download: {huggingface_path}")
            else:
                print(f"Error loading from Hugging Face {huggingface_path}: {e}. Make sure model exists.")
            return None

    if birefnet is not None:
        birefnet.to(device)
        # Keep model in FP32 - autocast handles precision during forward pass
        birefnet.eval()
    return birefnet

# Default model will be loaded on first use, not at startup
# This allows --offline flag to be processed first
birefnet = None

def create_transform(image_size=None):
    """Create transform pipeline that optionally skips resize."""
    if image_size is None:
        # Original resolution mode - no resize
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Resize mode
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def extract_object(birefnet, imagepath, image_size=(2048, 2048), precision_mode='bf16', background_color='transparent'):
    """
    Extract object with mixed precision support and optional resizing.
    
    Args:
        birefnet: Model instance
        imagepath: Path to input image
        image_size: Target size (W, H) or None for original resolution
        precision_mode: 'no', 'fp16', or 'bf16'
        background_color: 'transparent', 'white', or 'black'
    """
    image = Image.open(imagepath)
    if image is None:
        raise ValueError(f"Could not open image at {imagepath}")

    image = image.convert("RGB")
    original_size = image.size

    if image_size is None:
        width, height = original_size
        eff_w = max(32, (width // 32) * 32)
        eff_h = max(32, (height // 32) * 32)
        effective_size = (eff_w, eff_h)
    else:
        effective_size = image_size

    transform_image = create_transform(effective_size)

    input_tensor = transform_image(image)

    if input_tensor.shape[0] == 4:
        input_tensor = input_tensor[:3, :, :]

    input_images = input_tensor.unsqueeze(0).to(device)  # Keep in FP32

    # Use autocast for mixed precision
    autocast_ctx = get_autocast_context(precision_mode)
    with autocast_ctx, torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().to(torch.float32).cpu()  # Explicit FP32 conversion

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(original_size)
    
    # Apply background color based on setting
    if background_color == 'transparent':
        # Original behavior - transparent background
        image = image.convert("RGBA")
        image.putalpha(mask)
    else:
        # Create background with specified color
        if background_color == 'white':
            bg_color = (255, 255, 255)
        else:  # black
            bg_color = (0, 0, 0)
        
        # Create background image
        background = Image.new("RGB", original_size, bg_color)
        
        # Composite the foreground onto the background using the mask
        image = image.convert("RGB")
        image = Image.composite(image, background, mask)
        
        # Convert to RGBA for consistency (but with opaque alpha)
        image = image.convert("RGBA")

    return image, mask

def process_single_image(image_path: str, resolution_preset: str, resolution_custom: str, 
                         output_folder: str, model_name: str, precision_mode: str = 'bf16',
                         add_metadata: bool = True, background_color: str = 'transparent') -> Tuple[str, str, float]:
    """
    Process a single image with enhanced resolution and output handling.
    
    Args:
        image_path: Path to input image
        resolution_preset: Preset resolution option
        resolution_custom: Custom resolution string (used if preset is 'Custom')
        output_folder: Base output folder
        model_name: Model identifier for metadata
        precision_mode: 'no', 'fp16', or 'bf16'
        add_metadata: Whether to add model/resolution info to output path
        background_color: 'transparent', 'white', or 'black'
    """
    start_time = time.time()

    # Parse resolution using new system
    image_size = parse_resolution(resolution_preset, resolution_custom, image_path)

    try:
        output_image, _ = extract_object(birefnet, image_path, image_size=image_size, 
                                         precision_mode=precision_mode, background_color=background_color)
    except ValueError as e:
        return str(e), "", 0.0

    # Generate output path with optional metadata
    output_path = generate_output_path(output_folder, image_path, image_size, model_name, add_metadata)

    output_image.save(output_path)
    processing_time = time.time() - start_time
    return image_path, output_path, processing_time

def predict_single(image: str, resolution_preset: str, resolution_custom: str, weights_file: str, 
                  use_local: bool, precision_mode: str, add_metadata: bool, 
                  offline_mode: bool = False, background_color: str = 'Transparent (Default)') -> Tuple[str, List[Tuple[str, str]]]:
    """
    Process single image with new parameter structure.
    
    Args:
        image: Input image path
        resolution_preset: Preset resolution option
        resolution_custom: Custom resolution (if preset is 'Custom')
        weights_file: Model selection
        use_local: Use local model file
        precision_mode: Mixed precision mode ('no', 'fp16', 'bf16')
        add_metadata: Add model/resolution info to output paths
        offline_mode: If True, use only cached/local files (no internet)
        background_color: Background color option from UI
    """
    global birefnet
    
    # Save current settings as last used
    current_settings = {
        'resolution_preset': resolution_preset,
        'resolution_custom': resolution_custom,
        'weights_file': weights_file,
        'precision_mode': precision_mode,
        'use_local': use_local,
        'offline_mode': offline_mode,
        'add_metadata': add_metadata,
        'background_color': background_color
    }
    save_last_used(current_settings)
    
    # Update global offline mode state
    set_offline_mode(offline_mode)
    
    model_name = usage_to_weights_file.get(weights_file, 'BiRefNet_HR')
    birefnet = load_model(model_name, use_local, offline_mode=offline_mode)
    if birefnet is None:
        if offline_mode:
            return "Error: Model loading failed in offline mode. Make sure the model is cached.", []
        return "Error: Model loading failed.", []
    
    output_folder = "results"
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert precision mode from display name to internal format
    precision_internal = MIXED_PRECISION_OPTIONS.get(precision_mode, 'bf16')
    
    # Convert background color from display name to internal format
    bg_color_internal = BACKGROUND_OPTIONS.get(background_color, 'transparent')
    
    input_path, output_path, proc_time = process_single_image(
        image, resolution_preset, resolution_custom, output_folder, 
        model_name, precision_internal, add_metadata, bg_color_internal
    )
    
    if output_path:
        return "Single image processing complete.", [(output_path, f"{proc_time:.4f} seconds")]
    else:
        return input_path, []


def predict_batch(resolution_preset: str, resolution_custom: str, weights_file: str, use_local: bool, 
                 precision_mode: str, batch_folder: str, output_folder: str, 
                 display_images: bool, add_metadata: bool, offline_mode: bool = False,
                 background_color: str = 'Transparent (Default)'):
    """
    Process batch of images with new parameter structure.
    
    Args:
        resolution_preset: Preset resolution option
        resolution_custom: Custom resolution (if preset is 'Custom')
        weights_file: Model selection
        use_local: Use local model file
        precision_mode: Mixed precision mode
        batch_folder: Input folder path
        output_folder: Output folder path
        display_images: Whether to display images in gallery
        add_metadata: Add model/resolution info to output paths
        offline_mode: If True, use only cached/local files (no internet)
        background_color: Background color option from UI
    """
    global birefnet
    
    # Save current settings as last used
    current_settings = {
        'resolution_preset': resolution_preset,
        'resolution_custom': resolution_custom,
        'weights_file': weights_file,
        'precision_mode': precision_mode,
        'use_local': use_local,
        'offline_mode': offline_mode,
        'add_metadata': add_metadata,
        'background_color': background_color,
        'batch_folder': batch_folder,
        'output_folder': output_folder,
        'display_images': display_images
    }
    save_last_used(current_settings)
    
    # Update global offline mode state
    set_offline_mode(offline_mode)
    
    batch_processing_stop_event.clear()  # Reset the stop event at the start
    model_name = usage_to_weights_file.get(weights_file, 'BiRefNet_HR')
    birefnet = load_model(model_name, use_local, offline_mode=offline_mode)
    if birefnet is None:
        if offline_mode:
            yield "Error: Model loading failed in offline mode. Make sure the model is cached.", []
            return
        yield "Error: Model loading failed.", []
        return

    os.makedirs(output_folder, exist_ok=True)
    results = []
    image_files = glob(os.path.join(batch_folder, '*'))
    total_images = len(image_files)
    processed_images = 0
    start_time = time.time()
    
    # Convert precision mode from display name to internal format
    precision_internal = MIXED_PRECISION_OPTIONS.get(precision_mode, 'bf16')
    
    # Convert background color from display name to internal format
    bg_color_internal = BACKGROUND_OPTIONS.get(background_color, 'transparent')

    for img_path in image_files:
        if batch_processing_stop_event.is_set():  # Check if the stop event is set
            break
        try:
            input_path, output_path, proc_time = process_single_image(
                img_path, resolution_preset, resolution_custom, output_folder, 
                model_name, precision_internal, add_metadata, bg_color_internal
            )
            if output_path:
                # Crucial Change: Return (filepath, filepath) tuples for Gallery
                if display_images:
                    results.append((output_path, output_path)) # Both paths are the same
                else:
                    results.append((output_path,)) # Append only output path, Gallery component will be handled accordingly

            processed_images += 1
            elapsed_time = time.time() - start_time
            avg_time_per_image = elapsed_time / processed_images
            estimated_time_left = avg_time_per_image * (total_images - processed_images)
            status = f"Processed {processed_images}/{total_images} images. Estimated time left: {estimated_time_left:.2f} seconds"
            print(status)

            if display_images:
               yield status, results
            else:
               yield status, [] # Always yield empty list when display_images is false

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue

    if batch_processing_stop_event.is_set():
        status_msg = f"Batch processing stopped. Processed {processed_images}/{total_images} images."
    else:
        status_msg = f"Batch processing complete. Processed {processed_images}/{total_images} images."
    yield status_msg, [] if not display_images else results


def stop_batch_processing():
    """Sets the global stop event to terminate batch processing."""
    batch_processing_stop_event.set()
    return "Batch processing will stop after the current image."

def create_interface():
    # Load last selected preset name and its settings
    last_selected_preset = load_last_selected_preset()
    
    # If a preset was selected, load its settings; otherwise use last_used or defaults
    if last_selected_preset:
        initial_settings = load_preset(last_selected_preset)
        print(f"✓ Loading settings from preset: {last_selected_preset}")
    else:
        initial_settings = load_last_used()
    
    with gr.Blocks() as demo:
        gr.Markdown("## SECourses Improved BiRefNet HQ V10 - https://www.patreon.com/posts/121679760")
        gr.Markdown("**New Features:** Background color option (White/Black), Preset save/load system, Auto-remember settings")

        with gr.Tab("Single Image Processing"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="filepath", label="Input Image", height=512)
                    
                    submit_button = gr.Button("Process Single Image", variant="primary")
                    
                    with gr.Row():
                        resolution_preset = gr.Dropdown(
                            choices=list(RESOLUTION_PRESETS.keys()),
                            value=initial_settings.get('resolution_preset', '2048x2048'),
                            label="Resolution Preset",
                            info="Select resolution or use original size"
                        )
                    
                    resolution_custom = gr.Textbox(
                        label="Custom Resolution (WIDTHxHEIGHT)",
                        placeholder="3072x3072",
                        value=initial_settings.get('resolution_custom', '2048x2048'),
                        visible=(initial_settings.get('resolution_preset') == 'Custom'),
                        info="Only used when 'Custom' preset is selected"
                    )
                    
                    with gr.Row():
                        weights_file = gr.Dropdown(
                            choices=list(usage_to_weights_file.keys()),
                            value=initial_settings.get('weights_file', "BiRefNet_HR (High Resolution 2048x2048)"),
                            label="Model Selection"
                        )
                        
                        precision_mode = gr.Dropdown(
                            choices=list(MIXED_PRECISION_OPTIONS.keys()),
                            value=initial_settings.get('precision_mode', 'BFloat16 (BF16 - Recommended)'),
                            label="Mixed Precision Mode",
                            info="BF16: Best balance. FP16: Faster. FP32: Highest quality."
                        )
                    
                    background_color = gr.Dropdown(
                        choices=list(BACKGROUND_OPTIONS.keys()),
                        value=initial_settings.get('background_color', 'Transparent (Default)'),
                        label="Background Color",
                        info="Replace transparent areas with white or black background"
                    )
                    
                    add_metadata_checkbox = gr.Checkbox(
                        label="Organize Output by Model/Resolution",
                        value=initial_settings.get('add_metadata', True),
                        info="Creates subfolders like 'BiRefNet_HR-reso_2048x2048' for better organization"
                    )
                    
                    use_local_checkbox = gr.Checkbox(
                        label="Use Local Model (.pth file)",
                        value=initial_settings.get('use_local', False),
                        info="Load from local .pth file in models/ folder instead of HuggingFace"
                    )
                    
                    offline_mode_checkbox = gr.Checkbox(
                        label="🔌 Offline Mode (100% No Internet)",
                        value=initial_settings.get('offline_mode', False),
                        info="Use ONLY cached/local models. No internet connections will be made. Models must be pre-downloaded."
                    )
                    
                    single_output_text = gr.Textbox(label="Processing Status")

                with gr.Column():
                    output_image = gr.Gallery(label="Output Image", elem_id="gallery", height=512)
                    btn_open_outputs = gr.Button("📁 Open Results Folder")
                    btn_open_outputs.click(fn=open_folder)
                    
                    # --- Preset Management Section (under Open Results Folder) ---
                    with gr.Accordion("💾 Preset Management", open=True):
                        with gr.Row():
                            with gr.Column(scale=2):
                                preset_dropdown = gr.Dropdown(
                                    choices=get_preset_list(),
                                    value=last_selected_preset,
                                    label="Load Preset",
                                    info="Select a saved preset to load",
                                    allow_custom_value=False
                                )
                            with gr.Column(scale=2):
                                preset_name_input = gr.Textbox(
                                    label="Preset Name",
                                    placeholder="Enter name for new preset",
                                    info="Name to save current settings as"
                                )
                            with gr.Column(scale=1):
                                refresh_presets_btn = gr.Button("🔄 Refresh", size="sm")
                        
                        with gr.Row():
                            save_preset_btn = gr.Button("💾 Save Preset", variant="primary")
                            delete_preset_btn = gr.Button("🗑️ Delete Preset", variant="stop")
                        
                        preset_status = gr.Textbox(label="Preset Status", interactive=False)

            # Show/hide custom resolution based on preset selection
            def update_custom_visibility(preset):
                return gr.update(visible=(preset == 'Custom'))
            
            resolution_preset.change(
                update_custom_visibility,
                inputs=[resolution_preset],
                outputs=[resolution_custom]
            )

        with gr.Tab("Batch Processing"):
            with gr.Row():
                with gr.Column():
                    batch_folder = gr.Textbox(
                        label="Input Folder Path",
                        placeholder="Path to folder containing images",
                        value=initial_settings.get('batch_folder', '')
                    )
                    
                    batch_button = gr.Button("Start Batch Processing", variant="primary")
                    
                    with gr.Row():
                        resolution_preset_batch = gr.Dropdown(
                            choices=list(RESOLUTION_PRESETS.keys()),
                            value=initial_settings.get('resolution_preset', '2048x2048'),
                            label="Resolution Preset"
                        )
                    
                    resolution_custom_batch = gr.Textbox(
                        label="Custom Resolution (WIDTHxHEIGHT)",
                        placeholder="3072x3072",
                        value=initial_settings.get('resolution_custom', '2048x2048'),
                        visible=(initial_settings.get('resolution_preset') == 'Custom')
                    )
                    
                    with gr.Row():
                        weights_file_batch = gr.Dropdown(
                            choices=list(usage_to_weights_file.keys()),
                            value=initial_settings.get('weights_file', "BiRefNet_HR (High Resolution 2048x2048)"),
                            label="Model Selection"
                        )
                        
                        precision_mode_batch = gr.Dropdown(
                            choices=list(MIXED_PRECISION_OPTIONS.keys()),
                            value=initial_settings.get('precision_mode', 'BFloat16 (BF16 - Recommended)'),
                            label="Mixed Precision Mode"
                        )
                    
                    background_color_batch = gr.Dropdown(
                        choices=list(BACKGROUND_OPTIONS.keys()),
                        value=initial_settings.get('background_color', 'Transparent (Default)'),
                        label="Background Color",
                        info="Replace transparent areas with white or black background"
                    )
                    
                    add_metadata_checkbox_batch = gr.Checkbox(
                        label="Organize Output by Model/Resolution",
                        value=initial_settings.get('add_metadata', True)
                    )
                    
                    use_local_checkbox_batch = gr.Checkbox(
                        label="Use Local Model (.pth file)",
                        value=initial_settings.get('use_local', False),
                        info="Load from local .pth file in models/ folder"
                    )
                    
                    offline_mode_checkbox_batch = gr.Checkbox(
                        label="🔌 Offline Mode (100% No Internet)",
                        value=initial_settings.get('offline_mode', False),
                        info="Use ONLY cached/local models. No internet connections will be made."
                    )
                    
                    display_images_checkbox = gr.Checkbox(
                        label="Display Images During Processing",
                        value=initial_settings.get('display_images', False),
                        info="Disable for faster processing of large batches"
                    )
                    
                    stop_button = gr.Button("Stop Batch Processing", variant="stop")
                    
                    batch_output_text = gr.Textbox(label="Batch Processing Status")

                with gr.Column():
                    output_folder_batch = gr.Textbox(
                        label="Output Folder Path",
                        value=initial_settings.get('output_folder', 'results')
                    )
                    output_image_batch = gr.Gallery(label="Output Images", elem_id="gallery_batch")

            # Show/hide custom resolution based on preset selection
            resolution_preset_batch.change(
                update_custom_visibility,
                inputs=[resolution_preset_batch],
                outputs=[resolution_custom_batch]
            )

        # --- Preset Management Functions ---
        def save_preset_handler(preset_name, current_preset, res_preset, res_custom, model, prec_mode, 
                                bg_color, add_meta, use_loc, offline, 
                                b_folder, out_folder, disp_imgs):
            """Save current settings to a preset file. If no name provided, overwrite current preset."""
            # If no preset name entered, use currently loaded preset
            if not preset_name or preset_name.strip() == '':
                if current_preset and current_preset.strip() != '':
                    preset_name = current_preset
                else:
                    return "Error: Please enter a preset name or select an existing preset first.", gr.update()
            
            settings = {
                'resolution_preset': res_preset,
                'resolution_custom': res_custom,
                'weights_file': model,
                'precision_mode': prec_mode,
                'background_color': bg_color,
                'add_metadata': add_meta,
                'use_local': use_loc,
                'offline_mode': offline,
                'batch_folder': b_folder,
                'output_folder': out_folder,
                'display_images': disp_imgs
            }
            result = save_preset(preset_name, settings)
            
            # Get sanitized name for selection (same logic as save_preset)
            safe_name = "".join(c for c in preset_name if c.isalnum() or c in (' ', '-', '_')).strip() if preset_name else ""
            
            # Save as last selected preset for next startup
            if safe_name:
                save_last_selected_preset(safe_name)
            
            # Return updated dropdown with new preset selected
            return result, gr.update(choices=get_preset_list(), value=safe_name if safe_name else None)
        
        def load_preset_handler(preset_name):
            """Load settings from a preset file and update both tabs."""
            if not preset_name:
                return [gr.update()] * 20 + ["Please select a preset to load."]
            
            # Save the selected preset name for next startup
            save_last_selected_preset(preset_name)
            
            settings = load_preset(preset_name)
            is_custom = settings.get('resolution_preset') == 'Custom'
            
            return [
                # Single Image Tab
                gr.update(value=settings.get('resolution_preset', '2048x2048')),  # resolution_preset
                gr.update(value=settings.get('resolution_custom', '2048x2048'), visible=is_custom),  # resolution_custom
                gr.update(value=settings.get('weights_file', 'BiRefNet_HR (High Resolution 2048x2048)')),  # weights_file
                gr.update(value=settings.get('precision_mode', 'BFloat16 (BF16 - Recommended)')),  # precision_mode
                gr.update(value=settings.get('background_color', 'Transparent (Default)')),  # background_color
                gr.update(value=settings.get('add_metadata', True)),  # add_metadata_checkbox
                gr.update(value=settings.get('use_local', False)),  # use_local_checkbox
                gr.update(value=settings.get('offline_mode', False)),  # offline_mode_checkbox
                # Batch Tab - duplicate settings for batch controls
                gr.update(value=settings.get('resolution_preset', '2048x2048')),  # resolution_preset_batch
                gr.update(value=settings.get('resolution_custom', '2048x2048'), visible=is_custom),  # resolution_custom_batch
                gr.update(value=settings.get('weights_file', 'BiRefNet_HR (High Resolution 2048x2048)')),  # weights_file_batch
                gr.update(value=settings.get('precision_mode', 'BFloat16 (BF16 - Recommended)')),  # precision_mode_batch
                gr.update(value=settings.get('background_color', 'Transparent (Default)')),  # background_color_batch
                gr.update(value=settings.get('add_metadata', True)),  # add_metadata_checkbox_batch
                gr.update(value=settings.get('use_local', False)),  # use_local_checkbox_batch
                gr.update(value=settings.get('offline_mode', False)),  # offline_mode_checkbox_batch
                # Batch-specific settings
                gr.update(value=settings.get('batch_folder', '')),  # batch_folder
                gr.update(value=settings.get('output_folder', 'results')),  # output_folder_batch
                gr.update(value=settings.get('display_images', False)),  # display_images_checkbox
                f"✓ Preset '{preset_name}' loaded successfully!"  # preset_status
            ]
        
        def refresh_presets_handler():
            """Refresh the preset dropdown list."""
            return gr.update(choices=get_preset_list())
        
        def delete_preset_handler(preset_name):
            """Delete a preset file."""
            result, preset_list = delete_preset(preset_name)
            return result, gr.update(choices=preset_list)
        
        # --- Preset Event Handlers ---
        save_preset_btn.click(
            save_preset_handler,
            inputs=[
                preset_name_input,
                preset_dropdown,  # Current preset for overwriting if no name entered
                resolution_preset, resolution_custom, weights_file, precision_mode,
                background_color, add_metadata_checkbox, use_local_checkbox, offline_mode_checkbox,
                batch_folder, output_folder_batch, display_images_checkbox
            ],
            outputs=[preset_status, preset_dropdown]
        )
        
        preset_dropdown.change(
            load_preset_handler,
            inputs=[preset_dropdown],
            outputs=[
                # Single Image Tab
                resolution_preset, resolution_custom, weights_file, precision_mode,
                background_color, add_metadata_checkbox, use_local_checkbox, offline_mode_checkbox,
                # Batch Tab
                resolution_preset_batch, resolution_custom_batch, weights_file_batch, precision_mode_batch,
                background_color_batch, add_metadata_checkbox_batch, use_local_checkbox_batch, offline_mode_checkbox_batch,
                # Batch-specific
                batch_folder, output_folder_batch, display_images_checkbox,
                preset_status
            ]
        )
        
        refresh_presets_btn.click(
            refresh_presets_handler,
            inputs=[],
            outputs=[preset_dropdown]
        )
        
        delete_preset_btn.click(
            delete_preset_handler,
            inputs=[preset_dropdown],
            outputs=[preset_status, preset_dropdown]
        )

        # --- Single Image Processing ---
        submit_button.click(
            predict_single,
            inputs=[
                input_image,
                resolution_preset,
                resolution_custom,
                weights_file,
                use_local_checkbox,
                precision_mode,
                add_metadata_checkbox,
                offline_mode_checkbox,
                background_color
            ],
            outputs=[single_output_text, output_image]
        )

        # --- Batch Processing ---
        batch_button.click(
            predict_batch,
            inputs=[
                resolution_preset_batch,
                resolution_custom_batch,
                weights_file_batch,
                use_local_checkbox_batch,
                precision_mode_batch,
                batch_folder,
                output_folder_batch,
                display_images_checkbox,
                add_metadata_checkbox_batch,
                offline_mode_checkbox_batch,
                background_color_batch
            ],
            outputs=[batch_output_text, output_image_batch]
        )
        
        stop_button.click(
            stop_batch_processing,
            inputs=[],
            outputs=[batch_output_text]
        )

    return demo

if __name__ == "__main__":
    args = parse_args()
    
    # Handle offline mode from command line
    if args.offline:
        set_offline_mode(True)
        print("\n" + "="*60)
        print("OFFLINE MODE ENABLED via --offline flag")
        print("No internet connections will be made.")
        print("Make sure all required models are already cached!")
        print("="*60 + "\n")
    
    allowed_paths = args.allowed_paths.split(',') if args.allowed_paths else []  # Parse allowed paths
    demo = create_interface()
    demo.launch(inbrowser=True, share=args.share, allowed_paths=allowed_paths)  # Pass allowed paths to launch