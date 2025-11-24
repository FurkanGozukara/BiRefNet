
import os
import torch
import gradio as gr
import argparse
from typing import Optional, Tuple, List
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

def extract_object(birefnet, imagepath, image_size=(2048, 2048), precision_mode='bf16'):
    """
    Extract object with mixed precision support and optional resizing.
    
    Args:
        birefnet: Model instance
        imagepath: Path to input image
        image_size: Target size (W, H) or None for original resolution
        precision_mode: 'no', 'fp16', or 'bf16'
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
    image = image.convert("RGBA")
    image.putalpha(mask)

    return image, mask

def process_single_image(image_path: str, resolution_preset: str, resolution_custom: str, 
                         output_folder: str, model_name: str, precision_mode: str = 'bf16',
                         add_metadata: bool = True) -> Tuple[str, str, float]:
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
    """
    start_time = time.time()

    # Parse resolution using new system
    image_size = parse_resolution(resolution_preset, resolution_custom, image_path)

    try:
        output_image, _ = extract_object(birefnet, image_path, image_size=image_size, precision_mode=precision_mode)
    except ValueError as e:
        return str(e), "", 0.0

    # Generate output path with optional metadata
    output_path = generate_output_path(output_folder, image_path, image_size, model_name, add_metadata)

    output_image.save(output_path)
    processing_time = time.time() - start_time
    return image_path, output_path, processing_time

def predict_single(image: str, resolution_preset: str, resolution_custom: str, weights_file: str, 
                  use_local: bool, precision_mode: str, add_metadata: bool, 
                  offline_mode: bool = False) -> Tuple[str, List[Tuple[str, str]]]:
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
    """
    global birefnet
    
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
    
    input_path, output_path, proc_time = process_single_image(
        image, resolution_preset, resolution_custom, output_folder, 
        model_name, precision_internal, add_metadata
    )
    
    if output_path:
        return "Single image processing complete.", [(output_path, f"{proc_time:.4f} seconds")]
    else:
        return input_path, []


def predict_batch(resolution_preset: str, resolution_custom: str, weights_file: str, use_local: bool, 
                 precision_mode: str, batch_folder: str, output_folder: str, 
                 display_images: bool, add_metadata: bool, offline_mode: bool = False):
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
    """
    global birefnet
    
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

    for img_path in image_files:
        if batch_processing_stop_event.is_set():  # Check if the stop event is set
            break
        try:
            input_path, output_path, proc_time = process_single_image(
                img_path, resolution_preset, resolution_custom, output_folder, 
                model_name, precision_internal, add_metadata
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
    with gr.Blocks() as demo:
        gr.Markdown("## SECourses Improved BiRefNet HQ V9 - https://www.patreon.com/posts/121679760")
        gr.Markdown("**New Features:** BFloat16 precision (default), Original resolution mode, Enhanced output organization")

        with gr.Tab("Single Image Processing"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="filepath", label="Input Image", height=512)
                    
                    with gr.Row():
                        resolution_preset = gr.Dropdown(
                            choices=list(RESOLUTION_PRESETS.keys()),
                            value='2048x2048',
                            label="Resolution Preset",
                            info="Select resolution or use original size"
                        )
                    
                    resolution_custom = gr.Textbox(
                        label="Custom Resolution (WIDTHxHEIGHT)",
                        placeholder="3072x3072",
                        value="2048x2048",
                        visible=False,
                        info="Only used when 'Custom' preset is selected"
                    )
                    
                    weights_file = gr.Dropdown(
                        choices=list(usage_to_weights_file.keys()),
                        value="BiRefNet_HR (High Resolution 2048x2048)",
                        label="Model Selection"
                    )
                    
                    precision_mode = gr.Dropdown(
                        choices=list(MIXED_PRECISION_OPTIONS.keys()),
                        value='BFloat16 (BF16 - Recommended)',
                        label="Mixed Precision Mode",
                        info="BF16: Best balance of speed/quality. FP16: Faster but may have artifacts. FP32: Highest quality, slowest."
                    )
                    
                    add_metadata_checkbox = gr.Checkbox(
                        label="Organize Output by Model/Resolution",
                        value=True,
                        info="Creates subfolders like 'BiRefNet_HR-reso_2048x2048' for better organization"
                    )
                    
                    use_local_checkbox = gr.Checkbox(
                        label="Use Local Model (.pth file)",
                        value=False,
                        info="Load from local .pth file in models/ folder instead of HuggingFace"
                    )
                    
                    offline_mode_checkbox = gr.Checkbox(
                        label="🔌 Offline Mode (100% No Internet)",
                        value=False,
                        info="Use ONLY cached/local models. No internet connections will be made. Models must be pre-downloaded."
                    )
                    
                    submit_button = gr.Button("Process Single Image", variant="primary")
                    single_output_text = gr.Textbox(label="Processing Status")

                with gr.Column():
                    output_image = gr.Gallery(label="Output Image", elem_id="gallery", height=512)
                    btn_open_outputs = gr.Button("📁 Open Results Folder")
                    btn_open_outputs.click(fn=open_folder)

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
                        placeholder="Path to folder containing images"
                    )
                    
                    with gr.Row():
                        resolution_preset_batch = gr.Dropdown(
                            choices=list(RESOLUTION_PRESETS.keys()),
                            value='2048x2048',
                            label="Resolution Preset"
                        )
                    
                    resolution_custom_batch = gr.Textbox(
                        label="Custom Resolution (WIDTHxHEIGHT)",
                        placeholder="3072x3072",
                        value="2048x2048",
                        visible=False
                    )
                    
                    weights_file_batch = gr.Dropdown(
                        choices=list(usage_to_weights_file.keys()),
                        value="BiRefNet_HR (High Resolution 2048x2048)",
                        label="Model Selection"
                    )
                    
                    precision_mode_batch = gr.Dropdown(
                        choices=list(MIXED_PRECISION_OPTIONS.keys()),
                        value='BFloat16 (BF16 - Recommended)',
                        label="Mixed Precision Mode"
                    )
                    
                    output_folder_batch = gr.Textbox(
                        label="Output Folder Path",
                        value="results"
                    )
                    
                    add_metadata_checkbox_batch = gr.Checkbox(
                        label="Organize Output by Model/Resolution",
                        value=True
                    )
                    
                    use_local_checkbox_batch = gr.Checkbox(
                        label="Use Local Model (.pth file)",
                        value=False,
                        info="Load from local .pth file in models/ folder"
                    )
                    
                    offline_mode_checkbox_batch = gr.Checkbox(
                        label="🔌 Offline Mode (100% No Internet)",
                        value=False,
                        info="Use ONLY cached/local models. No internet connections will be made."
                    )
                    
                    display_images_checkbox = gr.Checkbox(
                        label="Display Images During Processing",
                        value=False,
                        info="Disable for faster processing of large batches"
                    )
                    
                    with gr.Row():
                        batch_button = gr.Button("Start Batch Processing", variant="primary")
                        stop_button = gr.Button("Stop Batch Processing", variant="stop")
                    
                    batch_output_text = gr.Textbox(label="Batch Processing Status")

                with gr.Column():
                    output_image_batch = gr.Gallery(label="Output Images", elem_id="gallery_batch")

            # Show/hide custom resolution based on preset selection
            resolution_preset_batch.change(
                update_custom_visibility,
                inputs=[resolution_preset_batch],
                outputs=[resolution_custom_batch]
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
                offline_mode_checkbox
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
                offline_mode_checkbox_batch
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