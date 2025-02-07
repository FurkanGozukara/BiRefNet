
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

# --- Assume models are in a 'models' subdirectory ---
try:
    from models.birefnet import BiRefNet  # Try local import
    LOCAL_MODEL_AVAILABLE = True
except ImportError:
    BiRefNet = None
    LOCAL_MODEL_AVAILABLE = False
    print("Warning: Local BiRefNet class not found.  Will attempt to load all models from Hugging Face.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run the image segmentation app")
    parser.add_argument("--share", action="store_true", help="Enable sharing of the Gradio interface")
    parser.add_argument("--allowed_paths", type=str, default="", help="Comma-separated list of additional allowed paths") # Added argument
    return parser.parse_args()

torch.set_float32_matmul_precision('high')
os.environ['HOME'] = os.path.expanduser('~')
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Global variable to control batch processing ---
batch_processing_stop_event = threading.Event()  # Use an event for thread-safe stopping

def open_folder():
    open_folder_path = os.path.abspath("results")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')

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

def load_model(weights_file: str, use_local: bool = False, use_half_precision: bool = True):
    model_path = os.path.join("models", f"{weights_file}.pth")
    huggingface_path = f'ZhengPeng7/{weights_file}'

    if use_local and LOCAL_MODEL_AVAILABLE and os.path.exists(model_path):
        try:
            birefnet = BiRefNet()
            birefnet.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from local path: {model_path}")
        except Exception as e:
            print(f"Error loading from local path {model_path}: {e}.  Falling back to Hugging Face.")
            use_local = False

    if not use_local or not os.path.exists(model_path):
        try:
            if LOCAL_MODEL_AVAILABLE:
                birefnet = BiRefNet.from_pretrained(huggingface_path)
                print(f"Loaded model from Hugging Face (BiRefNet): {huggingface_path}")
            else:
                from transformers import AutoModelForImageSegmentation
                birefnet = AutoModelForImageSegmentation.from_pretrained(huggingface_path, trust_remote_code=True)
                print(f"Loaded model from Hugging Face (AutoModel): {huggingface_path}")

        except Exception as e:
            print(f"Error loading from Hugging Face {huggingface_path}: {e}. Make sure model exists.")
            return None

    if birefnet is not None:
        birefnet.to(device)
        if device == 'cuda' and use_half_precision:
            birefnet.half()
        birefnet.eval()
    return birefnet

# Load default model
birefnet = load_model("BiRefNet_HR", use_half_precision=True)

def extract_object(birefnet, imagepath, image_size=(2048, 2048), use_half_precision=True):
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(imagepath)
    if image is None:
        raise ValueError(f"Could not open image at {imagepath}")

    image = image.convert("RGB")
    original_size = image.size
    input_tensor = transform_image(image)

    if input_tensor.shape[0] == 4:
        input_tensor = input_tensor[:3, :, :]

    input_images = input_tensor.unsqueeze(0)
    if device == 'cuda' and use_half_precision:
        input_images = input_images.to(device).half()
    else:
         input_images = input_images.to(device)

    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(original_size)
    image = image.convert("RGBA")
    image.putalpha(mask)

    return image, mask

def process_single_image(image_path: str, resolution: str, output_folder: str, use_half_precision:bool) -> Tuple[str, str, float]:
    start_time = time.time()

    if resolution == '':
        with Image.open(image_path) as img:
            resolution = f"{img.width}x{img.height}"
    resolution = [int(int(reso)//32*32) for reso in resolution.strip().split('x')]

    try:
        output_image, _ = extract_object(birefnet, image_path, image_size=tuple(resolution), use_half_precision=use_half_precision)
    except ValueError as e:
        return str(e), "", 0.0

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{base_filename}.png")

    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(output_folder, f"{base_filename}_{counter:04d}.png")
        counter += 1

    output_image.save(output_path)
    processing_time = time.time() - start_time
    return image_path, output_path, processing_time

def predict_single(image: str, resolution: str, weights_file: str, use_local: bool, use_half_precision: bool) -> Tuple[str, List[Tuple[str, str]]]:
    global birefnet
    model_name = usage_to_weights_file.get(weights_file, 'BiRefNet_HR')
    birefnet = load_model(model_name, use_local, use_half_precision)
    if birefnet is None:
        return "Error: Model loading failed.", []
    output_folder="results"
    os.makedirs(output_folder, exist_ok=True)
    input_path, output_path, proc_time = process_single_image(image, resolution, output_folder, use_half_precision)
    if output_path:
        return "Single image processing complete.", [(output_path, f"{proc_time:.4f} seconds")]
    else:
        return input_path, []


def predict_batch(resolution: str, weights_file: str, use_local: bool, use_half_precision: bool, batch_folder: str, output_folder: str, display_images: bool):
    global birefnet
    batch_processing_stop_event.clear()  # Reset the stop event at the start
    model_name = usage_to_weights_file.get(weights_file, 'BiRefNet_HR')
    birefnet = load_model(model_name, use_local, use_half_precision)
    if birefnet is None:
        return "Error: Model loading failed.", []

    os.makedirs(output_folder, exist_ok=True)
    results = []
    image_files = glob(os.path.join(batch_folder, '*'))
    total_images = len(image_files)
    processed_images = 0
    start_time = time.time()

    for img_path in image_files:
        if batch_processing_stop_event.is_set():  # Check if the stop event is set
            break
        try:
            input_path, output_path, proc_time = process_single_image(img_path, resolution, output_folder, use_half_precision)
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
        gr.Markdown("## SECourses Improved BiRefNet HQ V5 - Source : https://www.patreon.com/posts/121679760")

        with gr.Tab("Single Image Processing"):
            with gr.Row():
                with gr.Column():  # Use columns for better layout control
                    input_image = gr.Image(type="filepath", label="Input Image", height=512)
                    submit_button = gr.Button("Process Single Image")
                    weights_file = gr.Dropdown(choices=list(usage_to_weights_file.keys()), value="BiRefNet_HR (High Resolution 2048x2048)", label="Weights File")                    
                    single_output_text = gr.Textbox(label="Processing Status")
                    resolution = gr.Textbox(label="Resolution", placeholder="2048x2048 (or original)", value="2048x2048")                                                           

                with gr.Column():
                     output_image = gr.Gallery(label="Output Image", elem_id="gallery", height=512)
                     btn_open_outputs = gr.Button("Open Results Folder")
                     btn_open_outputs.click(fn=open_folder)                     
                     use_half_precision_checkbox = gr.Checkbox(label="Use CUDA Half Precision", value=True, interactive=True) 
                     use_local_checkbox = gr.Checkbox(label="Use Local Model", value=False)


        with gr.Tab("Batch Processing"):
            with gr.Row():
               with gr.Column():
                    batch_folder = gr.Textbox(label="Batch Folder Path")
                    resolution_batch = gr.Textbox(label="Resolution", placeholder="2048x2048 (or original)", value="2048x2048")  #Resolution for batch
                    weights_file_batch = gr.Dropdown(choices=list(usage_to_weights_file.keys()), value="BiRefNet_HR (High Resolution 2048x2048)", label="Weights File") # Weight file for batch
                    use_local_checkbox_batch = gr.Checkbox(label="Use Local Model", value=False) # Local model for batch
                    use_half_precision_checkbox_batch = gr.Checkbox(label="Use CUDA Half Precision", value=True, interactive=True) # Half precision for batch
                    output_folder_batch = gr.Textbox(label="Output Folder Path", value="results")  # Separate output folder for batch
                    display_images_checkbox = gr.Checkbox(label="Display Images During Batch Processing", value=True)  # Checkbox for displaying images


               with gr.Column():
                    output_image_batch = gr.Gallery(label="Output Image", elem_id="gallery_batch") # Batch output
            batch_button = gr.Button("Start Batch Processing")
            stop_button = gr.Button("Stop Batch Processing")
            batch_output_text = gr.Textbox(label="Batch Processing Status")



        # --- Single Image Processing ---
        submit_button.click(
            predict_single,
            inputs=[input_image, resolution, weights_file, use_local_checkbox, use_half_precision_checkbox],
            outputs=[single_output_text, output_image]
        )

        # --- Batch Processing ---
        batch_button.click(
            predict_batch,
            inputs=[resolution_batch, weights_file_batch, use_local_checkbox_batch, use_half_precision_checkbox_batch, batch_folder, output_folder_batch, display_images_checkbox],
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
    allowed_paths = args.allowed_paths.split(',') if args.allowed_paths else []  # Parse allowed paths
    demo = create_interface()
    demo.launch(inbrowser=True, share=args.share, allowed_paths=allowed_paths) # Pass allowed paths to launch