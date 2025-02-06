import os
import cv2
import numpy as np
import torch
import gradio as gr
import argparse
from pathlib import Path
from glob import glob
from typing import Optional, Tuple, List
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import time
import os
import platform

def parse_args():
    parser = argparse.ArgumentParser(description="Run the image segmentation app")
    parser.add_argument("--share", action="store_true", help="Enable sharing of the Gradio interface")
    return parser.parse_args()

torch.set_float32_matmul_precision('high')
torch.jit.script = lambda f: f

os.environ['HOME'] = os.path.expanduser('~')

device = "cuda" if torch.cuda.is_available() else "cpu"

def open_folder():
    open_folder_path = os.path.abspath("results")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')

class ImagePreprocessor():
    def __init__(self, resolution: Tuple[int, int] = (1024, 1024)) -> None:
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def proc(self, image: Image.Image) -> torch.Tensor:
        image = image.convert('RGB')  # Convert to RGB
        image = self.transform_image(image)
        return self.normalize(image)

usage_to_weights_file = {
    'General': 'BiRefNet',
    'General-Lite': 'BiRefNet_T',
    'Portrait': 'BiRefNet-portrait',
    'DIS': 'BiRefNet-DIS5K',
    'HRSOD': 'BiRefNet-HRSOD',
    'COD': 'BiRefNet-COD',
    'DIS-TR_TEs': 'BiRefNet-DIS5K-TR_TEs'
}

birefnet = AutoModelForImageSegmentation.from_pretrained('/'.join(('zhengpeng7', usage_to_weights_file['General'])), trust_remote_code=True)
birefnet.to(device)
birefnet.eval()

def process_single_image(image_path: str, resolution: str, output_folder: str) -> Tuple[str, str, float]:
    start_time = time.time()
    
    image = Image.open(image_path).convert('RGBA')
    
    if resolution == '':
        resolution = f"{image.width}x{image.height}"
    resolution = [int(int(reso)//32*32) for reso in resolution.strip().split('x')]
    
    image_shape = image.size[::-1]
    image_pil = image.resize(tuple(resolution))

    image_preprocessor = ImagePreprocessor(resolution=tuple(resolution))
    image_proc = image_preprocessor.proc(image_pil)
    image_proc = image_proc.unsqueeze(0)

    with torch.no_grad():
        scaled_pred_tensor = birefnet(image_proc.to(device))[-1].sigmoid()

    if device == 'cuda':
        scaled_pred_tensor = scaled_pred_tensor.cpu()
    
    pred = torch.nn.functional.interpolate(scaled_pred_tensor, size=image_shape, mode='bilinear', align_corners=True).squeeze().numpy()

    pred_rgba = np.zeros((*pred.shape, 4), dtype=np.uint8)
    pred_rgba[..., :3] = (pred[..., np.newaxis] * 255).astype(np.uint8)
    pred_rgba[..., 3] = (pred * 255).astype(np.uint8)

    image_array = np.array(image)
    image_pred = image_array * (pred_rgba / 255.0)
    
    output_image = Image.fromarray(image_pred.astype(np.uint8), 'RGBA')
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{base_filename}.png")
    
    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(output_folder, f"{base_filename}_{counter:04d}.png")
        counter += 1

    output_image.save(output_path)
    
    processing_time = time.time() - start_time
    print(f"Processed {image_path} in {processing_time:.4f} seconds")  # Added this line to print processing time
    return image_path, output_path, processing_time

def predict(
    image: str,
    resolution: str,
    weights_file: Optional[str],
    batch_folder: Optional[str] = None,
    output_folder: Optional[str] = None,
    is_batch: bool = False
) -> Tuple[str, List[Tuple[str, str]]]:
    global birefnet
    _weights_file = '/'.join(('zhengpeng7', usage_to_weights_file[weights_file] if weights_file is not None else usage_to_weights_file['General']))
    print('Using weights:', _weights_file)
    birefnet = AutoModelForImageSegmentation.from_pretrained(_weights_file, trust_remote_code=True)
    birefnet.to(device)
    birefnet.eval()

    if not output_folder:
        output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)

    results = []

    if is_batch and batch_folder:
        image_files = glob(os.path.join(batch_folder, '*'))
        total_images = len(image_files)
        processed_images = 0
        start_time = time.time()

        for img_path in image_files:
            try:
                input_path, output_path, proc_time = process_single_image(img_path, resolution, output_folder)
                results.append((output_path, f"{proc_time:.4f} seconds"))
                processed_images += 1
                elapsed_time = time.time() - start_time
                avg_time_per_image = elapsed_time / processed_images
                estimated_time_left = avg_time_per_image * (total_images - processed_images)

                status = f"Processed {processed_images}/{total_images} images. Estimated time left: {estimated_time_left:.2f} seconds"
                print(status)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue

        return f"Batch processing complete. Processed {processed_images}/{total_images} images.", results
    else:
        input_path, output_path, proc_time = process_single_image(image, resolution, output_folder)
        results.append((output_path, f"{proc_time:.4f} seconds"))
        return "Single image processing complete.", results

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## SECourses Improved BiRefNet V3 'Bilateral Reference for High-Resolution Dichotomous Image Segmentation' APP - SOTA Background Remover")
        gr.Markdown("## Most Advanced Latest Version On : https://www.patreon.com/posts/109913645")
        
        with gr.Row():
            input_image = gr.Image(type="filepath", label="Input Image",height=512)
            output_image = gr.Gallery(label="Output Image", elem_id="gallery",height=512)


        with gr.Row():
            resolution = gr.Textbox(label="Resolution", placeholder="1024x1024 - Optional - Don't enter to use original image resolution - Higher res uses more VRAM but still works perfect with shared VRAM so fast")
            weights_file = gr.Dropdown(choices=list(usage_to_weights_file.keys()), value="General", label="Weights File")
            btn_open_outputs = gr.Button("Open Results Folder")
            btn_open_outputs.click(fn=open_folder)

        with gr.Row():
            batch_folder = gr.Textbox(label="Batch Folder Path")
            output_folder = gr.Textbox(label="Output Folder Path", value="results")

        with gr.Row():
            submit_button = gr.Button("Single Image Process")
            batch_button = gr.Button("Batch Process Images in Given Folder")

        output_text = gr.Textbox(label="Processing Status")

        submit_button.click(
            predict,
            inputs=[input_image, resolution, weights_file, batch_folder, output_folder, gr.Checkbox(value=False, visible=False)],
            outputs=[output_text, output_image]
        )

        batch_button.click(
            predict,
            inputs=[input_image, resolution, weights_file, batch_folder, output_folder, gr.Checkbox(value=True, visible=False)],
            outputs=[output_text, output_image]
        )

    return demo

if __name__ == "__main__":
    args = parse_args()
    demo = create_interface()
    demo.launch(inbrowser=True, share=args.share)