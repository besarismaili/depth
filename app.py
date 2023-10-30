import gradio as gr
#from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import os
import cv2
from run import run
from midas.model_loader import default_models, load_model
############################################
def info_fun(input_file, output_path, model_path, model_type, optimize, side, height, square, grayscale):
    
    input_file = "input_file: "+str(input_file)
    model_type = ",  model_type: "+str(model_type)
    output_path = ",  output_path: "+str(output_path)
    model_path = ",  model_path: "+str(model_path)
    optimize = ",  optimize: "+str(optimize)
    side = ",  side: "+str(side)
    height = ",  height: "+str(height)
    square = ",  square: "+str(square)
    grayscale = ",  grayscale: "+str(grayscale)

    return input_file, output_path, model_path, model_type, optimize, side, height, square, grayscale

def weights(model_type):
    path="weights/"+model_type+".pt"

    return path
############################################

title = "# Depth estimation demo"
description = "Demo for Intel's DPT"

with gr.Blocks() as demo:

    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():

        with gr.Column():
            with gr.Tab(label='Singel image'):
                model_type = gr.Dropdown(["dpt_beit_large_512", "dpt_beit_large_384", "dpt_beit_base_384", "dpt_swin2_large_384", "dpt_swin2_base_384", "dpt_swin2_tiny_256", "dpt_swin_large_384", "dpt_next_vit_large_384", "dpt_levit_224", "dpt_large_384", "dpt_hybrid_384", "midas_v21_384", "midas_v21_small_256" , "openvino_midas_v21_small_256"], label="Model", value="dpt_beit_large_512", info="Choose the model to generate the depth")
                with gr.Accordion(label="More Settings",open=False):
                    output_path = gr.Text("output", label="Output path", visible=True)
                    model_path = gr.Text(value="weights/"+model_type.value+".pt", label="Model path", visible=True)

                    optimize = gr.Checkbox(label="Optimize", info="Use half-float optimization",value=False)
                    side = gr.Checkbox(label="Side", info="Output images contain RGB and depth images side by side")
                    height = gr.Radio([None, 32, 64, 128, 256, 512], label="Height", default=None)
                    square = gr.Checkbox(label="Square", info="Option to resize images to a square resolution by changing their widths when images are fed into the encoder during inference. If this parameter is not set, the aspect ratio of images is tried to be preserved if supported by the model.")
                    grayscale = gr.Checkbox(label="Grayscale", info="Use a grayscale colormap instead of the inferno one. Although the inferno colormap, 'which is used by default, is better for visibility, it does not allow storing 16-bit depth values in PNGs but only 8-bit ones due tothe precision limitation of this colormap.")
                #input_file = gr.File(type="file")
                input_file = gr.Text("input")
                
                get_depth_button = gr.Button(value="Get Depth",interactive=True, variant="primary")
                get_info_button = gr.Button(value="Get info",interactive=True)
                
        with gr.Column():
            with gr.Tab(label='Frames'):
                #output=gr.Video(label="Predicted Depth")
                message=gr.Text(value="Check output folder for the depth frames.",visible=False)
                image_output=gr.Image(type="filepath", label="Predicted Depth")
                info_output = gr.Textbox()
    model_type.change(fn=weights, inputs=[model_type], outputs=[model_path])    
    get_depth_button.click(fn=run, inputs=[input_file, output_path, model_path, model_type, optimize, side, height, square, grayscale], outputs=[image_output])
    get_info_button.click(fn=info_fun, inputs=[input_file, output_path, model_path, model_type, optimize, side, height, square, grayscale], outputs=[info_output])
if __name__ == "__main__":

    demo.launch()