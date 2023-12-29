import gradio as gr
import os 
from cassava_model import compile_new_model_convnext, make_gradcam_heatmap
import tensorflow as tf 
from tensorflow.keras import mixed_precision
import cv2 
import numpy as np 
import matplotlib.cm as cm
from PIL import Image
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

image_global = None
heatmap_global = None 
IMAGE_SIZE = (600, 600) # (800, 800)
    
label_code = {0: "Cassava Bacterial Blight (CBB)", 1:"Cassava Brown Streak Disease (CBSD)", 2:  "Cassava Green Mottle (CGM)",
                3: "Cassava Mosaic Disease (CMD)", 4: "Healthy"}

mixed_precision.set_global_policy('mixed_float16')

model = compile_new_model_convnext("convnext_large_in22k", IMAGE_SIZE, num_class=5)

model.load_weights("convnext_large-mixup-cutmix-convnext_large-fold-0.h5")
# model.load_weights("convnext_xlarge_in22k-cutmix-noext-800-convnext_xlarge_in22k-fold-0.h5")
model.summary()


def predict_image(inputs, alpha):
    global heatmap_global 
    global image_global 
    # resize image
    #print(inputs.shape)
    
    input_resized = cv2.resize(inputs, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    input_resized = np.expand_dims(input_resized, axis=0)
    #print(input_resized.shape)
    output, heatmap = make_gradcam_heatmap(img_array=input_resized, model=model) #model.predict(input_resized).flatten().astype(np.float32)
    #print(output)
    confidences = {label_code[i]:float(output[i]) for i in range(len(label_code))}
    #print(confidences)
    #print(alpha)
    image_global = input_resized.squeeze(0).copy()
    heatmap_global = heatmap.copy()
    #print(type(heatmap), type(input_resized))
    superimposed_img =  (heatmap * alpha  +  input_resized.squeeze(0))
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    return confidences, [superimposed_img]

def clear(heatmap, input_image, slide, histogram):
    global heatmap_global 
    global image_global 
    image_global = None 
    heatmap_global = None
    return None, None, 0.5, None

def update_heatmap(alpha):
    global heatmap_global 
    global image_global 
    if heatmap_global is None or image_global is None:
        return None
    superimposed_img =  (heatmap_global * alpha  +  image_global)
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    return  [superimposed_img]
 

with gr.Blocks() as demo:


    with gr.Row():
        with gr.Column():  
            inputs = gr.Image(shape=IMAGE_SIZE ).style(height=600, width=600)
            slider = gr.Slider(minimum=0, maximum=1.0, value=0.5, step=0.01, interactivate=True)
            
        with gr.Tab("Confidence") :
            histogram = gr.Label(num_top_classes=5,  min_width=500 )#.style(height=500, width=500)
        with gr.Tab("Heatmap") :
            heatmap = gr.Gallery(shape=IMAGE_SIZE)#.style(height=600, width=600)
    
        
    with gr.Row():
        btn1 = gr.Button("Submit")
        btn1.click(predict_image, inputs=[inputs, slider], outputs=[histogram, heatmap])
        btn2 = gr.Button("Clear")
        btn2.click(clear, inputs=[heatmap, inputs, slider, histogram], outputs=[heatmap, inputs, slider, histogram])


    slider.release(update_heatmap, inputs=[slider], outputs=[heatmap])


    gr.Examples(
        examples=[os.path.join(os.path.dirname(__file__), "examples/1000015157.jpg"), 
                os.path.join(os.path.dirname(__file__),  "examples/100560400.jpg"), 
                os.path.join(os.path.dirname(__file__), "examples/1000201771.jpg")],
        inputs=inputs,
        outputs=inputs,
        fn=lambda x: x,
        cache_examples=True,
    )
 
demo.launch()