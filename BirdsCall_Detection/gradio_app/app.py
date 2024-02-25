import gradio as gr
import os 
import torch
import numpy as np 
import matplotlib.cm as cm
from PIL import Image
from utils import load_model, compute_timestamps 
from audio_sed.sed_config import ConfigSED
import librosa 
import pandas as pd 
import json
import librosa
import matplotlib.pyplot as plt
import librosa.display
import cv2
plt.rcParams['figure.figsize'] = [10, 4]
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
def load_json(filename):
    mapping = json.load(open(filename, 'r'))
    news = {}
    for k, v in mapping.items():
        news[v] = k 
    return news

def extract_spectrogram(model, input):
    with torch.no_grad():
        x =  model.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x =  model.logmel_extractor(x) # (batch_size, 1, time_steps, mel_bins)
    spectrogram_mel = x.cpu().numpy().squeeze(0).squeeze(0).T
    
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spectrogram_mel, x_axis='time',
                         y_axis='mel', sr=32000, cmap="viridis")
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    fig.savefig("temp/mel_spectrogram.jpg")

    t = np.array(list(range(len(input.squeeze(0)))))/32000
    fig, ax = plt.subplots()
    plt.plot(t, input.cpu().numpy().squeeze(0))
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.savefig('temp/signal.jpg')
    img1 =  cv2.cvtColor(cv2.imread("temp/mel_spectrogram.jpg"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('temp/signal.jpg'), cv2.COLOR_BGR2RGB)
    return img1, img2 

infos = {"data/tf_efficientnet_b0_bird.pth":{'backbone':'tf_efficientnet_b0_ns' , 'label_code':{0: "bird"}},
        "data/AudioSED_effb2_pretrained.pth":{'backbone':'tf_efficientnet_b2_ns' , 'label_code':load_json('data/primary.json')}
        }

sample_rate = 32000
max_length = 5

checkpoint = "data/tf_efficientnet_b0_bird.pth" # "data/tf_efficientnet_b0_bird.pth"# "data/AudioSED_effb2_pretrained.pth" #
label_code = infos[checkpoint]['label_code']
num_classe = len(label_code) # bird or nocall

IMAGE_SIZE = (300, 600)


device = "cuda" if torch.cuda.is_available() else "cpu"
cfg_sed =  ConfigSED(window='hann', center=True, pad_mode='reflect', windows_size=1024, hop_size=320,
                sample_rate= sample_rate, mel_bins=128, fmin=20, fmax=16000, ref=1.0, amin=1e-10, top_db=None)



model = load_model(model_name=infos[checkpoint]['backbone'], cfg_sed=cfg_sed, max_length=max_length, sample_rate=sample_rate, num_classes=num_classe)
model.load_state_dict(torch.load(checkpoint))
model = model.to(device)
model.eval()
segmentwise_global = None 


def predict_signal(inputs, alpha):
    def normalise(x):
        min_ = x.min()
        max_ = x.max()
        if min_ == max_:
            x_norm = np.zeros(x.shape).astype(np.float32)
        else:
            x_norm =  (x-min_)/(max_ - min_)
        return x_norm

    global segmentwise_global
    rate, sample = inputs 
     
    # resample
    if rate != sample_rate:
        sample = librosa.resample(sample.astype(np.float32), rate, sample_rate)

    length = int(max_length * sample_rate)
    temp = np.zeros((length,))
    if len(sample) < max_length*sample_rate:
        temp[:len(sample)] =  signal
    else:
        temp =  sample[:length]
    signal = torch.as_tensor(normalise(temp)).float().to(device)

    # prediction
    with torch.no_grad():
        output = model(signal.unsqueeze(0))[0] 
        clipwise, segmentwise = output['clipwise'].cpu().numpy().squeeze(0), output["segmentwise"].cpu().numpy() 
    spectrogram_img, signal_img = extract_spectrogram(model, input=signal.unsqueeze(0))#.squeeze(0)


    confidences = {label_code[i]:float(clipwise[i]) for i in range(len(label_code))}

    df = compute_timestamps(segmentwise, alpha, max_length, label_code)
    segmentwise_global = segmentwise.copy()
    return confidences, df, [signal_img], [spectrogram_img]


def clear(timestamps, input_image, slide, histogram):
    return None, None, 0.5, None, None, None

def update_ts(alpha):
    global segmentwise_global 
    if segmentwise_global is None:
        return None
    df = compute_timestamps(segmentwise, alpha, max_length)
    return  df
 

with gr.Blocks() as demo:


    with gr.Row():
        with gr.Column():  
            inputs = gr.Audio() 
            slider = gr.Slider(minimum=0, maximum=1.0, value=0.5, step=0.01, interactive=True)
            with gr.Tab("Signal") :
                #heatmap = gr.Gallery(shape=IMAGE_SIZE)#.style(height=600, width=600)
                signals = gr.Gallery(height="auto", scale=3,columns=1, rows=1, allow_preview=True, preview=True,   object_fit="scale-down")#.style(height=300, width=600)
            with gr.Tab("Spectrogram") :
                spectrograms = gr.Gallery(height="auto", scale=3, columns=1 , rows=1, allow_preview=True,  preview=True,   object_fit="scale-down")#.style(height=300, width=600)
        with gr.Tab("Confidence") :
            histogram = gr.Label(num_top_classes=5,  min_width=500 )#.style(height=500, width=500)
        with gr.Tab("Table") :
            #heatmap = gr.Gallery(shape=IMAGE_SIZE)#.style(height=600, width=600)
            timestamps = gr.DataFrame(headers=['start', 'end', 'specie'])
        
    with gr.Row():
        btn1 = gr.Button("Submit")
        btn1.click(predict_signal, inputs=[inputs, slider], outputs=[histogram, timestamps, signals, spectrograms])
        btn2 = gr.Button("Clear")
        btn2.click(clear, inputs=[timestamps, inputs, slider, histogram], outputs=[timestamps, inputs, slider, histogram, signals, spectrograms])


    slider.release(update_ts, inputs=[slider], outputs=[timestamps])


    gr.Examples(
        examples=[os.path.join(os.path.dirname(__file__), "examples/HSN_001_20150708_061805_0079_nocall.wav"), 
                os.path.join(os.path.dirname(__file__),  "examples/HSN_100_20150712_091105_0118.wav"), 
                os.path.join(os.path.dirname(__file__), "examples/NES_001_S01_20190914_043000_0056_nocall.wav"),
                os.path.join(os.path.dirname(__file__), "examples/NES_002_S01_20190914_043000_0635.wav") 
                ],
        inputs=inputs,
        outputs=inputs,
        fn=lambda x: x,
        cache_examples=False,
    )
 
demo.launch()