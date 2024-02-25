import timm 
from audio_sed.pytorch.sed_models import AudioSED, shape_from_backbone
from torch import nn 
import numpy as np 
import torch 
import pandas as pd 
import matplotlib.pyplot as plt

def load_model(model_name, cfg_sed, max_length, sample_rate, num_classes):

    backbone = timm.create_model(model_name, pretrained=False)
    if "efficientnet" in model_name:
        backbone.global_pool =  nn.Identity()
        in_feat = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
    elif "convnext" in  model_name:
        in_feat = backbone.head.fc.in_features
        backbone.head = nn.Identity()

    in_features = shape_from_backbone(inputs=torch.as_tensor(np.random.uniform(0, 1, (1, int(max_length * sample_rate)))).float(), model=backbone, use_logmel=True, config_sed = cfg_sed.__dict__)[2]
    print("Num timestamps features:",in_features)
    model = AudioSED(backbone, num_classes=[num_classes], in_features=in_feat, hidden_size=1024, activation= 'sigmoid', use_logmel=True, 
                spectrogram_augmentation = None, apply_attention="step", drop_rate = [0.5, 0.5], config_sed= cfg_sed.__dict__)


    return model 

def compute_timestamps(segmentwise, alpha, max_length, label_code):
    # compute timestamps for bird/nocall
    num_steps = segmentwise.shape[1]
    step_time = max_length/num_steps
    isPos = segmentwise.squeeze(0) >= alpha
    starts = [] 
    ends = []
    species = []
    st, end = None, None
    for cls_ in range(len(label_code)):
        st = None 
        end = None 
        for i, result in enumerate(isPos[:, cls_]):
            if result:
                if st is None:
                    st = (i * step_time)
            else:
                if st is not None:
                    end = (i * step_time)
                    starts.append(st)
                    ends.append(end)
                    species.append(label_code[cls_])
                    st = None 
                    end = None 
        if st is not None:
            starts.append(st)
            ends.append(max_length)
            species.append(label_code[cls_])
    df = pd.DataFrame({'start': starts, 'end':ends, "specie":species})
    return df 


if __name__ == "__main__":
    from audio_sed.sed_config import ConfigSED
    cfg_sed =  ConfigSED(window='hann', center=True, pad_mode='reflect', windows_size=1024, hop_size=320,
                sample_rate=32000, mel_bins=128, fmin=20, fmax=16000, ref=1.0, amin=1e-10, top_db=None)

    checkpoint = "tf_efficientnet_b0_bird.pth"
    model = load_model(model_name="tf_efficientnet_b0_ns", cfg_sed=cfg_sed, max_length=5, sample_rate=32000, num_classes=1)
    model.load_state_dict(torch.load(checkpoint))
    print(model)
