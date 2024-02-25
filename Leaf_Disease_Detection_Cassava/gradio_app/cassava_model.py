from tfimm.architectures.convnext import ConvNeXtConfig, ConvNeXtStage, _weight_initializers
from collections import OrderedDict
from typing import List, Tuple
from tfimm.layers import MLP, ConvMLP, DropPath, norm_layer_factory
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from copy import deepcopy
from typing import Callable, List, Optional
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.python.keras import backend as K

from tfimm.models.registry import is_model, model_class, model_config
from tfimm.utils import cached_model_path, load_pth_url_weights, load_timm_weights
import logging
import re
import numpy as np 
from tfimm.models.factory import transfer_weights
import matplotlib.cm as cm
from PIL import Image

class ConvNeXt(tf.keras.Model):
    """
    Class implementing a ConvNeXt network.
    Paper: `A ConvNet for the 2020s <https://arxiv.org/pdf/2201.03545.pdf>`_.
    Parameters:
        cfg: Configuration class for the model.
        **kwargs: Arguments are passed to ``tf.keras.Model``.
    """

    cfg_class = ConvNeXtConfig

    def __init__(self, cfg: ConvNeXtConfig, **kwargs):
        kwargs["name"] = kwargs.get("name", cfg.name)
        super().__init__(**kwargs)
        self.cfg = cfg
        norm_layer = norm_layer_factory(cfg.norm_layer)
        kernel_initializer, bias_initializer = _weight_initializers()

        self.stem_conv = tf.keras.layers.Conv2D(
            filters=cfg.embed_dim[0],
            kernel_size=cfg.patch_size,
            strides=cfg.patch_size,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="stem/0",
        )
        self.stem_norm = norm_layer(name="stem/1")

        # Stochastic depth
        dpr = np.linspace(0.0, cfg.drop_path_rate, sum(cfg.nb_blocks))
        dpr = np.split(dpr, np.cumsum(cfg.nb_blocks))

        self.stages = []
        nb_stages = len(cfg.nb_blocks)
        for j in range(nb_stages):
            self.stages.append(
                ConvNeXtStage(
                    stride=2 if j > 0 else 1,
                    embed_dim=cfg.embed_dim[j],
                    nb_blocks=cfg.nb_blocks[j],
                    mlp_ratio=cfg.mlp_ratio,
                    conv_mlp_block=cfg.conv_mlp_block,
                    drop_rate=cfg.drop_rate,
                    drop_path_rate=dpr[j],
                    norm_layer=cfg.norm_layer,
                    act_layer=cfg.act_layer,
                    init_scale=cfg.init_scale,
                    name=f"stages/{j}",
                )
            )


    @property
    def dummy_inputs(self) -> tf.Tensor:
        """Returns a tensor of the correct shape for inference."""
        return tf.zeros((1, *self.cfg.input_size, self.cfg.in_channels))

    @property
    def feature_names(self) -> List[str]:
        """
        Names of features, returned when calling ``call`` with ``return_features=True``.
        """
        _, features = self(self.dummy_inputs, return_features=True)
        return list(features.keys())

    def forward_features(
        self, x, training: bool = False, return_features: bool = False
    ):
        """
        Forward pass through model, excluding the classifier layer. This function is
        useful if the model is used as input for downstream tasks such as object
        detection.
        Arguments:
             x: Input to model
             training: Training or inference phase?
             return_features: If ``True``, we return not only the model output, but a
                dictionary with intermediate features.
        Returns:
            If ``return_features=True``, we return a tuple ``(y, features)``, where
            ``y`` is the model output and ``features`` is a dictionary with
            intermediate features.
            If ``return_features=False``, we return only ``y``.
        """
        features = OrderedDict()
        x = self.stem_conv(x)
        x = self.stem_norm(x, training=training)
        features["stem"] = x

        for stage_idx, stage in enumerate(self.stages):
            x = stage(x, training=training, return_features=return_features)
            if return_features:
                x, stage_features = x
                for key, val in stage_features.items():
                    features[f"stage_{stage_idx}/{key}"] = val
        features["conv_features"] = x

        return (x, features) if return_features else x

    def call(self, x, training: bool = False, return_features: bool = False):
        """
        Forward pass through the full model.
        Arguments:
             x: Input to model
             training: Training or inference phase?
             return_features: If ``True``, we return not only the model output, but a
                dictionary with intermediate features.
        Returns:
            If ``return_features=True``, we return a tuple ``(y, features)``, where
            ``y`` is the model output and ``features`` is a dictionary with
            intermediate features.
            If ``return_features=False``, we return only ``y``.
        """
        features = OrderedDict()
        x = self.forward_features(x, training, return_features)
        if return_features:
            x, features = x
        return (x, features) if return_features else x


def create_model(
    model_name: str,
    pretrained: bool = False,
    model_path: str = "",
    *,
    in_channels: Optional[int] = None,
    nb_classes: Optional[int] = None,
    **kwargs,) -> tf.keras.Model:
    """
    Creates a model.
    Args:
        model_name: Name of model to instantiate
        pretrained: If ``True``, load pretrained weights as specified by the ``url``
            field in config. We will check the cache first and download weights only
            if they cannot be found in the cache.
            If ``url`` is ``[timm]``, the weights will be downloaded from ``timm`` and
            converted to TensorFlow. Requires ``timm`` and ``torch`` to be installed.
            If ``url`` starts with ``[pytorch]``, the weights are in PyTorch format
            and ``torch`` needs to be installed to convert them.
        model_path: Path of model weights to load after model is initialized. This takes
            over ``pretrained``.
        in_channels: Number of input channels for model. If ``None``, use default
            provided by model.
        nb_classes: Number of classes for classifier. If set to 0, no classifier is
            used and last layer is pooling layer. If ``None``, use default provided by
            model.
        **kwargs: Other kwargs are model specific.
    Returns:
        The created model.
    """
    if not is_model(model_name):
        raise RuntimeError(f"Unknown model {model_name}.")

    cls = model_class(model_name) if "convnext" not in model_name else ConvNeXt
    cfg = model_config(model_name)
    convnext_register = {
          'convnext_tiny_in22ft1k': 'convnext_tiny.fb_in22k_ft_in1k',
          'convnext_small_in22ft1k': 'convnext_small.fb_in22k_ft_in1k',
          'convnext_base_in22ft1k': 'convnext_base.fb_in22k_ft_in1k',
          'convnext_large_in22ft1k': 'convnext_large.fb_in22k_ft_in1k',
          'convnext_xlarge_in22ft1k': 'convnext_xlarge.fb_in22k_ft_in1k',
          'convnext_tiny_384_in22ft1k': 'convnext_tiny.fb_in22k_ft_in1k_384',
          'convnext_small_384_in22ft1k': 'convnext_small.fb_in22k_ft_in1k_384',
          'convnext_base_384_in22ft1k': 'convnext_base.fb_in22k_ft_in1k_384',
          'convnext_large_384_in22ft1k': 'convnext_large.fb_in22k_ft_in1k_384',
          'convnext_xlarge_384_in22ft1k': 'convnext_xlarge.fb_in22k_ft_in1k_384',
          'convnext_tiny_in22k': 'convnext_tiny.fb_in22k',
          'convnext_small_in22k': 'convnext_small.fb_in22k',
          'convnext_base_in22k': 'convnext_base.fb_in22k',
          'convnext_large_in22k': 'convnext_large.fb_in22k',
          'convnext_xlarge_in22k': 'convnext_xlarge.fb_in22k',
      }
    if model_path:
        loaded_model = tf.keras.models.load_model(model_path, compile=False)
    elif pretrained:
        # First try loading model from cache
        model_path = cached_model_path(model_name)
        if model_path:
            loaded_model = tf.keras.models.load_model(model_path)
        elif cfg.url.startswith("[timm]"):
            loaded_model = cls(cfg)
            loaded_model(loaded_model.dummy_inputs)
            # Url can be "[timm]timm_model_name" or "[timm]" in which case we default
            # to model_name.
            timm_model_name = cfg.url[len("[timm]") :] or model_name
            if "convnext" in model_name:
                timm_model_name = convnext_register[timm_model_name]
            load_timm_weights(loaded_model, timm_model_name)
        elif cfg.url.startswith("[pytorch]"):
            url = cfg.url[len("[pytorch]") :]
            loaded_model = cls(cfg)
            loaded_model(loaded_model.dummy_inputs)
            load_pth_url_weights(loaded_model, url)
        else:
            raise NotImplementedError(
                "Model not found in cache. Download of weights only implemented for "
                "PyTorch models."
            )
    else:
        loaded_model = None

    # Update config with kwargs
    cfg = deepcopy(cfg)
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            logging.warning(
                f"Config for {model_name} does not have field `{key}`. Ignoring field."
            )
    if in_channels is not None:
        setattr(cfg, "in_channels", in_channels)
    if nb_classes is not None:
        setattr(cfg, "nb_classes", nb_classes)

    # `keras.Model` kwargs need separate treatment. For now we support only `name`.
    model_kwargs = {}
    if "name" in kwargs:
        model_kwargs["name"] = kwargs["name"]

    # If we have loaded a model and the model has the correct config, then we are done.
    if loaded_model is not None and loaded_model.cfg == cfg:
        return loaded_model

    # Otherwise, we build a new model and transfer the weights to it. This is because
    # some parameter changes (in_channels and nb_classes) require changing the shape of
    # some weights or dropping of others. And there might be non-trivial interactions
    # between various parameters, e.g., global_pool can be None only if nb_classes is 0.
    model = cls(cfg, **model_kwargs)
    model(model.dummy_inputs)  # Call model to build layers

    # Now we need to transfer weights from loaded_model to model
    if loaded_model is not None:
        transfer_weights(loaded_model, model)

    return model

def make_gradcam_heatmap(img_array, model,  alpha=0.8, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        image = (img_array/255.0).astype(np.float32)
        
        preds, last_conv_layer_output = model(image)
        
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    img_array =  img_array.squeeze(0)  
    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]), resample=Image.LANCZOS)
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    
    # Superimpose the heatmap on original image
 
    img_array = np.array(img_array)  
    jet_heatmap = np.array(jet_heatmap) 
 
    #superimposed_img =  (jet_heatmap * alpha  +  img_array)   #np.array( jet_heatmap * alpha + img_array )  (jet_heatmap * img_array) * alpha +
    #superimposed_img = (superimposed_img/np.max(superimposed_img.reshape(-1, 3), axis=0) * 255)
 
    #superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
 
    return tf.squeeze(preds).numpy(), jet_heatmap #superimposed_img#.astype(np.uint8)

def compile_new_model_convnext(model_name, input_shape, num_class):
    backbone = create_model(model_name, pretrained=False)
    backbone.summary()
    inputs = Input((*input_shape, 3))
    x = BatchNormalization()(inputs)
    features = backbone(x)
    x = GlobalAveragePooling2D()(features)
    x =  Dropout(0.25)(x)
    x = Dense(num_class, activation="softmax")(x)
    #x = Dense(num_class, activation=None)(x)
    model = Model(inputs, outputs=[x, features])
    return model