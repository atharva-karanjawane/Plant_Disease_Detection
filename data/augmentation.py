import tensorflow as tf
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, layer_name=None):
        """Initialize GradCAM with a model and target layer."""
        self.model = model
        
        if layer_name is None:
            for layer in reversed(model.model.layers):
                if len(layer.output_shape) == 4:
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        self.grad_model = self._build_grad_model()
        
    def _build_grad_model(self):
        """Build a gradient model for GradCAM."""
       
        layer = self.model.model.get_layer(self.layer_name)
       
        return tf.keras.models.Model(
            inputs=[self.model.model.inputs],
            outputs=[layer.output, self.model.model.output]
        )
    
    def compute_heatmap(self, image, class_idx, eps=1e-8):
        """Generate GradCAM heatmap for the specified class."""
      
        with tf.GradientTape() as tape:
           
            conv_outputs, predictions = self.grad_model(image)
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, conv_outputs)
        
        cast_conv_outputs = tf.cast(conv_outputs > 0, tf.float32)
        cast_grads = tf.cast(grads > 0, tf.float32)
        guided_grads = cast_conv_outputs * cast_grads * grads
        
        weights = tf.reduce_mean(guided_grads, axis=(1, 2))
        
        conv_outputs = conv_outputs[0]
        weights = weights[0]
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        
        heatmap = np.maximum(cam, 0)
        max_heat = np.max(heatmap)
        if max_heat != 0:
            heatmap = heatmap / max_heat
        
        return heatmap