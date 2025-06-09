import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

class CropDiseaseModel:
    def __init__(self, num_classes=38, img_size=224):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = self._build_model()
        
    def _build_model(self):
        """Build and compile the EfficientNetB0 model."""
        # Base model
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_weights(self, weights_path):
        """Load model weights."""
        try:
            self.model.load_weights(weights_path)
            print(f"Loaded weights from {weights_path}")
        except:
            print(f"Could not load weights from {weights_path}. Using initialized weights.")
    
    def predict(self, img):
        """Make a prediction on a preprocessed image."""
        return self.model.predict(img)
    
    def get_model(self):
        """Return the Keras model."""
        return self.model