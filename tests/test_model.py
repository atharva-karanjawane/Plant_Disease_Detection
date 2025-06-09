import unittest
import numpy as np
import tensorflow as tf
from models.efficientnet import CropDiseaseModel
from models.gradcam import GradCAM

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = CropDiseaseModel(num_classes=38, img_size=224)
        self.img_size = 224
        
    def test_model_structure(self):
        self.assertIsNotNone(self.model.model)
        
        self.assertEqual(self.model.model.input_shape, (None, self.img_size, self.img_size, 3))
        
        self.assertEqual(self.model.model.output_shape, (None, 38))
    
    def test_model_prediction(self):
       
        img = np.random.random((1, self.img_size, self.img_size, 3))
        
        prediction = self.model.predict(img)
        
        self.assertEqual(prediction.shape, (1, 38))
        
        self.assertAlmostEqual(np.sum(prediction), 1.0, places=5)
    
    def test_gradcam(self):
       
        gradcam = GradCAM(self.model)
        
        img = np.random.random((1, self.img_size, self.img_size, 3))
        
        heatmap = gradcam.compute_heatmap(img, class_idx=0)
        
        self.assertEqual(heatmap.shape, (7, 7))  # EfficientNetB0 final conv layer is 7x7
        
        self.assertTrue(np.all(heatmap >= 0))
        self.assertTrue(np.all(heatmap <= 1))

if __name__ == '__main__':
    unittest.main()