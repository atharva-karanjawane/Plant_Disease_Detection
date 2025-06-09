import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
from models.efficientnet import CropDiseaseModel
from data.dataset import PlantVillageDataset
from models.gradcam import GradCAM

def evaluate_model(config):
    """Evaluate the trained model on the test set."""
    print("Loading dataset...")
    dataset = PlantVillageDataset(
        data_dir=config.DATASET_PATH,
        img_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE
    )
    
    _, _, test_ds = dataset.load_data(
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT
    )
    
    print("Loading model...")
    model = CropDiseaseModel(
        num_classes=config.NUM_CLASSES,
        img_size=config.IMG_SIZE
    )
    model.load_weights(config.MODEL_PATH)
    
    os.makedirs('training/evaluation', exist_ok=True)
    
    print("Evaluating model...")
    test_loss, test_acc = model.model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    y_pred = []
    y_true = []
    
    for images, labels in test_ds:
        predictions = model.model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels, axis=1))
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('training/evaluation/confusion_matrix.png')
    plt.close()
    
    report = classification_report(
        y_true, y_pred, 
        target_names=dataset.class_names,
        output_dict=True
    )
    
    # Plot classification report
    plot_classification_report(report, dataset.class_names)
    
    generate_gradcam_examples(model, test_ds, dataset.class_names)
    
    return {
        'accuracy': test_acc,
        'loss': test_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_classification_report(report, class_names):
    """Plot classification report as a heatmap."""
    
    classes = list(report.keys())[:-3]  
    
    
    metrics = ['precision', 'recall', 'f1-score']
    data = []
    
    for cls in classes:
        row = [report[cls][metric] for metric in metrics]
        data.append(row)
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data, 
        annot=True, 
        cmap='YlGnBu', 
        xticklabels=metrics, 
        yticklabels=[name.replace('___', ' - ').replace('_', ' ') for name in classes],
        vmin=0, vmax=1
    )
    plt.title('Classification Report')
    plt.tight_layout()
    plt.savefig('training/evaluation/classification_report.png')
    plt.close()

def generate_gradcam_examples(model, test_ds, class_names, num_examples=5):
    """Generate GradCAM visualizations for a few examples."""
    
    gradcam = GradCAM(model)
    
    os.makedirs('training/evaluation/gradcam', exist_ok=True)
    
    for images, labels in test_ds:
        for i in range(min(num_examples, len(images))):
            img = images[i]
            true_label = np.argmax(labels[i])
            
            pred = model.model.predict(np.expand_dims(img, axis=0))[0]
            pred_label = np.argmax(pred)
            
            heatmap = gradcam.compute_heatmap(
                np.expand_dims(img, axis=0), 
                pred_label
            )
            
            img = (img * 255).astype(np.uint8)
            
            heatmap = np.uint8(255 * heatmap)
            heatmap = np.expand_dims(heatmap, axis=-1)
            heatmap = np.tile(heatmap, (1, 1, 3))
            
            heatmap = plt.cm.jet(heatmap / 255.0) * 255
            heatmap = heatmap[:, :, :3].astype(np.uint8)
            
            superimposed = np.uint8(0.6 * img + 0.4 * heatmap)
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f"Original: {class_names[true_label].replace('___', ' - ').replace('_', ' ')}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(superimposed)
            plt.title(f"Prediction: {class_names[pred_label].replace('___', ' - ').replace('_', ' ')} ({pred[pred_label]:.2f})")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'training/evaluation/gradcam/example_{i+1}.png')
            plt.close()
        
        break

if __name__ == "__main__":
    from config import DevelopmentConfig
    
    evaluate_model(DevelopmentConfig)