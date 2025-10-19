import warnings
# Suppress urllib3 OpenSSL warning on macOS (harmless)
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

import gradio as gr
import torch
import json
from PIL import Image
from pathlib import Path
import numpy as np

from model import CarClassifier
from dataset import get_transforms


class CarPredictor:
    """Car classification predictor with Gradio interface"""

    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.checkpoint_path = Path(checkpoint_path)

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load model
        self.model, self.class_mapping = self._load_model()
        self.transform = get_transforms(train=False, img_size=224)

        print(f"Model loaded successfully!")
        print(f"Number of classes: {len(self.class_mapping['idx_to_class'])}")

    def _load_model(self):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        num_classes = checkpoint['num_classes']
        model_name = checkpoint.get('model_name', 'efficientnet_b0')

        # Create model
        model = CarClassifier(num_classes=num_classes, model_name=model_name, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        # Load class mapping
        class_mapping_file = self.checkpoint_path.parent / 'class_mapping.json'
        with open(class_mapping_file, 'r') as f:
            class_mapping = json.load(f)

        # Convert string keys back to integers for idx_to_class
        class_mapping['idx_to_class'] = {int(k): v for k, v in class_mapping['idx_to_class'].items()}

        return model, class_mapping

    def predict(self, image: Image.Image, top_k: int = 5):
        """
        Predict car make, model, and year from image

        Args:
            image: PIL Image
            top_k: Number of top predictions to return

        Returns:
            Dictionary of predictions with confidence scores
        """
        if image is None:
            return {"error": "No image provided"}

        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.class_mapping['idx_to_class'])))

        # Format results
        predictions = {}
        for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
            class_name = self.class_mapping['idx_to_class'][idx.item()]

            # Parse class name to extract make, model, year
            parsed_name = self._parse_class_name(class_name)

            predictions[f"Prediction {i+1}"] = {
                "class": parsed_name,
                "confidence": f"{prob.item() * 100:.2f}%"
            }

        return predictions

    def _parse_class_name(self, class_name: str) -> str:
        """
        Parse class name to extract readable make, model, year

        Examples:
        - 'chevrolet_impala_2008' -> 'Chevrolet Impala 2008'
        - 'compcars_135_947_2009' -> 'CompCars ID:135/947 (2009)'
        """
        if class_name.startswith('compcars_'):
            parts = class_name.replace('compcars_', '').split('_')
            if len(parts) == 3:
                return f"CompCars ID:{parts[0]}/{parts[1]} ({parts[2]})"
            return class_name
        else:
            # Replace underscores and capitalize
            parts = class_name.split('_')
            return ' '.join(word.capitalize() for word in parts)

    def predict_gradio(self, image):
        """Gradio-compatible prediction function"""
        if image is None:
            return "Please upload an image"

        predictions = self.predict(image, top_k=5)

        # Format output as readable text
        result = "🚗 Car Classification Results:\n\n"
        for pred_name, pred_data in predictions.items():
            result += f"{pred_name}:\n"
            result += f"  Car: {pred_data['class']}\n"
            result += f"  Confidence: {pred_data['confidence']}\n\n"

        return result


def create_gradio_interface(checkpoint_path: str):
    """Create and launch Gradio interface"""
    predictor = CarPredictor(checkpoint_path)

    # Create simple interface
    interface = gr.Interface(
        fn=predictor.predict_gradio,
        inputs=gr.Image(type="pil", label="Upload Car Image"),
        outputs=gr.Textbox(label="Predictions", lines=15),
        title="Car Vision - Make, Model & Year Classifier",
        description="Upload an image of a car to identify its make, model, and year. The model will return the top 5 predictions with confidence scores.",
        allow_flagging="never"
    )

    return interface


def main():
    """Main function to launch Gradio app"""
    import argparse

    parser = argparse.ArgumentParser(description='Launch car classification web app')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--share', action='store_true',
                        help='Create public share link')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to run server on')

    args = parser.parse_args()

    print("Launching Car Vision Classification App...")
    interface = create_gradio_interface(args.checkpoint)
    interface.launch(share=args.share, server_port=args.port)


if __name__ == '__main__':
    main()
