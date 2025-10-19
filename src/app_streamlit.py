import warnings
# Suppress urllib3 OpenSSL warning on macOS (harmless)
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

import streamlit as st
import torch
import json
from PIL import Image
from pathlib import Path
import sys

from model import CarClassifier
from dataset import get_transforms


class CarPredictor:
    """Car classification predictor"""

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

        # Load model
        self.model, self.class_mapping = self._load_model()
        self.transform = get_transforms(train=False, img_size=224)

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
            List of (class_name, confidence) tuples
        """
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.class_mapping['idx_to_class'])))

        # Format results
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_name = self.class_mapping['idx_to_class'][idx.item()]
            parsed_name = self._parse_class_name(class_name)
            confidence = prob.item() * 100
            predictions.append((parsed_name, confidence))

        return predictions

    def _parse_class_name(self, class_name: str) -> str:
        """
        Parse class name to extract readable make, model, year
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


@st.cache_resource
def load_predictor(checkpoint_path):
    """Load predictor (cached)"""
    return CarPredictor(checkpoint_path)


def main():
    # Page config
    st.set_page_config(
        page_title="Car Vision Classifier",
        page_icon="🚗",
        layout="centered"
    )

    # Title
    st.title("🚗 Car Vision Classifier")
    st.markdown("Upload an image of a car to identify its **make**, **model**, and **year**")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This model is trained on:
        - Most Stolen Cars dataset
        - SubsetVMMR dataset
        - CompCars dataset

        The model will return the top 5 predictions with confidence scores.
        """)

        checkpoint_path = st.text_input(
            "Model Checkpoint Path",
            value="checkpoints/best_model.pth"
        )

    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        st.error(f"❌ Checkpoint not found: {checkpoint_path}")
        st.info("Please train the model first using: `python src/train.py --datasets stolen_cars vmmr --epochs 10`")
        return

    # Load predictor
    try:
        with st.spinner("Loading model..."):
            predictor = load_predictor(checkpoint_path)
        st.success(f"✅ Model loaded! Using device: {predictor.device}")
        st.info(f"📊 Number of classes: {len(predictor.class_mapping['idx_to_class'])}")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a car image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear photo of a car (front, side, or rear view works best)"
    )

    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)

        # Predict
        with st.spinner("Analyzing car..."):
            predictions = predictor.predict(image, top_k=5)

        # Display predictions
        with col2:
            st.subheader("Predictions")

            for i, (car_name, confidence) in enumerate(predictions, 1):
                # Color code by confidence
                if confidence > 50:
                    conf_color = "🟢"
                elif confidence > 20:
                    conf_color = "🟡"
                else:
                    conf_color = "🔴"

                st.markdown(f"**#{i}** {conf_color} **{car_name}**")
                st.progress(confidence / 100)
                st.caption(f"Confidence: {confidence:.2f}%")
                st.markdown("---")

        # Show top prediction prominently
        st.success(f"### 🎯 Most Likely: **{predictions[0][0]}** ({predictions[0][1]:.1f}%)")

    else:
        # Placeholder
        st.info("👆 Upload an image to get started!")

        # Example instructions
        st.markdown("### Tips for best results:")
        st.markdown("""
        - ✅ Use clear, well-lit photos
        - ✅ Car should be the main subject
        - ✅ Side, front, or rear views work best
        - ❌ Avoid heavily obscured or distant cars
        - ❌ Very new cars (2020+) may not be in the dataset
        """)


if __name__ == '__main__':
    main()
