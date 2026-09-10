import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# ── Model architecture ──────────────────────────────────────────────────
# Must exactly match the training notebook's ResNet50Model so the saved
# state_dict (weights only, not a full pickled model) loads correctly.
class ResNet50Model(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(ResNet50Model, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(num_features, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ── Config ────────────────────────────────────────────────────────────
# Confirmed against the PASSION training notebook: Resize(256) then
# CenterCrop(224), ImageNet normalization, classes sorted alphabetically
# (Eczema, Fungal, Others, Scabies), checkpoint is a state_dict.
#
# Paths are resolved relative to this script's own location, not the
# current working directory, since Streamlit Cloud and local `streamlit
# run` can have different working directories for the same script.
APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = APP_DIR / "models" / "best_model_stage2.pt"

NUM_CLASSES = 4
CLASS_NAMES = ["Eczema", "Fungal", "Others", "Scabies"]
DEVICE = torch.device("cpu")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.set_page_config(page_title="PASSION — Skin Disease Diagnosis", page_icon="🩺")


@st.cache_resource
def load_model():
    model = ResNet50Model(NUM_CLASSES, pretrained=False)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict(image: Image.Image):
    model = load_model()
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()
    return probs


# ── UI ───────────────────────────────────────────────────────────────
st.title("🩺 PASSION")
st.caption("Skin disease diagnosis from a photo of a skin lesion.")

st.info(
    "**This is a research demo, not a medical diagnostic tool.** "
    "It is not a substitute for professional medical advice. If you have "
    "a concerning skin condition, please consult a qualified healthcare "
    "provider."
)

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running inference..."):
        probs = predict(image)

    top_idx = int(probs.argmax())
    st.metric("Diagnosis", CLASS_NAMES[top_idx], f"{probs[top_idx]:.1%} confidence")

    st.write("Full breakdown:")
    for name, p in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1]):
        st.write(f"{name}: {p:.1%}")
        st.progress(float(p))
else:
    st.info("Upload a skin lesion image above to get started.")

st.divider()
with st.expander("About this model"):
    st.write(
        "PASSION is a ResNet-50 model trained on the PASSION MICCAI 2024 "
        "dataset (4,901 images, 1,653 subjects) across four classes: "
        "Eczema, Fungal infection, Scabies, and Others. Trained with a "
        "two-stage fine-tuning approach, reaching 80.6% test accuracy "
        "and 80.6% balanced accuracy on held-out data."
    )