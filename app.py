import os
import torch
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    ToTensord,
)
from monai.networks.nets import DenseNet121, EfficientNetBN
import torch.nn.functional as F

##############################################################################
# 1. Flask App Setup
##############################################################################

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Label mapping from your training code: 0 -> Benign, 1 -> Malignant
class_labels = {0: "Benign", 1: "Malignant"}

##############################################################################
# 2. Device and Model Setup
##############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same logic as your "get_model" function in model.py
def get_model(model_name: str, pretrained: bool = False):
    if model_name == "DenseNet121":
        return DenseNet121(spatial_dims=2, in_channels=3, out_channels=2, pretrained=pretrained)
    elif model_name == "EfficientNetB0":
        return EfficientNetBN("efficientnet-b0", spatial_dims=2, in_channels=3, num_classes=2, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

MODEL_NAME = "DenseNet121"  # or "EfficientNetB0", if that's what you trained
MODEL_PATH = "DenseNet_pretrained3.pth"  # Adjust if your weights file is named differently
model = get_model(MODEL_NAME, pretrained=False)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

# Load the trained weights and set eval mode
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

##############################################################################
# 3. Define the Same Test Transforms (dictionary-based) from model.py
##############################################################################

test_transforms = Compose([
    LoadImaged(keys=["image_path"]),             # loads file -> numpy array
    EnsureChannelFirstd(keys=["image_path"]),    # (C, H, W)
    ScaleIntensityd(keys=["image_path"]),        # min-max scale each image
    Resized(keys=["image_path"], spatial_size=(224, 224)),
    ToTensord(keys=["image_path"]),              # convert to torch.Tensor
])

##############################################################################
# 4. Flask Routes
##############################################################################

@app.route("/", methods=["GET"])
def index():
    """
    Render a simple page with an image upload form.
    """
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    1) Save the uploaded file
    2) Use dictionary-based MONAI transforms to preprocess
    3) Run inference using the loaded model
    4) Display the prediction and softmax probabilities
    """
    try:
        # Check if a file was uploaded
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template("index.html", error="No file selected.")

        # Save the uploaded image
        file = request.files["file"]
        filename = file.filename
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # -------------------------------------------------------
        # A) Dictionary-based transforms for a single image
        # -------------------------------------------------------
        sample_dict = {"image_path": file_path}
        sample_dict = test_transforms(sample_dict)  
        # sample_dict["image_path"] is now a torch tensor of shape [C, H, W]

        input_tensor = sample_dict["image_path"].unsqueeze(0).to(device)  # [1, C, H, W]

        # -------------------------------------------------------
        # B) Inference
        # -------------------------------------------------------
        with torch.no_grad():
            logits = model(input_tensor)   # shape: [1, 2]
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # shape: (2,)
            predicted_idx = np.argmax(probs)

        predicted_label = class_labels.get(predicted_idx, "Unknown")
        prob_benign = probs[0]
        prob_malignant = probs[1]

        # -------------------------------------------------------
        # C) Render the result
        # -------------------------------------------------------
        return render_template(
            "result.html",
            filename=filename,
            predicted_label=predicted_label,
            prob_benign=f"{prob_benign:.4f}",
            prob_malignant=f"{prob_malignant:.4f}"
        )

    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """
    If you want to directly access the uploaded file via browser,
    this route will redirect to the static/uploads folder.
    """
    return redirect(url_for("static", filename=f"uploads/{filename}"))

##############################################################################
# 5. Run the Flask App
##############################################################################

if __name__ == "__main__":
    # Debug=True for easier troubleshooting; remove in production
    app.run(host="0.0.0.0", port=8000, debug=True)
