from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import base64
import io
import os
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model, load_model

app = Flask(__name__, static_folder="static")
CORS(app)

# ─── Load your trained model ──────────────────────────────────────────────────
# Replace 'tb_model.h5' with the actual path to your saved Keras model file
MODEL_PATH = os.environ.get("MODEL_PATH", "tb_model.h5")

try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"⚠️  Could not load model ({e}). Running in DEMO mode.")
    model = None


# ─── Helpers ──────────────────────────────────────────────────────────────────

IMG_SIZE = (256, 256)

def preprocess(pil_image: Image.Image) -> np.ndarray:
    """Resize, convert to RGB, normalize exactly as done during training."""
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr                          # shape (256,256,3)


def generate_gradcam(model: Model, img_array: np.ndarray) -> str:
    """
    Returns a base64-encoded PNG of the Grad-CAM heatmap overlay.
    img_array: shape (256,256,3), values in [0,1]
    """
    img_batch = np.expand_dims(img_array, 0)            # (1,256,256,3)

    # Find last Conv2D layer
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
            break
    if last_conv is None:
        return None

    grad_model = Model(inputs=model.inputs,
                       outputs=[last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)
        loss = preds[:, 0]

    grads    = tape.gradient(loss, conv_out)
    weights  = tf.reduce_mean(grads[0], axis=(0, 1))
    cam      = np.zeros(conv_out.shape[1:3], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w.numpy() * conv_out[0, :, :, i].numpy()

    cam = cv2.resize(cam, (IMG_SIZE[1], IMG_SIZE[0]))
    cam = np.maximum(cam, 0)
    if cam.max() != 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = np.uint8(255 * cam)

    heatmap    = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    original   = np.uint8(img_array * 255)
    overlay    = cv2.addWeighted(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
                                 0.45, original, 0.55, 0)

    pil_out = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    try:
        pil_img   = Image.open(file.stream)
        img_array = preprocess(pil_img)
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {e}"}), 400

    # ── Prediction ────────────────────────────────────────────────────────────
    if model is None:
        # Demo / no model loaded — return random plausible result
        prob         = float(np.random.random())
        label        = "Tuberculosis" if prob > 0.5 else "Normal"
        confidence   = prob if prob > 0.5 else 1 - prob
        gradcam_b64  = None
    else:
        prob        = float(model.predict(np.expand_dims(img_array, 0))[0][0])
        label       = "Tuberculosis" if prob > 0.5 else "Normal"
        confidence  = prob if prob > 0.5 else 1 - prob
        gradcam_b64 = generate_gradcam(model, img_array)

    # ── Return original image as base64 too (for display) ────────────────────
    buf = io.BytesIO()
    pil_img.convert("RGB").resize(IMG_SIZE).save(buf, format="PNG")
    original_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({
        "label":      label,
        "probability": round(prob, 4),
        "confidence":  round(confidence * 100, 2),
        "original":    original_b64,
        "gradcam":     gradcam_b64,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
