from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from diffusers import DiffusionPipeline
import torch
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

app = Flask(__name__)
CORS(app)

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the DeepFloyd IF model (Stage I)
print("⏳ Loading DeepFloyd IF model (Stage I)...")

try:
    pipe = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.float32
)
    pipe.to("cpu")  # Change to "cuda" if you use GPU
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", str(e))
    pipe = None  # Prevent crash if model failed to load

@app.route("/generate", methods=["POST"])
def generate_image():
    if pipe is None:
        return jsonify({"error": "Model is not loaded"}), 500

    try:
        data = request.json
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        print(f"🚀 Generating image for prompt: {prompt}")
        result = pipe(prompt=prompt)
        image = result.images[0]

        # Save image with a unique name
        filename = f"{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image.save(filepath)

        return jsonify({"filename": filename})

    except Exception as e:
        print("❌ Error during generation:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/image/<filename>")
def get_image(filename):
    try:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='image/png')
        else:
            return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
