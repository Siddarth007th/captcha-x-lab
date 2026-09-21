from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.abspath("services/api/src"))
from api.main import app
from PIL import Image
import io

client = TestClient(app)

img = Image.new('RGB', (100, 100), color='white')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
file_bytes = img_byte_arr.getvalue()

print("Sending request to /inference/predict without text query (Vision)...")
response = client.post(
    "/inference/predict",
    files={"file": ("dummy.png", file_bytes, "image/png")}
)
print("Response JSON:")
print(response.json())

print("\nSending request to /inference/predict with text query (Reasoning)...")
response = client.post(
    "/inference/predict",
    files={"file": ("dummy.png", file_bytes, "image/png")},
    data={"text_query": "What is this?"}
)
print("Response JSON:")
print(response.json())
