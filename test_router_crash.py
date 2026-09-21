from ml.master.router.router import MasterRouter
from PIL import Image
import io

router = MasterRouter()
img = Image.new('RGB', (100, 100), color='white')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
file_bytes = img_byte_arr.getvalue()

print("Testing Vision route:")
result_vision = router.infer(file_bytes, "image/png")
print(result_vision)

print("Testing Reasoning route:")
result_reasoning = router.infer(file_bytes, "image/png", "What is this?")
print(result_reasoning)
