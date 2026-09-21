from ml.master.router.router import MasterRouter
import traceback

router = MasterRouter()
file_path = "/Users/siddarthguruprasad/.gemini/antigravity/brain/tempmediaStorage/media__1789952000585.png"

with open(file_path, "rb") as f:
    file_bytes = f.read()

print("Testing with actual uploaded image (Vision route):")
try:
    result = router.infer(file_bytes, "image/png")
    print(result)
    if not result["success"]:
        print("ERROR:", result["error_information"])
except Exception as e:
    traceback.print_exc()
