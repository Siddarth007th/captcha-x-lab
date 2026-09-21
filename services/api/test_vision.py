import os
import sys

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ml.master.router.router import MasterRouter
import traceback

print("Testing Vision Model Initialization...")
try:
    router = MasterRouter()
    model, processor = router.manager._load_vision()
    print("SUCCESS!")
except Exception as e:
    traceback.print_exc()
