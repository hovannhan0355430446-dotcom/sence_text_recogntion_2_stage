import os
import sys
from pathlib import Path
import uvicorn

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    os.chdir(current_dir)
    sys.path.insert(0, str(current_dir))
    print("Dang khoi dong he thong OCR tai http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

