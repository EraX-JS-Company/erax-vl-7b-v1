import os
import sys
from pathlib import Path
import argparse
import json
import json_repair

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from erax_vl_7b_v1.prompts import (
    sickness_medicines, 
    cities, 
    default_prompt, 
    PDF_prompt, 
    popular_prompt, 
    ycbt_ocr_only_prompt, 
    pdf_full_prompt, 
    ycbt_ocr_conversation_single_image_prompt
)
from erax_vl_7b_v1.utils import (
    process_lr,
    get_json,
    openBase64_Image,
    add_img_content,
    add_pdf_content,
    add_pdf_content_json
)
from erax_vl_7b_v1.erax_api_lib import (
    API_Image_OCR_EraX_VL_7B_vLLM,
    API_PDF_OCR_EraX_VL_7B_vLLM,
    API_Chat_OCR_EraX_VL_7B_vLLM,
    API_Multiple_Images_OCR_EraX_VL_7B_vLLM,
    API_PDF_Full_OCR_EraX_VL_7B_vLLM
)

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    ERAX_URL_ID = os.getenv('ERAX_URL_ID')
    API_KEY = os.getenv('API_KEY')
    
    image_path = "./images/hoadon.jpg"
    
    prompt = """Hãy trích xuất toàn bộ chi tiết của các bức ảnh này theo đúng thứ tự của nội dung bằng định dạng json và không bình luận gì.
    """
    
    result, history =  API_Image_OCR_EraX_VL_7B_vLLM(
            image_paths=image_path, 
            is_base64=False,
            prompt=prompt, 
            erax_url_id=ERAX_URL_ID, 
            API_key=API_KEY,
            force_scale=True
        )
    
    # Convert string json to json. It is result.
    json_result = get_json(result) 
    
    print(json_result)