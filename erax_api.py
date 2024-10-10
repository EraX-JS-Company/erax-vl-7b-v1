import os
import sys
import argparse
import json
import json_repair
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

def main(args):
    ERAX_URL_ID = os.getenv('ERAX_URL_ID')
    API_KEY = os.getenv('API_KEY')
    
    files = args.paths
    if files[0].split(".")[-1].lower() == "pdf":
        
        if  args.prompt:
            prompt = args.prompt
        else:
            prompt =  PDF_prompt
            
        # PDF file
        result, history =  API_PDF_OCR_EraX_VL_7B_vLLM(
            pdf_paths=files, 
            is_base64=False,
            prompt=prompt, 
            erax_url_id=ERAX_URL_ID, 
            API_key=API_KEY
        )
                
    else:
        if  args.prompt:
            prompt = args.prompt
        else:
            prompt = default_prompt
            
        # Images files
        result, history =  API_Image_OCR_EraX_VL_7B_vLLM(image_paths=files, 
                                                        is_base64=False,
                                                        prompt=prompt, 
                                                        erax_url_id=ERAX_URL_ID, 
                                                        API_key=API_KEY)
    
    final_result = get_json(result)
    with open("output.json", 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
        print(final_result)

    print("Output is at output.json")
    print("Enjoy!")

def parse_opt():
    parser = argparse.ArgumentParser(description='EraX API')
    parser.add_argument("-p", "--paths", metavar='-p', 
                        type=str, 
                        required=True,
                        help='List of paths to images or PDF files, paths separated by commas')
    parser.add_argument("-r", "--prompt", metavar='-r', 
                        type=str, 
                        required=False,
                        help='Optional: prompt for captioning')

    args = parser.parse_args()
    args.path = args.paths.split(",")
    return args

if __name__ == "__main__":
    """* Attention: please check erax_api_lib.py to: 
        - use proper PROMPT which is VERY important
        - tune parameters likes temperature, top_p, top_k
        - only use 1-3 images for testing
    """
    
    args = parse_opt()
    main(args=args)