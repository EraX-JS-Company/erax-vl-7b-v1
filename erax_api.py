import sys,os,argparse
import argparse, json

from erax_api_lib import *

parser = argparse.ArgumentParser(description='EraX API')
parser.add_argument("-p", "--paths", metavar='-p', 
                    type=str, 
                    required=True,
                    help='List of paths to images or PDF files.')
parser.add_argument("-r", "--prompt", metavar='-r', 
                    type=str, 
                    required=False,
                    help='Optional: prompt for captioning.')

args = parser.parse_args()
files = args.paths.split(",")

'''
Check erax_api_lib.py to:
- use proper PROMPT which is VERY important
- tune parameters likes temperature, top_p, top_k
- only use 1-5 images for testing
'''

print ("""* Attention: please check erax_api_lib.py to: 
- use proper PROMPT which is VERY important
- tune parameters likes temperature, top_p, top_k
- only use 1-3 images for testing

""")

    
# Run
if files[0].split(".")[-1].lower()=="pdf":
    if  args.prompt:
        prompt = args.prompt
    else:
        prompt =  PDF_prompt
        
    # PDF file
    result, history =  API_PDF_OCR_EraX_VL_7B_vLLM(pdf_paths=files, 
                                                   is_base64=False,
                                                   prompt=PDF_prompt, 
                                                   erax_url_id=erax_url_a100_id, 
                                                   API_key=API_key_3P)
    
else:
    if  args.prompt:
        prompt = args.prompt
    else:
        prompt =  ycbt_prompt
    # Images files
    result, history =  API_Image_OCR_EraX_VL_7B_vLLM(image_paths=files, 
                                                     is_base64=False,
                                                     prompt=ycbt_prompt, 
                                                     erax_url_id=erax_url_id, 
                                                     API_key=API_key_3P)

final_result =  {
    "text": result, 
}

final_result = json_repair.loads(str(final_result))

# Store
import json
with open("output.json", 'w') as fout:
    json_dumps_str = json.dumps(final_result, indent=4)
    print(json_dumps_str, file=fout)

print ("Output is at output.json")
print ("Enjoy!")
