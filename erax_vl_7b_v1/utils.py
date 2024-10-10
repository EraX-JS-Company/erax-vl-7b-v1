import os
import regex as re
import base64
from io import BytesIO
import uuid
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt

import pymupdf

from json_repair import repair_json
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

uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU" 

def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

dicchar = loaddicchar()

def covert_unicode(txt):
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

char  =  "àảãáạăằẳẵắặâầẩẫấậòỏõóọôồổỗốộơờởỡớợèẻẽéẹêềểễếệùủũúụưừửữứựìỉĩíịỳỷỹýỵđ" #kí tự thuần việt
kdau  =  "aAăĂâÂbBcCdDđĐeEêÊfFgGhHiIjJkKlLmMnNoOôÔơƠpPqQrRsStTuUưƯvVwWxXyYzZ"
kdau_c = "aAăĂâÂeEêÊiIoOôÔơƠuUưƯ"
cbo_di_khi_lay_van = "bcdfghjklmnpqrstuvwz"
cdau   = "aàảãáạăằẳẵắặâầẩẫấậeèẻẽéẹêềểễếệiìỉĩíịoòỏõóọôồổỗốộơờởỡớợuùủũúụưừửữứựyỳỷỹýỵAÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬEÈẺẼÉẸÊỀỂỄẾỆIÌỈĨÍỊOÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢUÙỦŨÚỤƯỪỬỮỨỰYỲỶỸÝỴ"

thanh_bang = "àằầòồờèềùừìỳy" + "àằầòồờèềùừìỳy".upper() + kdau_c
thanh_trac = "".join([i for i in cdau if i not in thanh_bang])

all_chars = cdau + kdau
bang_trac = (thanh_bang + thanh_trac)

all_chars_lower = all_chars.lower()
all_chars_lower = covert_unicode(all_chars_lower)
all_chars_lower = set(all_chars_lower)
all_chars_lower = "".join(all_chars_lower)

all_chars_lower_extra = all_chars_lower + ",;!.?"

def process_lr(text):
    tmp = text.split('\n')
    new_text = ""
    for idx, i in enumerate(tmp):
        i = i.strip()
        if len(i)==0 or i.isdigit():
            continue
        # start new line but lower case --> bring them up
        if i[0] in all_chars_lower_extra:
            new_text += " " + i
        else:
            if idx!=0:
                new_text += "\n" + i
            else:
                new_text += i
    return new_text.replace('  ', ' ').replace(' ,', ',').replace(' .', '.').strip()

def get_json(content):
    import copy
    original_content = copy.deepcopy(content)

    key = ""
    if "[" in original_content:
        if "{" in original_content:
            point_0 = original_content.find("[")
            point_1 = original_content.find("{")
            if point_0 < point_1:
                key = "["
            else:
                key = "{"
        else:
            key = "["
    else:
        key = "{"

    if key == "{":
        key_2 = "}"
    else:
        key_2 = "]"
        
    try:
        start = original_content.find(key)
        end = original_content.rfind(key_2)
        if end == -1:
            return dict()
        
        content = original_content[start:end + 1]
        content = repair_json(content, return_objects=True)
    except Exception as e:
        print(e)
        return dict()
        
    return content

max_allowed_images = 5
max_width_mm = 448

def openBase64_Image(b64):
    return Image.open(BytesIO(base64.b64decode(b64)))

# -------------------------------------------
# Generate base64 & prompt for image captioning.
# If there are more than 1 images then all images will be scaled to max_width_mm
# -------------------------------------------
def add_img_content(
    image_paths=None, 
    is_base64=False, 
    prompt=default_prompt, 
    tmp_path="./tmp/", 
    max_images=max_allowed_images, 
    force_scale=True
):
    os.makedirs(tmp_path, exist_ok=True)

    max_width_mm =  448
    scaled = False
    
    if image_paths is not None:
        if type(image_paths) == str:
            image_paths = [image_paths, ] 
        else:
            scaled = True
        if scaled or force_scale:
            scaled = True
            
            # Multi images --> scale all images to max_width_mm
            img_path_new = []
            for img_path in image_paths[:max_images]:
                if not is_base64:
                    img = Image.open(img_path)
                else:
                    img = openBase64_Image(img_path)
                w, h = img.size
                ratio = w / h
                w =  max_width_mm
                h = int(w / ratio)
                img = img.resize((w, h))
                file_name =  tmp_path + str(uuid.uuid4()) + ".jpg"
                if img.mode in ["RGBA", "P"]: 
                    img = img.convert("RGB")
                img.save(file_name)
                img_path_new.append(file_name)
            image_paths = img_path_new
        
        content_img = []
        for image_path in image_paths:
            with open(image_path, "rb") as f:
                encoded_image = base64.b64encode(f.read())
    
            img_base64 = encoded_image.decode('utf-8')
            img_base64 = f"data:image;base64,{img_base64}"
    
            content_img.append(
                   {
                    "type": "image_url",
                    "image_url": 
                        {
                            "url": img_base64
                        }
                    } 
            )
            if scaled:
                os.remove(image_path)
                
        tag_image = "".join(["<image>"] * len(image_paths))
        messages = [
            {
                "role": "user",
                "content": content_img + [
                    {
                        "type": "text", 
                        "text": f"{tag_image}\n{prompt}"
                    }]
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": f"{prompt}"
                    },
                ],
            }
        ]
    return messages

# -------------------------------------------
# Generate base64 for PDF w/ validation
# -------------------------------------------
def add_pdf_content(
    pdf_paths=None, 
    force_validated=True
):
    if type(pdf_paths)==str:
        pdf_paths =  [pdf_paths]
        
    all_base64_pdf = []
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as f:
            encoded_pdf = base64.b64encode(f.read())
        pdf_base64 = encoded_pdf.decode('utf-8')
        
        if force_validated:
            dump_name = str(uuid.uuid4())
            with open(f"{dump_name}.pdf", "wb") as f:
                f.write(base64.b64decode(pdf_base64))
            doc = pymupdf.open(f"{dump_name}.pdf")
            os.remove(f"{dump_name}.pdf")
            
        all_base64_pdf.append(pdf_base64)

    return all_base64_pdf

# ---------------------------------------------------
# Generate base64 & prompt for PDF parsing
# If these are insurance forms, use "ycbt_ocr_only_prompt" 
# ---------------------------------------------------
def add_pdf_content_json(
    pdf_paths=None, 
    prompt=PDF_prompt, 
    force_validated=True, 
    is_base64=True
):
    if type(pdf_paths)==str:
        pdf_paths = [pdf_paths, ]
        
    if not is_base64:
        all_base64_pdf = add_pdf_content(pdf_paths=pdf_paths, force_validated=force_validated)
    else:
        all_base64_pdf = pdf_paths
        
    if len(all_base64_pdf) > 0:
        messages = [
            {
                "role": "user",
                "content":[
                    {
                        "type": "pdf",
                        "text": f"{all_base64_pdf[0]}",
                        "prompt": prompt
                    }]
            }
        ]
        return messages
        
    else:
        return None