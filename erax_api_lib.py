#pip install json_repair numpy matplotlib tqdm pillow pymupdf opencv-python

import os, base64, cv2, json_repair, requests, pymupdf
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from PIL import Image 
import pymupdf

erax_url = "https://api.runpod.ai/v2/a4li0qhs1a6f7e/runsync"
API_key_3P =  "API key" # Digins


# # Common prompts

default_prompt = "Hãy trích xuất toàn bộ chi tiết của các bức ảnh này theo đúng thứ tự của nội dung bằng định dạng json và không bình luận gì"

PDF_prompt = """Hãy trích xuất toàn bộ chi tiết của bức ảnh này theo đúng thứ tự của nội dung trong ảnh. Không bình luận gì thêm.
Lưu ý:
1. Nếu có chữ trong ảnh thì phải trích xuất ra hết theo thứ tự và định dạng của câu chữ.
2. Nếu k=có bảng biểu (table) thì phải trả lại định dạng như bảng biểu trong hình và text của nó.
3. Nếu bức ảnh không có bất kỳ ký tự nào, hãy diễn giải bức ảnh đó.
4. Chỉ trả lại bằng tiếng Việt.

# Output:
"""

ycbt_prompt = """
Bạn là một hệ thống AI đẳng cấp thế giới hỗ trợ nhận diện ký tự quang học (Optical Character Recognition - OCR) từ hình ảnh.
Bạn được cung cấp hình ảnh hợp pháp, không vi phạm. 
Bạn phải thực hiện 01 (một) nhiệm vụ chính, bao gồm:

## Nhận diện ký tự quang học (Optical Character Recognition - OCR)
- Các ảnh được cung cấp về các giấy tờ như: phiếu khám bệnh, xét nghiệm, biên lai thu tiền, cccd, hồ sơ bệnh án, bảng kê chi phí, giấy yêu cầu bồi thường, hoá đơn giá trị gia tăng, đơn thuốc, giấy hẹn, giấy nhập viện, giấy ra viện, giấy phẩu thuật, bảng kê, hồ sơ
- Bạn cần nhận diện trung thực và chính xác nhất các từ ngữ, kí tự, số liệu xuất hiện trong hình ảnh được cung cấp.
- Ngôn ngữ chính là tiếng Việt, có thể xuất hiện thêm tiếng Anh và chữ viết tay.
- Có thể xuất hiện các bảng đơn thuốc, dịch vụ, ...
- Ảnh có thể ngược

## Yêu cầu
- Tuân thủ thứ tự: từ trái sang phải và sau đó từ trên xuống dưới
- Đặt biệt PHẢI chú ý không bỏ sót thông tin
- KHÔNG bịa đặt, đưa thêm thông tin
- Trả về kết quả theo định dạng json
- chú ý các cụm  từ viết tắt: bảo hiểm y tế (BHYT)

{
    "loại giấy tờ": <str tên loại giấy tờ>,
    "thông tin đơn vị cung cấp": { ... },
    "thông tin khách hàng": { ... },
    "thông tin sản phẩm, dịch vụ": { ... },
    ... tất cả các thông tin khác
}

## Output:
"""


# # Thư viện
# - Bạn có thể dùng thư viện Python này hoặc chuyển sang ngôn ngữ tương ứng dễ dàng

max_allowed_images = 20
max_width_mm = 448

def openBase64_Image(b64):
    from PIL import Image
    from io import BytesIO
    import base64

    return Image.open(BytesIO(base64.b64decode(b64)))
    
# -------------------------------------------
# Generate base64 & prompt for image captioning.
# If there are more thn 1 images then all images will be scaled to max_width_mm
# -------------------------------------------
def add_img_content(image_paths=None, prompt=default_prompt, tmp_path="./tmp/", max_images=max_allowed_images):

    import uuid
    os.makedirs(tmp_path, exist_ok=True)

    max_width_mm =  448
    scaled = False
    
    if image_paths is not None:
        
        if type(image_paths)==str:
            image_paths = [image_paths]
        else:
            scaled = True
            
            # Multi images --> scale all images to max_width_mm
            img_path_new = []
            for img_path in image_paths[:max_images]:

                # Use PIL instead of OpenCV
                img = Image.open(img_path)
                w, h = img.size
                ratio = w/h
                w =  max_width_mm
                h = int(w/ratio)
                img = img.resize((w, h))
                file_name =  tmp_path + str(uuid.uuid4())+".jpg" # +img_path.split(".")[-1]
                if img.mode in ("RGBA", "P"): 
                    img = img.convert("RGB")
                img.save(file_name)
                img_path_new.append(file_name)
                
            image_paths =  img_path_new
        
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

        tag_image = "".join(["<image>"]*len(image_paths))
        
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
def add_pdf_content(pdf_paths=None, force_validated=True):

    import uuid

    if type(pdf_paths)==str:
        pdf_paths =  [pdf_paths]
        
    all_base64_pdf = []
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as f:
            encoded_pdf = base64.b64encode(f.read())

        pdf_base64 = encoded_pdf.decode('utf-8')

        if force_validated:
            print ("Validating...")
            dump_name = str(uuid.uuid4())
            with open(f"{dump_name}.pdf", "wb") as f:
                f.write(base64.b64decode(pdf_base64))
            doc = pymupdf.open(f"{dump_name}.pdf")
            os.remove(f"{dump_name}.pdf")
            print ("Done Ok.")
            
        all_base64_pdf.append(pdf_base64)

    return all_base64_pdf

# ---------------------------------------------------
# Generate base64 & prompt for PDF parsing
# If these are insurance forms, use "ycbt_prompt" 
# ---------------------------------------------------
def add_pdf_content_json(pdf_paths=None, prompt=PDF_prompt, force_validated=True, is_base64=True):

    import uuid

    if type(pdf_paths)==str:
        pdf_paths = [pdf_paths]
        
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


# ---------------------------------------------------
# Extract PDF (file path or base64 --> text & images) by using EraX API
# ---------------------------------------------------
def extract_PDF_EraX_vLLM(file, pdf_base64=False, 
                          run_OCR=True, 
                          tmp_path="./tmp/", 
                          prompt=PDF_prompt,
                          OCR_func=None, erax_url=erax_url, API_key=API_key_3P):
    import os
    os.makedirs(tmp_path, exist_ok=True)

    if pdf_base64:
        dump_name = str(uuid.uuid4())
        file_name = f"{dump_name}.pdf"
        with open(file_name, "wb") as f:
            f.write(base64.b64decode(file))
        new_file =  file_name
    else:
        file_name = file.split("/")[-1]
        new_file = file
    
    doc = pymupdf.open(new_file) # open a document

    all_content = []
    
    for page_index in tqdm(range(len(doc)), total=len(doc)): # iterate over pdf pages

        page_content = {}
        page_content["page"] =  page_index
        
        page = doc[page_index] # get the page
        
        page_content["text"] =  page.get_text().encode("utf8").decode("utf-8")

        image_list = page.get_images()
    
        # print the number of images found on the page
        if image_list:
            print(f"Found {len(image_list)} images on page {page_index}")
        else:
            print("No images found on page", page_index)

        page_img = []
        
        for image_index, img in enumerate(image_list, start=1): # enumerate the image list
            xref = img[0] # get the XREF of the image
            pix = pymupdf.Pixmap(doc, xref) # create a Pixmap
    
            if pix.n - pix.alpha > 3: # CMYK: convert to RGB first
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

            image_path = tmp_path+file_name+"_page_%s-image_%s.png" % (page_index, image_index)
            
            pix.save(image_path) # save the image as png

            if run_OCR:
                tmp, _ = OCR_func(image_path, prompt=prompt, erax_url=erax_url, API_key=API_key)
            else:
                tmp = ""
                
            page_img.append({
                "index": image_index,
                "image": Image.open(image_path),
                "text" : "** Ảnh " + str(image_index) + "/trang " + str(page_index+1) + " **\n" + tmp
            })
            os.remove(image_path)
            
            pix = None
        
        page_content["images_text"] =  page_img
        
        all_content.append(page_content)

    full_text = ""
    for content in all_content:
        full_text += "** Trang " + str(content["page"]+1) + " **\n" + process_lr(content["text"]) + "\n"
        for image_text in content["images_text"]:
            full_text += image_text["text"] + "\n"
        full_text += "\n"
        
    return all_content, full_text


# # Image captioning w/ list of [images base64]
# - Bạn có thể dùng API này để captioning ảnh
# - Lưu ý prompt hợp lý theo đúng kiểu văn bản cần parse
# - API chỉ chấp nhận tối đa 20 ảnh nhưng bạn nên captioning tối đa 3 ảnh
# - API này kỳ vọng bạn truyền vào list các base64 thuần của ảnh
# - Lưu ý prefix: API đã thêm "data:image;base64" trước decoded {base64} của ảnh rồi

# -------------------------------------------
# Call API to parse Images. Pass in base64
# -------------------------------------------
def API_Image_Base64_OCR_EraX_VL_7B_vLLM(images_base64=None, prompt=default_prompt, erax_url=erax_url, API_key=API_key_3P):

    tag_image = "".join(["<image>"]*len(images_base64))
    content_img
    for b64 in images_base64:      
        content_img.append(
                   {
                    "type": "image_url",
                    "image_url": 
                        {
                            "url": f"data:image;base64,{b64}"
                        }
                    } 
            )
    
    
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
    
    content = {
        "generation_config": 
        {
            "temperature": float(0.2),
            "top_p": float(0.001),
            "top_k": int(1),
            "repetition_penalty": float(1.1),
            "max_tokens": 32000
        },
        "messages": messages
    }   

    data_to_send ={
        "input": content
    }

    head = {}
    head["authorization"] = API_key

    print ("Calling EraX API ....")
    
    res = requests.post(erax_url, headers=head, json=data_to_send)
    
    error = False
    
    try:
        result =  res.json()["output"]
    except:
        result =  res.json()
        error = True
        
    content["messages"].append({
            "role": "assistant",
            "content": result,
            "error": error
        }
    )
    
    return result, content
 


# # Image captioning w/ list of [images paths]
# - Bạn có thể dùng API này để captioning ảnh
# - Lưu ý prompt hợp lý theo đúng kiểu văn bản cần parse
# - API chỉ chấp nhận tối đa 20 ảnh nhưng bạn nên captioning tối đa 3 ảnh vì GPU có hạn
# - API này kỳ vọng bạn truyền vào đường dẫn đến các file ảnh (jpg, png, gif etc..)

# -------------------------------------------
# Call API to parse Images. Pass in image paths
# -------------------------------------------
def API_Image_Paths_OCR_EraX_VL_7B_vLLM(image_paths=None, prompt=default_prompt, erax_url=erax_url, API_key=API_key_3P):

    messages = add_img_content(image_paths, prompt=prompt)
    
    content = {
        "generation_config": 
        {
            "temperature": float(0.2),
            "top_p": float(0.001),
            "top_k": int(1),
            "repetition_penalty": float(1.1),
            "max_tokens": 32000
        },
        "messages": messages
    }   

    data_to_send ={
        "input": content
    }

    head = {}
    head["authorization"] = API_key

    print ("Calling EraX API ....")
    
    res = requests.post(erax_url, headers=head, json=data_to_send)
    
    error = False
    
    try:
        result =  res.json()["output"]
    except:
        result =  res.json()
        error = True
        
    content["messages"].append({
            "role": "assistant",
            "content": result,
            "error": error
        }
    )
    
    return result, content


# # PDF captioning w/ PDF paths
# - Bạn có thể dùng API này để parse PDF cả text & ảnh trong đó
# - Lưu ý prompt hợp lý theo đúng kiểu văn bản cần parse
# - API chỉ chấp nhận 1 PDF tại 1 thời điểm
# - API này kỳ vọng bạn truyền vào đường dẫn đến file PDF

def API_PDF_Paths_OCR_EraX_VL_7B_vLLM(pdf_paths=None, prompt=PDF_prompt, erax_url=erax_url, API_key=API_key_3P):

    messages = add_pdf_content_json(pdf_paths, prompt=prompt, is_base64=False)
    
    content = {
        "generation_config":
        {
            "temperature": float(0.2),
            "top_p": float(0.001),
            "top_k": int(1),
            "repetition_penalty": float(1.1),
            "max_tokens": 32000
        },
        "messages": messages
    }   

    data_to_send ={
        "input": content
    }

    head = {}
    head["authorization"] = API_key

    print ("Calling EraX API ....")
    
    res = requests.post(erax_url, headers=head, json=data_to_send)
    
    error = False
    
    try:
        result =  res.json()["output"]
    except:
        result =  res.json()
        error = True
        
    content["messages"].append({
            "role": "assistant",
            "content": result,
            "error": error
        }
    )
    
    return result, content


# # PDF captioning w/ PDF base64
# - Bạn có thể dùng API này để parse PDF cả text & ảnh trong đó
# - Lưu ý prompt hợp lý theo đúng kiểu văn bản cần parse
# - API chỉ chấp nhận 1 PDF tại 1 thời điểm
# - API này kỳ vọng bạn đã tạo decoded-base64 cho file PDF của mình

def API_PDF_Base64_OCR_EraX_VL_7B_vLLM(pdf_base64=None, prompt=PDF_prompt, erax_url=erax_url, API_key=API_key_3P):

    messages = add_pdf_content_json(pdf_base64, prompt=prompt, is_base64=True)
    
    content = {
        "generation_config": 
        {
            "temperature": float(0.2),
            "top_p": float(0.001),
            "top_k": int(1),
            "repetition_penalty": float(1.1),
            "max_tokens": 32000
        },
        "messages": messages
    }   

    data_to_send ={
        "input": content
    }

    head = {}
    head["authorization"] = API_key

    print ("Calling EraX API ....")
    
    res = requests.post(erax_url, headers=head, json=data_to_send)
    
    error = False
    
    try:
        result =  res.json()["output"]
    except:
        result =  res.json()
        error = True
        
    content["messages"].append({
            "role": "assistant",
            "content": result,
            "error": error
        }
    )
    
    return result, content


# # Chat with the previous result from EraX
# - Bạn có thể hội thoại liên tục với kết quả EraX đã captioning lần trước hoặc đơn giản là chat với QWen2

def API_Chat_OCR_EraX_VL_7B_vLLM(prompt, history=None, erax_url=erax_url, API_key=API_key_3P):

    history["messages"].append({
        "role": "user",
        "content": prompt
        }
    )
                               
    data_to_send ={
        "input": history
    }

    head = {}
    head["authorization"] = API_key
    
    print ("Calling EraX API ....")
    
    res = requests.post(erax_url, headers=head, json=data_to_send)
    
    result =  res.json()["output"]

    history["messages"].append({
            "role": "assistant",
            "content": result
    }
    )
    
    return result, history
