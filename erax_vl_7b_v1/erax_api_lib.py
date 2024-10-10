import os
import regex as re
import base64
from io import BytesIO
import uuid
import json_repair
from tqdm import tqdm

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import pymupdf

import requests
import time
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
from erax_vl_7b_v1.utils import (
    process_lr,
    get_json,
    openBase64_Image,
    add_img_content,
    add_pdf_content,
    add_pdf_content_json
)

erax_url_id = "<erax_url_id>"
API_key = "<API_key>"

def checkStatusLongRun(
    ocr_result, 
    erax_url_id=erax_url_id, 
    API_key=API_key
):
    """RunPod long run
        - Lưu ý cho PDF nhiều trang hay nhiều ảnh, API sẽ trả về IN_PROGRESS
        - Dùng /status để check progress và lấy về
    """
    final_result = ocr_result.copy()
    while True:
        time.sleep(0.5)
        if type(final_result) == dict:
            if "status" in final_result.keys() and (final_result["status"] == "IN_PROGRESS" or final_result["status"] == "IN_QUEUE"):
                job_id = final_result["id"]
                print(f"Check status & result...{job_id}")
                runpod_status_url = f"https://api.runpod.ai/v2/{erax_url_id}/status/{job_id}"
                head = dict()
                head["authorization"] = API_key
                final_result = requests.post(runpod_status_url, headers=head, timeout=120).json()
            else:
                break
        else:        
            break
            
    return final_result

def API_Image_OCR_EraX_VL_7B_vLLM(
    image_paths=None,
    is_base64=True, 
    prompt=default_prompt, 
    erax_url_id=erax_url_id, 
    API_key=API_key, 
    force_scale=True
):
    """Image Captioning w/ list of [images paths or base64]
        - Bạn có thể dùng API này để captioning ảnh
        - Lưu ý prompt hợp lý theo đúng kiểu văn bản cần parse
        - API chỉ chấp nhận tối đa 20 ảnh nhưng bạn nên captioning tối đa 3 ảnh
        - API này kỳ vọng bạn truyền vào list các base64 thuần của ảnh
        - Lưu ý prefix: API đã thêm "data:image;base64" trước decoded {base64} của ảnh rồi
    """
    messages = add_img_content(
        image_paths, 
        is_base64=is_base64, 
        prompt=prompt, 
        force_scale =force_scale
    )
    
    content = {
        "generation_config": 
        {
            "temperature": float(0.01),
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

    erax_url = f"https://api.runpod.ai/v2/{erax_url_id}/runsync"   
    res = requests.post(erax_url, headers=head, json=data_to_send, timeout=3600)
    
    error = False
    
    try:
        result =  res.json()["output"]
    except:
        result =  res.json()
        try:
            result = checkStatusLongRun(result, erax_url_id=erax_url_id)["output"]
        except:
            error = True
                
    content["messages"].append({
            "role": "assistant",
            "content": result,
            "error": error
        }
    )
    
    return result, content

def API_PDF_OCR_EraX_VL_7B_vLLM(
    pdf_paths=None, 
    is_base64=False,
    prompt=PDF_prompt, 
    erax_url_id=erax_url_id, 
    API_key=API_key
):
    """PDF Captioning with PDF paths or base64
        - Bạn có thể dùng API này để parse PDF cả text & ảnh trong đó
        - Lưu ý prompt hợp lý theo đúng kiểu văn bản cần parse
        - API chỉ chấp nhận 1 PDF tại 1 thời điểm
        - API này kỳ vọng bạn truyền vào đường dẫn đến file PDF or list of PDF's base64
    """
    messages = add_pdf_content_json(
        pdf_paths, 
        prompt=prompt, 
        is_base64=is_base64
    )
    
    content = {
        "generation_config": 
        {
            "temperature": float(0.01),
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

    erax_url = f"https://api.runpod.ai/v2/{erax_url_id}/runsync"   
    res = requests.post(erax_url, headers=head, json=data_to_send, timeout=3600)
    
    error = False
    
    try:
        result =  res.json()["output"]
    except:
        result =  res.json()
        try:
            result = checkStatusLongRun(result, erax_url_id=erax_url_id)["output"]
        except:
            error = True
                        
    content["messages"].append({
            "role": "assistant",
            "content": result,
            "error": error
        }
    )
    
    return result, content

def API_Chat_OCR_EraX_VL_7B_vLLM(
    prompt,
    history=None, 
    erax_url_id=erax_url_id, 
    API_key=API_key
):
    """Chat with the previous results from EraX
        - Bạn có thể hội thoại liên tục với kết quả EraX đã captioning lần trước hoặc đơn giản là chat với QWen2
    """
    if history is not None:
        history["messages"].append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":prompt
                }
            ]
        })
    else:
        history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text":prompt
                    }
                ]
            }
        ]
    content = {
        "generation_config": 
        {
            "temperature": float(0.1),
            "top_p": float(0.1),
            "top_k": int(10),
            "repetition_penalty": float(1.1),
            "max_tokens": 8192
        },
        "messages": history
    }   
    data_to_send ={
        "input": content
    }

    head = {}
    head["authorization"] = API_key
    
    erax_url = f"https://api.runpod.ai/v2/{erax_url_id}/runsync"   
    res = requests.post(erax_url, headers=head, json=data_to_send, timeout=3600)
    
    error = False
    
    try:
        result =  res.json()["output"]
    except:
        result =  res.json()
        try:
            result = checkStatusLongRun(result, erax_url_id=erax_url_id)["output"]
        except:
            error = True
                                
    content["messages"].append(
        {
            "role": "assistant",
            "content": result
        }
    )
    
    return result, content

def API_Multiple_Images_OCR_EraX_VL_7B_vLLM(
    image_paths=None,
    is_base64=False,
    prompt=ycbt_ocr_only_prompt, 
    pdf_full_prompt=pdf_full_prompt, 
    erax_url_id=erax_url_id, 
    API_key=API_key
):
    """Multiple Images Captioning with paths OR Base64
        - Bạn có thể dùng API này để parse multiple images cả text & ảnh trong đó
        - Lưu ý prompt hợp lý theo đúng kiểu văn bản cần parse
    """
    print ("--> Parsing all images...")

    output_text = ""
    for idx, img_path in enumerate(image_paths):
        print (f"- Parsing image...{idx}")        
        ocr_result, _ = API_Image_OCR_EraX_VL_7B_vLLM(
            image_paths=img_path,
            is_base64=is_base64,
            prompt=prompt,
            erax_url_id=erax_url_id, 
            API_key=API_key,
            force_scale=True
        )
            
        output_text += f"** Nội dung của giấy tờ trong ảnh số {idx+1}**\n" + \
                           ocr_result.replace("```json", "").replace("```", "") +"\n\n"
                
    print ("--> Summarize result...")
    
    pdf_full_prompt_to_send =  pdf_full_prompt.replace("ocr_results", output_text)
    
    if pdf_full_prompt_to_send == pdf_full_prompt:
        pdf_full_prompt_to_send = f"{pdf_full_prompt}\n\n{output_text}"
        
    new_prompt = f"{pdf_full_prompt_to_send}"
    
    # print(new_prompt)
    

    # Chat w/ API to summarize all into 1
    try:
        final_result, history = API_Chat_OCR_EraX_VL_7B_vLLM(
            new_prompt, 
            history=None, 
            erax_url_id=erax_url_id, 
            API_key=API_key
        )
                                                             
    except Exception as E:
        print ("ERROR chatting w/ API: ", str(E))
        return new_prompt, None
    
    # print(final_result)
        
    # # Done
    # final_text =  final_result.replace("```json", "").replace("```", "")
    # try:
    #     final_text = json_repair.loads(final_text)
    # except:
    #     pass
        
    return final_result, history

def API_PDF_Full_OCR_EraX_VL_7B_vLLM(
    pdf_paths=None,
    is_base64=False,
    prompt=ycbt_ocr_only_prompt, 
    pdf_full_prompt=pdf_full_prompt, 
    erax_url_id=erax_url_id, 
    API_key=API_key
):
    """PDF Captioning ALL pages with PDF paths OR Base64
        - Bạn có thể dùng API này để parse PDF cả text & ảnh trong đó
        - Lưu ý prompt hợp lý theo đúng kiểu văn bản cần parse
        - API chỉ chấp nhận 1 PDF tại 1 thời điểm
        - API này kỳ vọng bạn truyền vào đường dẫn đến file PDF hoặc Base64
    """

    def getPDF_text(json_content):
        text = ""
        for data in json_content:
            text += data["text"]
            for img_text in data["images_text"]:
                text += "\n\n" + img_text["text"] 
        return text
    
    print ("Parsing PDF...")
    ocr_result, history = API_PDF_OCR_EraX_VL_7B_vLLM(
        pdf_paths=pdf_paths, 
        is_base64=is_base64,
        prompt=prompt,
        erax_url_id=erax_url_id, API_key=API_key
    )
                                                
    try:
        final_pdf = json_repair.loads(ocr_result)
        final_pdf =  str(getPDF_text(final_pdf["json_content"]).replace("```json", "").replace("```", "").replace("\n\n", "\n").replace("\n\n", "\n"))
    except Exception as E:
        print("ERROR wrong PDF output format!", str(E))
        return ocr_result, None
        
    print ("Summarize result...")
    pdf_full_prompt_to_send =  pdf_full_prompt.replace("ocr_results", final_pdf)
    
    new_prompt =  f"{pdf_full_prompt_to_send}"

    # Chat w/ API to summarize all into 1
    try:
        final_result, history = API_Chat_OCR_EraX_VL_7B_vLLM(
            new_prompt,
            history=None, 
            erax_url_id=erax_url_id, API_key=API_key
        ) 
                                                             
    except Exception as E:
        print ("ERROR chatting w/ API: ", str(E))
        return new_prompt, None
        
    # Done
    # final_pdf_text =  final_result.replace("```json", "").replace("```", "").replace("\n\n", "\n").replace("\n\n", "\n")  
    # try:
        # final_pdf_text = json_repair.loads(final_pdf_text)
    #     final_pdf_text = get_json(final_result)
    # except:
    #     pass
        
    return final_result, history  