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
    
    image_paths = ["./images/bao_hiem_0.jpg", "./images/bao_hiem_1.jpg"]
    
    prompt = """Hãy trích xuất toàn bộ chi tiết của các bức ảnh này theo đúng thứ tự của nội dung bằng định dạng json và không bình luận gì.
    """
    
    full_prompt = """Bạn là một chuyên gia bồi thường bảo hiểm xuất sắc.
Bạn được cung cấp danh sách các json là kết quả đã được OCR chính xác từ một (1) hay nhiều ảnh khác nhau.
Các json này là của một hay nhiều phiếu trong một bộ hồ sơ yêu cầu bồi thường bảo hiểm, bao gồm một hay nhiều hoá đơn các loại.

Bạn có 1 nhiệm vụ: phân tích và tổng hợp các jsons được cung cấp này:
- Phân tích kỹ lưỡng tất cả các json OCR được cung cấp, không đưỡ bỏ sót json nào.
- Tổng hợp tất cả các json trên bằng 1 json khác có tính tổng hợp nhưng đầy đủ chi tiết để công ty tiến hành xem xét thủ tục bồi thường chính xác và công bằng với định dạng json dưới đây.
- Không được bỏ qua bất kỳ chi tiết nào về các triệu chứng, các loại thuốc được kê mua, tên bệnh, đề xuất, các phí dịch vụ y tế và các chi phí khác.
- Không được bỏ qua bất kỳ nội dung nào, kể cả các ghi chú, điều kiện, uỷ quyền, cam kết
- Lưu ý phải tóm tắt ngắn gọn sau khi phân tích
- Các từ viết tắt thông dụng: NĐBH: Người được Bảo hiểm, CMND: Chứng minh nhân dân, GCNBH: Giấy chứng nhận Bảo hiểm, Số thẻ BH: Số thẻ Bảo hiểm, CCCD: Căn cước công dân, YCBT: Yêu cầu bồi thường, PK: Phòng khám, BVĐK: Bệnh viện Đa khoa, CTY: Công ty, BS: Bác sĩ, ThS: Thạc sĩ, STT: Số thứ tự, Mã BN: Mã bệnh nhân, GCN: Giấy chứng nhận, BS.CKII: Bác sĩ chuyên khoa 2, BHXH: Bảo hiểm Xã hội, ...
- Các họ phổ biến ở Việt Nam: NGUYỄN, Nguyễn, TRẦN, Trần, LÊ, Lê, ĐINH, Đinh, PHẠM, Phạm, TRỊNH, Trịnh, LÝ, Lý, HOÀNG, Hoàng, BÙI, Bùi, NGÔ, Ngô, PHAN, Phan, VÕ, Võ, HỒ, Hồ, HUỲNH, Huỳnh, TRƯƠNG, Trương, ĐẶNG, Đặng, ĐỖ, Đỗ, ...
- Các tỉnh, thành phố ở Việt Nam: An Giang, Bà Rịa-Vũng Tàu, Bắc Giang, Bắc Kạn, Bạc Liêu, Bắc Ninh, Bến Tre, Bình Định, Bình Dương, Bình Phước, Bình Thuận, Cà Mau, Cần Thơ, Cao Bằng, Đà Nẵng, Đắk Lắk, Đắk Nông, Điện Biên, Đồng Nai, Đồng Tháp, Gia Lai, Hà Giang, Hà Nam, Hà Nội, Hà Tĩnh, Hải Dương, Hải Phòng, Hậu Giang, TP. Hồ Chí Minh, Hòa Bình, Hưng Yên, Khánh Hòa, Kiên Giang, Kon Tum, Lai Châu, Lâm Đồng, Lạng Sơn, Lào Cai, Long An, Nam Định, Nghệ An, Ninh Bình, Ninh Thuận, Phú Thọ, Phú Yên, Quảng Bình, Quảng Nam, Quảng Ngãi, Quảng Ninh, Quảng Trị, Sóc Trăng, Sơn La, Tây Ninh, Thái Bình, Thái Nguyên, Thanh Hóa, Thừa Thiên - Huế, Tiền Giang, Trà Vinh, Tuyên Quang, Vĩnh Long, Vĩnh Phúc, Yên Bái.

# Dữ liệu được cung cấp:

{ocr_results}

Trả về định dạng json. Không diễn giải cách làm, không tóm tắt, chỉ trả lại duy nhất 1 json.
```

# Output:
    """

    result, history =  API_Multiple_Images_OCR_EraX_VL_7B_vLLM(
            image_paths=image_paths, 
            is_base64=False,
            prompt=prompt, 
            pdf_full_prompt=full_prompt,
            erax_url_id=ERAX_URL_ID, 
            API_key=API_KEY,
        )
    
    # Convert string json to json. It is result.
    json_result = get_json(result) 
    
    print(json_result)