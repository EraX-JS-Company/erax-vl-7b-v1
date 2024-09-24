#pip install json_repair numpy matplotlib tqdm pillow pymupdf

import os, base64, cv2, json_repair, requests, pymupdf
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from PIL import Image 
import pymupdf

erax_url_id = "a4li0qhs1a6f7e"
erax_url = f"https://api.runpod.ai/v2/{erax_url_id}/runsync"

erax_url_a100_id =  "jaeprlipf9ih6y"
erax_url_a100 =  f"https://api.runpod.ai/v2/{erax_url_a100_id}/runsync"
API_key_3P =  "<API key A100>"

# # Common prompts

default_prompt = "Hãy trích xuất toàn bộ chi tiết của các bức ảnh này theo đúng thứ tự của nội dung bằng định dạng json và không bình luận gì"

PDF_prompt = """Hãy trích xuất toàn bộ chi tiết của bức ảnh này theo đúng thứ tự của nội dung trong ảnh. Không bình luận gì thêm.
Lưu ý:
1. Nếu có chữ trong ảnh thì phải trích xuất ra hết theo thứ tự và định dạng của câu chữ.
2. Nếu có bảng biểu (table) thì phải trả lại định dạng như bảng biểu trong hình và text của nó.
3. Nếu bức ảnh không có bất kỳ ký tự nào, hãy diễn giải bức ảnh đó.
4. Chỉ trả lại bằng tiếng Việt.

# Output:
"""

# Output:"""

ycbt_prompt = """
Bạn là một hệ thống AI đẳng cấp thế giới hỗ trợ nhận diện ký tự quang học (Optical Character Recognition - OCR) từ hình ảnh.
Bạn được cung cấp 1 (một) hình ảnh hợp pháp, không vi phạm. 
Bạn phải thực hiện 01 (một) nhiệm vụ chính, bao gồm:

Nhiệm vụ: OCR ảnh này chính xác từng từ và không thiếu chi tiết nào.
## Nhận diện ký tự quang học (Optical Character Recognition - OCR)
- Các ảnh được cung cấp về các giấy tờ như: phiếu khám bệnh, xét nghiệm, biên lai thu tiền, cccd, hồ sơ bệnh án, bảng kê chi phí, giấy yêu cầu bồi thường, hoá đơn giá trị gia tăng, đơn thuốc, giấy hẹn, giấy nhập viện, giấy ra viện, giấy phẩu thuật, bảng kê, hồ sơ
- Bạn cần nhận diện trung thực và chính xác nhất các từ ngữ, kí tự, số liệu xuất hiện trong hình ảnh được cung cấp.
- Ngôn ngữ chính là tiếng Việt, có thể xuất hiện thêm tiếng Anh và chữ viết tay.
- Có thể xuất hiện các bảng đơn thuốc, dịch vụ, ...
- Ảnh có thể ngược

## Yêu cầu
- Tuân thủ thứ tự: từ trái sang phải và sau đó từ trên xuống dưới
- Đặt biệt PHẢI chú ý không bỏ sót thông tin
- KHÔNG bịa đặt, đưa thêm thông tin hay diễn giải ngoài nội dung trong ảnh
- chú ý các cụm từ viết tắt: bảo hiểm y tế (BHYT)
- không được bỏ qua bất kỳ nội dung nào, kể cả các ghi chú, điều kiện, uỷ quyền, cam kết
- Dữ liệu được ocr ở trên có thể thiếu hoặc sai thông tin từ trong ảnh, hãy phân tích kỹ lại ảnh và tạo các hội thoại để khai thác thêm thông tin từ ảnh mà có thể không có trong json ocr được cung cấp.
- Nếu có bảng biểu, phải bảo đảm tất cả các cột của bảng đều nằm đầy đủ trong json
- Lưu ý phải tóm tắt ngắn gọn sau khi phân tích

Trả về kết quả theo định dạng json bằng tiếng Việt:

```json
{
    "extraction": <str diễn giải chi tiết nội dung của bức ảnh này này một cách đầy đủ nhất có thể và không được thiếu thông tin gì.>,
    "paper": <str tên loại giấy tờ của bức ảnh này. Nếu không có thì để None. >,
    "customer": { các thông tin của bức ảnh này về khách hàng nếu có. Nếu không có thì để None },
    "status": [<liệt kê các bệnh sử y khoa của bệnh nhân đã khai báo nếu có>, ...],
    "reasons": <diễn giải càng chi tiết càng tốt lý do bệnh nhân phải sử dụng dịch vụ khám, chữa bệnh, xét nghiệm, phẫu thuật hay mua các sản phẩm trong ảnh. Nếu không có thì để None>,
    "results": [<liệt kê chi tiết các kết quả xét nghiệm, bao gồm tên xủa xét nghiệm hay thủ thuật y tế và kết quả liên quan, là các chỉ số y khoa trong khoảng tham chiếu và ngưỡng cần phải lưu ý. Có thể bao gồm: chỉ định, kết quả, đơn vị, khoảng tham chiếu, quy trình, thiết bị, các lưu ý...> , ... Nếu không có thì để None ],
    "conclusions": { các kết luận của các y sỹ, bác sỹ trong bức ảnh này nếu có. Nếu không có thì để None },
    "supplier": { các thông tin của bức ảnh này về nhà cung cấp, bệnh viện, phòng khác nếu có... Nếu không có thì để None. },
    "doctors": { các thông tin của bức ảnh này về bác sỹ, y sỹ tham gia điều trị, xét nghiệm, đánh giá bệnh hay kê đơn,... Nếu không có thì để None. },
    "medicines": [ liệt kê chính xác thông tin của bức ảnh này về tất cả các loại thuốc được kê đơn và thông tin y tế của chúng... Nếu không có thì để None 
             {
                "medicine": <tên thuốc. Diễn giải đầy đủ, chính xác tên thuốc và các đặc điểm của thuốc như trong ảnh>,
                "unit of dosage": <liều lượng sử dụng của thuốc này. Ví dụ: viên, ống, bút, UI, cái, hộp... Nếu không có thì để None>,
                "frequency and dosage": ["sáng": <liều lượng. Ví dụ: 1 viên, 2 ống, 5 bút, 20 UI...>, "chiều": <liều lượng. Ví dụ: 1 viên, 2 ống, 5 bút, 20 UI...>,],
                "unit of purchase": <đơn vị tính của lượng thuốc này khi mua, thường nằm ở cột có tên là 'Đơn vị'. Ví dụ: viên, ống, bút, UI, cái, hộp... Nếu không có thì để None>,
                "quantity purchased": <số lượng đơn vị được kê đơn hay mua, thường nằm ở cột có tên là 'Số lượng'. Ví dụ: 1, 2, 10, 20, 42... Nếu không có thì để None>,
                "nota bene": <các lưu ý quan trọng về việc bảo quản hay trước hoặc sau khi sử dụng thuốc này.>
            }
        ],
    "products and services": { các thông tin của bức ảnh về các sản phẩm hay dịch vụ được cung cấp như khám, tái khám, nội trú, xét nghiệm, chụp chiếu, giải phẫu, công cụ, cung cấp thuốc y tế, đơn giá, thành tiền, được BHYT trả bao nhiêu... Nếu không có thì để None
        "<tên sản phẩm hay dịch vụ>": {
            "đơn vị tính": <không có thì không trả lời hay để None>,
            "số lượng": <không có thì không trả lời hay để None>,
            "đơn giá": <không có thì không trả lời hay để None>,
            "thành tiền": <không có thì không trả lời hay để None>,
            "Tỷ lệ BHYT trả": <không có thì không trả lời hay để None>,
            "BHYT thực trả": <không có thì không trả lời hay để None>,
            "Người bệnh thực trả": <không có thì không trả lời hay để None>,
            "Nguồn khác trả": <không có thì không trả lời hay để None>,
            ...các thông tin khác về sản phẩm hay dịch vụ
        }
    "total amount": <tổng cố tiền phải trả nếu có trong của bức ảnh này. Nếu không có thì để None>,
    "others": { các thông tin khác trong bức ảnh này như các lưu ý, ghi chú, điều kiện, uỷ quyền, cam kết của cả người mua và người bán... Nếu không có thì để None },
    "summary": <str tóm tắt hồ sơ này và các điểm nhấn quan trọng>,
    ... tất cả thông tin khác nếu có...
}
```

## Output:
"""

pdf_full_prompt = """
Bạn là một chuyên gia bồi thường bảo hiểm xuất sắc.
Bạn được cung cấp danh sách các json là kết quả đã được OCR chính xác từ một (1) hay nhiều ảnh khác nhau.
Các json này là của một hay nhiều phiếu trong một bộ hồ sơ yêu cầu bồi thường bảo hiểm, bao gồm một hay nhiều hoá đơn các loại.

Bạn có 1 nhiệm vụ: phân tích và tổng hợp các jsons được cung cấp này:
- Phân tích kỹ lưỡng tất cả các json OCR được cung cấp, không đưỡ bỏ sót json nào.
- Tổng hợp tất cả các json trên bằng 1 json khác có tính tổng hợp nhưng đầy đủ chi tiết để công ty tiến hành xem xét thủ tục bồi thường chính xác và công bằng với định dạng json dưới đây.
- Không được bỏ qua bất kỳ chi tiết nào về các triệu chứng, các loại thuốc được kê mua, tên bệnh, đề xuất, các phí dịch vụ y tế và các chi phí khác.
- Không được bỏ qua bất kỳ nội dung nào, kể cả các ghi chú, điều kiện, uỷ quyền, cam kết
- Lưu ý phải tóm tắt ngắn gọn sau khi phân tích

# Dữ liệu được cung cấp:

{ocr_results}

Trả về định dạng json đa văn bản như sau. Không diễn giải cách làm, không tóm tắt, chỉ trả lại duy nhất 1 json như sau:

```json
{
    "<str tên loại giấy tờ được cung cấp trong các json trên>":
    {
        "extraction": <str diễn giải nội dung của giấy tờ này này một cách đầy đủ nhất có thể và không được thiếu thông tin gì.>,
        "customer": { các thông tin của bức ảnh này về khách hàng nếu có. Nếu không có thì để None },
        "status": [<liệt kê các bệnh sử y khoa của bệnh nhân đã khai báo nếu có>, ...],
        "reasons": <diễn giải càng chi tiết càng tốt lý do bệnh nhân phải sử dụng dịch vụ khám, chữa bệnh, xét nghiệm, phẫu thuật hay mua các sản phẩm trong ảnh. Nếu không có thì để None>,
        "results": [<liệt kê chi tiết các kết quả xét nghiệm, bao gồm tên xủa xét nghiệm hay thủ thuật y tế và kết quả liên quan, là các chỉ số y khoa trong khoảng tham chiếu và ngưỡng cần phải lưu ý. Có thể bao gồm: chỉ định, kết quả, đơn vị, khoảng tham chiếu, quy trình, thiết bị, các lưu ý...> , ... Nếu không có thì để None ],
        "conclusions": { các kết luận của các y sỹ, bác sỹ trong bức ảnh này nếu có. Nếu không có thì để None },
        "supplier": { các thông tin của bức ảnh này về nhà cung cấp, bệnh viện, phòng khác nếu có... Nếu không có thì để None. },
        "doctors": { các thông tin của bức ảnh này về bác sỹ, y sỹ tham gia điều trị, xét nghiệm, đánh giá bệnh hay kê đơn,... Nếu không có thì để None. },
        "medicines": [ liệt kê chính xác thông tin của bức ảnh này về tất cả các loại thuốc được kê đơn và thông tin y tế của chúng... Nếu không có thì để None 
                 {
                    "medicine": <tên thuốc. Diễn giải đầy đủ, chính xác tên thuốc và các đặc điểm của thuốc như trong ảnh>,
                    "unit of dosage": <liều lượng sử dụng của thuốc này. Ví dụ: viên, ống, bút, UI, cái, hộp... Nếu không có thì để None>,
                    "frequency and dosage": ["sáng": <liều lượng. Ví dụ: 1 viên, 2 ống, 5 bút, 20 UI...>, "chiều": <liều lượng. Ví dụ: 1 viên, 2 ống, 5 bút, 20 UI...>,],
                    "unit of purchase": <đơn vị tính của lượng thuốc này khi mua, thường nằm ở cột có tên là 'Đơn vị'. Ví dụ: viên, ống, bút, UI, cái, hộp... Nếu không có thì để None>,
                    "quantity purchased": <số lượng đơn vị được kê đơn hay mua, thường nằm ở cột có tên là 'Số lượng'. Ví dụ: 1, 2, 10, 20, 42... Nếu không có thì để None>,
                    "nota bene": <các lưu ý quan trọng về việc bảo quản hay trước hoặc sau khi sử dụng thuốc này.>
                }
            ],
        "products and services": { các thông tin của bức ảnh về các sản phẩm hay dịch vụ được cung cấp như khám, tái khám, nội trú, xét nghiệm, chụp chiếu, giải phẫu, công cụ, cung cấp thuốc y tế, đơn giá, thành tiền, được BHYT trả bao nhiêu... Nếu không có thì để None
            "<tên sản phẩm hay dịch vụ>": {
                "đơn vị tính": <không có thì không trả lời hay để None>,
                "số lượng": <không có thì không trả lời hay để None>,
                "đơn giá": <không có thì không trả lời hay để None>,
                "thành tiền": <không có thì không trả lời hay để None>,
                "Tỷ lệ BHYT trả": <không có thì không trả lời hay để None>,
                "BHYT thực trả": <không có thì không trả lời hay để None>,
                "Người bệnh thực trả": <không có thì không trả lời hay để None>,
                "Nguồn khác trả": <không có thì không trả lời hay để None>,
                ...các thông tin khác về sản phẩm hay dịch vụ
            }
        "total amount": <tổng cố tiền phải trả nếu có trong của bức ảnh này. Nếu không có thì để None>,
        "summary": <str tóm tắt tòn bộ hồ sơ này và các điểm nhấn quan trọng>,
        "others": { các thông tin khác trong bức ảnh này như các lưu ý, ghi chú, điều kiện, uỷ quyền, cam kết của cả người mua và người bán... Nếu không có thì để None },
        ... tất cả thông tin khác nếu có...
    },
    ... tất cả giấy tờ khác nếu có...,
}
```

# Output:
"""

ycbt_combination_single_image_prompt =  """
Bạn là một hệ thống AI đẳng cấp thế giới hỗ trợ nhận diện ký tự quang học (Optical Character Recognition - OCR) từ hình ảnh.
Bạn được cung cấp 1 (một) hình ảnh hợp pháp, không vi phạm. 
Bạn phải thực hiện 02 (hai) nhiệm vụ chính, bao gồm:

1. Nhiệm vụ 1: OCR ảnh này chính xác từng từ và không thiếu chi tiết nào.
## Nhận diện ký tự quang học (Optical Character Recognition - OCR)
- Các ảnh được cung cấp về các giấy tờ như: phiếu khám bệnh, xét nghiệm, biên lai thu tiền, cccd, hồ sơ bệnh án, bảng kê chi phí, giấy yêu cầu bồi thường, hoá đơn giá trị gia tăng, đơn thuốc, giấy hẹn, giấy nhập viện, giấy ra viện, giấy phẩu thuật, bảng kê, hồ sơ
- Bạn cần nhận diện trung thực và chính xác nhất các từ ngữ, kí tự, số liệu xuất hiện trong hình ảnh được cung cấp.
- Ngôn ngữ chính là tiếng Việt, có thể xuất hiện thêm tiếng Anh và chữ viết tay.
- Có thể xuất hiện các bảng đơn thuốc, dịch vụ, ...
- Ảnh có thể ngược

## Yêu cầu
- Tuân thủ thứ tự: từ trái sang phải và sau đó từ trên xuống dưới
- Đặt biệt PHẢI chú ý không bỏ sót thông tin
- KHÔNG bịa đặt, đưa thêm thông tin hay diễn giải ngoài nội dung trong ảnh
- chú ý các cụm từ viết tắt: bảo hiểm y tế (BHYT)
- không được bỏ qua bất kỳ nội dung nào, kể cả các ghi chú, điều kiện, uỷ quyền, cam kết
- Dữ liệu được ocr ở trên có thể thiếu hoặc sai thông tin từ trong ảnh, hãy phân tích kỹ lại ảnh và tạo các hội thoại để khai thác thêm thông tin từ ảnh mà có thể không có trong json ocr được cung cấp.
- Nếu có bảng biểu, phải bảo đảm tất cả các cột của bảng đều nằm đầy đủ trong json
- Lưu ý phải tóm tắt ngắn gọn sau khi phân tích

2. Nhiệm vụ 2:
## Tạo hội thoại dựa vào thông tin được OCR:
- Hãy tạo ra hội thoại (conversations) 15 lượt giữa Human đang thắc mắc và assistant để khai thác tất cả các chi tiết bao gồm: thông tin cá nhân, thông tin về sự kiện bảo hiểm, thông tin về người yêu cầu bồi thường, các chứng từ đính kèm, hình thức bồi thường, các cam kết.
- Không được bỏ qua bất kỳ chi tiết nào.
- human không nhớ gì và sẽ đặt ra các câu hỏi khó, dài, phức tạp, thử thách assistant để khai thác thông tin theo phương pháp lý luận từng bước (step-by-step and chains-of-thought)
- human không nhớ số liệu hay thông tin gì và nhờ assistant trả lời theo phương pháp lý luận từng bước (step-by-step and chains-of-thought)
- assistant phải liên kết, suy luận theo phương pháp lý luận từng bước (step-by-step and chains-of-thought) từ nhiều thông tin để đưa ra câu trả lời
- Câu trả lời phải theo phương pháp lý luận từng bước (step-by-step and chains-of-thought)
- Trong một câu hỏi có thể có 1 hoặc nhiều câu hỏi nhỏ và assistant phải trả lời từng câu rõ ràng theo phương pháp lý luận từng bước (step-by-step and chains-of-thought) từ nhiều thông tin để đưa ra câu trả lời
- Từ câu thứ 10 trở đi, các câu hỏi của human theo phương pháp lý luận từng bước (step-by-step and chains-of-thought), khó tìm thậm chí là các câu hỏi không thể tìm thấy câu trả lời từ json được cung cấp.

**Yêu cầu đối với hội thoại**
- Hỏi đáp trực tiếp trên thông tin, không chào hỏi vòng vo
- Mỗi câu hỏi đến từ các thông tin khác nhau
- assistant chỉ được phép trả lời và không hỏi hay yêu cầu human
- PHẢI nghiêm túc trả lời theo trọng tâm nội dung
- Đoạn hội thoại PHẢI bám sát các nội dung quan trọng của bài viết
- Kết thúc hội thoại phải là câu trả lời của assistant. 
- KHÔNG tạm biệt, cảm ơn, nhờ vả, khen ngợi, ...
- Cả human và assistant không được nói đến sự tồn tại của json và các key của json này.
- Dữ liệu được ocr ở trên có thể thiếu hoặc sai thông tin từ trong ảnh, hãy phân tích kỹ lại ảnh và tạo các hội thoại để khai thác thêm thông tin từ ảnh mà có thể không có trong json ocr được cung cấp.

Yêu cầu trả lại định dạng chính xác như sau với kết qủa OCR và 15 multi-turns chat bằng tiếng Việt:

Trả về kết quả theo định dạng json bằng tiếng Việt:

```json
{
     "extraction": <str diễn giải nội dung của bức ảnh này một cách đầy đủ nhất có thể và không được thiếu thông tin gì.>,
     "paper": <str tên loại giấy tờ của bức ảnh này. Nếu không có thì để None. >,
     "customer": { các thông tin của bức ảnh này về khách hàng nếu có. Nếu không có thì để None },
     "status": [<liệt kê các bệnh sử y khoa của bệnh nhân đã khai báo nếu có>, ...],
     "reasons": <diễn giải càng chi tiết càng tốt lý do bệnh nhân phải sử dụng dịch vụ khám, chữa bệnh, xét nghiệm, phẫu thuật hay mua các sản phẩm trong ảnh. Nếu không có thì để None>,
     "results": [<liệt kê chi tiết các kết quả xét nghiệm, bao gồm tên xủa xét nghiệm hay thủ thuật y tế và kết quả liên quan, là các chỉ số y khoa trong khoảng tham chiếu và ngưỡng cần phải lưu ý. Có thể bao gồm: chỉ định, kết quả, đơn vị, khoảng tham chiếu, quy trình, thiết bị, các lưu ý...> , ... Nếu không có thì để None ],
     "conclusions": { các kết luận của các y sỹ, bác sỹ trong bức ảnh này nếu có. Nếu không có thì để None },
     "supplier": { các thông tin của bức ảnh này về nhà cung cấp, bệnh viện, phòng khác nếu có... Nếu không có thì để None. },
     "doctors": { các thông tin của bức ảnh này về bác sỹ, y sỹ tham gia điều trị, xét nghiệm, đánh giá bệnh hay kê đơn,... Nếu không có thì để None. },
     "medicines": [ liệt kê chính xác thông tin của bức ảnh này về tất cả các loại thuốc được kê đơn và thông tin y tế của chúng... Nếu không có thì để None 
         {
            "medicine": <tên thuốc. Diễn giải đầy đủ, chính xác tên thuốc và các đặc điểm của thuốc như trong ảnh>,
            "unit of dosage": <liều lượng sử dụng của thuốc này. Ví dụ: viên, ống, bút, UI, cái, hộp... Nếu không có thì để None>,
            "frequency and dosage": ["sáng": <liều lượng. Ví dụ: 1 viên, 2 ống, 5 bút, 20 UI...>, "chiều": <liều lượng. Ví dụ: 1 viên, 2 ống, 5 bút, 20 UI...>,],
            "unit of purchase": <đơn vị tính của lượng thuốc này khi mua, thường nằm ở cột có tên là 'Đơn vị'. Ví dụ: viên, ống, bút, UI, cái, hộp... Nếu không có thì để None>,
            "quantity purchased": <số lượng đơn vị được kê đơn hay mua, thường nằm ở cột có tên là 'Số lượng'. Ví dụ: 1, 2, 10, 20, 42... Nếu không có thì để None>,
            "nota bene": <các lưu ý quan trọng về việc bảo quản hay trước hoặc sau khi sử dụng thuốc này.>
        }
    ],
     "products and services": { các thông tin của bức ảnh về các sản phẩm hay dịch vụ được cung cấp như khám, tái khám, nội trú, xét nghiệm, chụp chiếu, giải phẫu, công cụ, cung cấp thuốc y tế, đơn giá, thành tiền, được BHYT trả bao nhiêu... Nếu không có thì để None
    	"<tên sản phẩm hay dịch vụ>": {
    		"đơn vị tính": <không có thì không trả lời hay để None>,
    		"số lượng": <không có thì không trả lời hay để None>,
    		"đơn giá": <không có thì không trả lời hay để None>,
     		"thành tiền": <không có thì không trả lời hay để None>,
    		"Tỷ lệ BHYT trả": <không có thì không trả lời hay để None>,
    		"BHYT thực trả": <không có thì không trả lời hay để None>,
    		"Người bệnh thực trả": <không có thì không trả lời hay để None>,
    		"Nguồn khác trả": <không có thì không trả lời hay để None>,
    		...các thông tin khác về sản phẩm hay dịch vụ
     },
     "total amount": <tổng cố tiền phải trả nếu có trong của bức ảnh này. Nếu không có thì để None>,
     "others": { các thông tin khác trong bức ảnh này như các lưu ý, ghi chú, điều kiện, uỷ quyền, cam kết của cả người mua và người bán... Nếu không có thì để None },
     "summary": <str tóm tắt hồ sơ này và các điểm nhấn quan trọng>,
     ... tất cả thông tin khác nếu có...,
    "conversations": [
         {
          "role": "human"
          "text": : <str hội thoại của human
         },
         {
          "role": "assistant"
          "text": <str hội thoại của assistant>
         },
          ...  
     ]
}
```

## Output:
"""

# # Thư viện
# - Bạn có thể dùng thư viện Python này hoặc chuyển sang ngôn ngữ tương ứng dễ dàng

max_allowed_images = 5
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
def add_img_content(image_paths=None, 
                    is_base64=False, 
                    prompt=default_prompt, 
                    tmp_path="./tmp/", 
                    max_images=max_allowed_images, 
                    force_scale=True):

    import uuid
    os.makedirs(tmp_path, exist_ok=True)

    max_width_mm =  448
    scaled = False
    
    if image_paths is not None:
        
        if type(image_paths)==str:
            image_paths = [image_paths]
        else:
            scaled = True
            
        if scaled or force_scale:
            scaled = True
            print (f"Có {len(image_paths)} ảnh.")
            # Multi images --> scale all images to max_width_mm
            img_path_new = []
            for img_path in image_paths[:max_images]:

                # Use PIL instead of OpenCV
                if not is_base64:
                    img = Image.open(img_path)
                else:
                    img =  openBase64_Image(img_path)
                    
                w, h = img.size
                ratio = w/h
                w =  max_width_mm
                h = int(w/ratio)
                img = img.resize((w, h))
                file_name =  tmp_path + str(uuid.uuid4())+".jpg"
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
            dump_name = str(uuid.uuid4())
            with open(f"{dump_name}.pdf", "wb") as f:
                f.write(base64.b64decode(pdf_base64))
            doc = pymupdf.open(f"{dump_name}.pdf")
            os.remove(f"{dump_name}.pdf")
            
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


# # RunPod long run
# - Lưu ý cho PDF nhiều trang hay nhiều ảnh, API sẽ trả về IN_PROGRESS
# - Dùng /status để check progress và lấy về
def checkStatusLongRun(ocr_result, erax_url_id=erax_url, API_key=API_key_3P):
    import time
    
    final_result =  ocr_result.copy()
    while True:
        time.sleep(0.5)
        if type(final_result)==dict:
            if "status" in final_result.keys() and (final_result["status"]=="IN_PROGRESS" or final_result["status"]=="IN_QUEUE"):
                job_id    =  final_result["id"]
                print(f"Check status & result...{job_id}")
                runpod_status_url = f"https://api.runpod.ai/v2/{erax_url_id}/status/{job_id}"
                head = {}
                head["authorization"] = API_key
                final_result = requests.post(runpod_status_url, headers=head, timeout=120).json()
            else:
                break
        else:        
            break
            
    return final_result


# # Image captioning w/ list of [images paths or base64]
# - Bạn có thể dùng API này để captioning ảnh
# - Lưu ý prompt hợp lý theo đúng kiểu văn bản cần parse
# - API chỉ chấp nhận tối đa 20 ảnh nhưng bạn nên captioning tối đa 3 ảnh
# - API này kỳ vọng bạn truyền vào list các base64 thuần của ảnh
# - Lưu ý prefix: API đã thêm "data:image;base64" trước decoded {base64} của ảnh rồi
def API_Image_OCR_EraX_VL_7B_vLLM(image_paths=None,
                                  is_base64=True, 
                                  prompt=default_prompt, 
                                  erax_url_id=erax_url_id, 
                                  API_key=API_key_3P, 
                                  force_scale=True):

    messages = add_img_content(image_paths, is_base64=is_base64, prompt=prompt, force_scale =force_scale)
    
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


# # PDF captioning w/ PDF paths or base64
# - Bạn có thể dùng API này để parse PDF cả text & ảnh trong đó
# - Lưu ý prompt hợp lý theo đúng kiểu văn bản cần parse
# - API chỉ chấp nhận 1 PDF tại 1 thời điểm
# - API này kỳ vọng bạn truyền vào đường dẫn đến file PDF or list of PDF's base64
def API_PDF_OCR_EraX_VL_7B_vLLM(pdf_paths=None, 
                                is_base64=False,
                                prompt=PDF_prompt, 
                                erax_url_id=erax_url_id, 
                                API_key=API_key_3P):

    messages = add_pdf_content_json(pdf_paths, prompt=prompt, is_base64=is_base64)
    
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

# # Chat with the previous result from EraX
# - Bạn có thể hội thoại liên tục với kết quả EraX đã captioning lần trước hoặc đơn giản là chat với QWen2
def API_Chat_OCR_EraX_VL_7B_vLLM(prompt, history=None, erax_url_id=erax_url_id, API_key=API_key_3P):

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
            "temperature": float(0.2),
            "top_p": float(0.95),
            "top_k": int(10),
            "repetition_penalty": float(1.1),
            "max_tokens": 32000
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
                                
    content["messages"].append({
            "role": "assistant",
            "content": result
    }
    )
    
    return result, content


# # PDF captioning ALL pages w/ PDF paths OR Base64
# - Bạn có thể dùng API này để parse PDF cả text & ảnh trong đó
# - Lưu ý prompt hợp lý theo đúng kiểu văn bản cần parse
# - API chỉ chấp nhận 1 PDF tại 1 thời điểm
# - API này kỳ vọng bạn truyền vào đường dẫn đến file PDF hoặc Base64

def API_PDF_Full_OCR_EraX_VL_7B_vLLM(pdf_paths=None,
                                       is_base64=False,
                                       prompt=ycbt_prompt, 
                                       pdf_full_prompt=pdf_full_prompt, 
                                       erax_url_id=erax_url_a100_id, 
                                       API_key=API_key_3P):

    def getPDF_text(json_content):
        text = ""
        for data in json_content:
            text += data["text"]
            for img_text in data["images_text"]:
                text += "\n\n" + img_text["text"] 
        return text
    
    print ("Parsing PDF...")
    ocr_result, history = API_PDF_OCR_EraX_VL_7B_vLLM(pdf_paths=pdf_paths, 
                                                is_base64=is_base64,
                                                prompt=prompt,
                                                erax_url_id=erax_url_id, API_key=API_key)
        
    try:
        final_pdf = json_repair.loads(ocr_result)
        final_pdf =  str(getPDF_text(final_pdf["json_content"]).replace("```json", "").replace("```", "").replace("\n\n", "\n").replace("\n\n", "\n"))
    except Exception as E:
        print("ERROR wrong PDF output format!", str(E))
        return ocr_result, None
        
    print ("Summarize result...")
    pdf_full_prompt_to_send =  pdf_full_prompt.replace("ocr_results", final_pdf)
    
    new_prompt =  f"{pdf_full_prompt_to_send}"

    print (new_prompt)
    
    # Chat w/ API to summarize all into 1
    try:
        final_result, history = API_Chat_OCR_EraX_VL_7B_vLLM(new_prompt, 
                                                             history=None, 
                                                             erax_url_id=erax_url_id, API_key=API_key_3P)
    except Exception as E:
        print ("ERROR chatting w/ API: ", str(E))
        return new_prompt, None
        
    # Done
    final_pdf_text =  final_result.replace("```json", "").replace("```", "").replace("\n\n", "\n").replace("\n\n", "\n")  
    try:
        final_pdf_text = json_repair.loads(final_pdf_text)
    except:
        pass
        
    return final_pdf_text, history  

# # Captioning multiple images w/ paths OR Base64
# - Bạn có thể dùng API này để parse multiple images cả text & ảnh trong đó
# - Lưu ý prompt hợp lý theo đúng kiểu văn bản cần parse
# 
def API_Multiple_Images_OCR_EraX_VL_7B_vLLM(image_paths=None,
                                       is_base64=False,
                                       prompt=ycbt_prompt, 
                                       pdf_full_prompt=pdf_full_prompt, 
                                       erax_url_id=erax_url_a100_id, 
                                       API_key=API_key_3P):

    print ("--> Parsing all images...")

    output_text = ""
    for idx, img_path in enumerate(image_paths):
        print (f"- Parsing image...{idx}")
        add_img_content(img_path, )
        
        ocr_result, _ = API_Image_OCR_EraX_VL_7B_vLLM(image_paths=img_path,
                                                      is_base64=is_base64,
                                                      prompt=prompt,
                                                      erax_url_id=erax_url_id, 
                                                      API_key=API_key)
            
        output_text += f"** Nội dung của giấy tờ trong ảnh số {idx+1}**\n" + \
                           ocr_result.replace("```json", "").replace("```", "") +"\n\n"
                
    print ("--> Summarize result...")
    
    pdf_full_prompt_to_send =  pdf_full_prompt.replace("ocr_results", final_pdf)
    new_prompt =  f"{pdf_full_prompt_to_send}"

    print (new_prompt)
    
    # Chat w/ API to summarize all into 1
    try:
        final_result, history = API_Chat_OCR_EraX_VL_7B_vLLM(new_prompt, 
                                                             history=None, 
                                                             erax_url_id=erax_url_id, API_key=API_key_3P)
    except Exception as E:
        print ("ERROR chatting w/ API: ", str(E))
        return new_prompt, None
        
    # Done
    final_text =  final_result.replace("```json", "").replace("```", "")
    try:
        final_text = json_repair.loads(final_text)
    except:
        pass
        
    return final_text, history
