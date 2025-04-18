# from transformers import AutoTokenizer, AutoModel
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import re

# #Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# # Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
# model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')

# # Your JSON data
# data = [
#     {
#         "id": "chunk_2_13",
#         "chuong": "chương 2. hành vi vi phạm, hình thức, mức xử phạt, mức trừ điểm giấy phép lái xe và biện pháp khắc phục hậu quả vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ",
#         "muc": "mục 1. vi phạm quy tắc giao thông đường bộ",
#         "dieu": "điều 6. xử phạt, trừ điểm giấy phép lái xe của người điều khiển xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô vi phạm quy tắc giao thông đường bộ",
#         "khoan": "khoản 13. phạt tiền từ 50.000.000 đồng đến 70.000.000 đồng đối với người điều khiển xe thực hiện hành vi vi phạm quy định tại khoản 12 điều này mà gây tai nạn giao thông.",
#         "content": "khoản 13. phạt tiền từ 50.000.000 đồng đến 70.000.000 đồng đối với người điều khiển xe thực hiện hành vi vi phạm quy định tại khoản 12 điều này mà gây tai nạn giao thông.",
#         "reference_id": "2.1.6.13"
#     },
#     {
#         "id": "chunk_2_14",
#         "chuong": "chương 2. hành vi vi phạm, hình thức, mức xử phạt, mức trừ điểm giấy phép lái xe và biện pháp khắc phục hậu quả vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ",
#         "muc": "mục 1. vi phạm quy tắc giao thông đường bộ",
#         "dieu": "điều 6. xử phạt, trừ điểm giấy phép lái xe của người điều khiển xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô vi phạm quy tắc giao thông đường bộ",
#         "khoan": "khoản 14. tịch thu phương tiện đối với người điều khiển xe tái phạm hành vi điều khiển xe lạng lách, đánh võng quy định tại khoản 12 điều này.",
#         "content": "khoản 14. tịch thu phương tiện đối với người điều khiển xe tái phạm hành vi điều khiển xe lạng lách, đánh võng quy định tại khoản 12 điều này.",
#         "reference_id": "2.1.6.14"
#     },
#     {
#         "id": "chunk_2_15",
#         "chuong": "chương 2. hành vi vi phạm, hình thức, mức xử phạt, mức trừ điểm giấy phép lái xe và biện pháp khắc phục hậu quả vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ",
#         "muc": "mục 1. vi phạm quy tắc giao thông đường bộ",
#         "dieu": "điều 6. xử phạt, trừ điểm giấy phép lái xe của người điều khiển xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô vi phạm quy tắc giao thông đường bộ",
#         "khoan": "khoản 15. ngoài việc bị phạt tiền, người điều khiển xe thực hiện hành vi vi phạm còn bị áp dụng các hình thức xử phạt bổ sung sau đây:",
#         "content": "khoản 15. ngoài việc bị phạt tiền, người điều khiển xe thực hiện hành vi vi phạm còn bị áp dụng các hình thức xử phạt bổ sung sau đây:\na) thực hiện hành vi quy định tại điểm e khoản 5 điều này còn bị tịch thu thiết bị phát tín hiệu ưu tiên lắp đặt, sử dụng trái quy định;\nb) thực hiện hành vi quy định tại khoản 12 điều này bị tước quyền sử dụng giấy phép lái xe từ 10 tháng đến 12 tháng;\nc) thực hiện hành vi quy định tại điểm a, điểm b, điểm c, điểm d khoản 11; khoản 13; khoản 14 điều này bị tước quyền sử dụng giấy phép lái xe từ 22 tháng đến 24 tháng.",
#         "reference_id": "2.1.6.15"
#     },
#     {
#         "id": "chunk_2_16",
#         "chuong": "chương 2. hành vi vi phạm, hình thức, mức xử phạt, mức trừ điểm giấy phép lái xe và biện pháp khắc phục hậu quả vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ",
#         "muc": "mục 1. vi phạm quy tắc giao thông đường bộ",
#         "dieu": "điều 6. xử phạt, trừ điểm giấy phép lái xe của người điều khiển xe ô tô, xe chở người bốn bánh có gắn động cơ, xe chở hàng bốn bánh có gắn động cơ và các loại xe tương tự xe ô tô vi phạm quy tắc giao thông đường bộ",
#         "khoan": "khoản 16. ngoài việc bị áp dụng hình thức xử phạt, người điều khiển xe thực hiện hành vi vi phạm còn bị trừ điểm giấy phép lái xe như sau:",
#         "content": "khoản 16. ngoài việc bị áp dụng hình thức xử phạt, người điều khiển xe thực hiện hành vi vi phạm còn bị trừ điểm giấy phép lái xe như sau:\na) thực hiện hành vi quy định tại điểm h, điểm i khoản 3; điểm a, điểm b, điểm c, điểm d, điểm đ, điểm g khoản 4; điểm a, điểm b, điểm c, điểm d, điểm đ, điểm e, điểm g, điểm i, điểm k, điểm n, điểm o khoản 5 điều này bị trừ điểm giấy phép lái xe 02 điểm;\nb) thực hiện hành vi quy định tại điểm h khoản 5; khoản 6; điểm b khoản 7; điểm b, điểm c, điểm d khoản 9 điều này bị trừ điểm giấy phép lái xe 04 điểm;\nc) thực hiện hành vi quy định tại điểm p khoản 5; điểm a, điểm c khoản 7; khoản 8 điều này bị trừ điểm giấy phép lái xe 06 điểm;\nd) thực hiện hành vi quy định tại điểm a khoản 9, khoản 10, điểm đ khoản 11 điều này bị trừ điểm giấy phép lái xe 10 điểm.",
#         "reference_id": "2.1.6.16"
#     },
#     {
#         "id": "chunk_2_17",
#         "chuong": "chương 2. hành vi vi phạm, hình thức, mức xử phạt, mức trừ điểm giấy phép lái xe và biện pháp khắc phục hậu quả vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ",
#         "muc": "mục 1. vi phạm quy tắc giao thông đường bộ",
#         "dieu": "điều 7. xử phạt, trừ điểm giấy phép lái của người điều khiển xe mô tô, xe gắn máy, các loại xe tương tự xe mô tô và các loại xe tương tự xe gắn máy vi phạm quy tắc giao thông đường bộ",
#         "khoan": "khoản 1. phạt tiền từ 200.000 đồng đến 400.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:",
#         "content": "khoản 1. phạt tiền từ 200.000 đồng đến 400.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:\na) không chấp hành hiệu lệnh, chỉ dẫn của biển báo hiệu, vạch kẻ đường, trừ các hành vi vi phạm quy định tại điểm b, điểm d, điểm e khoản 2; điểm a, điểm c, điểm d, điểm h khoản 3; điểm a, điểm b, điểm c, điểm d khoản 4; điểm b, điểm d khoản 6; điểm a, điểm b, điểm c khoản 7; điểm a khoản 8; điểm b khoản 9; điểm a khoản 10 điều này;\nb) không có tín hiệu trước khi vượt hoặc có tín hiệu vượt xe nhưng không sử dụng trong suốt quá trình vượt xe;\nc) lùi xe mô tô ba bánh không quan sát hai bên, phía sau xe hoặc không có tín hiệu lùi xe;\nd) chở người ngồi trên xe sử dụng ô (dù);\nđ) không tuân thủ các quy định về nhường đường tại nơi đường giao nhau, trừ các hành vi vi phạm quy định tại điểm c, điểm d khoản 6 điều này;\ne) chuyển làn đường không đúng nơi cho phép hoặc không có tín hiệu báo trước hoặc chuyển làn đường không đúng quy định \"mỗi lần chuyển làn đường chỉ được phép chuyển sang một làn đường liền kề\";\ng) không sử dụng đèn chiếu sáng trong thời gian từ 18 giờ ngày hôm trước đến 06 giờ ngày hôm sau hoặc khi có sương mù, khói, bụi, trời mưa, thời tiết xấu làm hạn chế tầm nhìn;\nh) tránh xe không đúng quy định; sử dụng đèn chiếu xa khi gặp người đi bộ qua đường hoặc khi đi trên đoạn đường qua khu dân cư có hệ thống chiếu sáng đang hoạt động hoặc khi gặp xe đi ngược chiều (trừ trường hợp dải phân cách có khả năng chống chói) hoặc khi chuyển hướng xe tại nơi đường giao nhau; không nhường đường cho xe đi ngược chiều theo quy định tại nơi đường hẹp, đường dốc, nơi có chướng ngại vật;\ni) sử dụng còi trong thời gian từ 22 giờ ngày hôm trước đến 05 giờ ngày hôm sau trong khu đông dân cư, khu vực cơ sở khám bệnh, chữa bệnh, trừ các xe ưu tiên đang đi làm nhiệm vụ theo quy định;\nk) điều khiển xe chạy dưới tốc độ tối thiểu trên đoạn đường bộ có quy định tốc độ tối thiểu cho phép.",
#         "reference_id": "2.1.7.1"
#     },
#     {
#         "id": "chunk_2_18",
#         "chuong": "chương 2. hành vi vi phạm, hình thức, mức xử phạt, mức trừ điểm giấy phép lái xe và biện pháp khắc phục hậu quả vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ",
#         "muc": "mục 1. vi phạm quy tắc giao thông đường bộ",
#         "dieu": "điều 7. xử phạt, trừ điểm giấy phép lái của người điều khiển xe mô tô, xe gắn máy, các loại xe tương tự xe mô tô và các loại xe tương tự xe gắn máy vi phạm quy tắc giao thông đường bộ",
#         "khoan": "khoản 2. phạt tiền từ 400.000 đồng đến 600.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:",
#         "content": "khoản 2. phạt tiền từ 400.000 đồng đến 600.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:\na) dừng xe, đỗ xe trên phần đường xe chạy ở đoạn đường ngoài đô thị nơi có lề đường;\nb) điều khiển xe chạy quá tốc độ quy định từ 05 km/h đến dưới 10 km/h;\nc) điều khiển xe chạy tốc độ thấp mà không đi bên phải phần đường xe chạy gây cản trở giao thông;\nd) dừng xe, đỗ xe ở lòng đường gây cản trở giao thông; tụ tập từ 03 xe trở lên ở lòng đường, trong hầm đường bộ; đỗ, để xe ở lòng đường, vỉa hè trái phép;\nđ) xe không được quyền ưu tiên lắp đặt, sử dụng thiết bị phát tín hiệu của xe được quyền ưu tiên;\ne) dừng xe, đỗ xe trên điểm đón, trả khách, nơi đường bộ giao nhau, trên phần đường dành cho người đi bộ qua đường; dừng xe nơi có biển \"cấm dừng xe và đỗ xe\"; đỗ xe tại nơi có biển \"cấm đỗ xe\" hoặc biển \"cấm dừng xe và đỗ xe\"; không tuân thủ các quy định về dừng xe, đỗ xe tại nơi đường bộ giao nhau cùng mức với đường sắt; dừng xe, đỗ xe trong phạm vi hành lang an toàn giao thông đường sắt;\ng) chở theo 02 người trên xe, trừ trường hợp chở người bệnh đi cấp cứu, trẻ em dưới 12 tuổi, người già yếu hoặc người khuyết tật, áp giải người có hành vi vi phạm pháp luật;\nh) không đội \"mũ bảo hiểm cho người đi mô tô, xe máy\" hoặc đội \"mũ bảo hiểm cho người đi mô tô, xe máy\" không cài quai đúng quy cách khi điều khiển xe tham gia giao thông trên đường bộ;\ni) chở người ngồi trên xe không đội \"mũ bảo hiểm cho người đi mô tô, xe máy\" hoặc đội \"mũ bảo hiểm cho người đi mô tô, xe máy\" không cài quai đúng quy cách, trừ trường hợp chở người bệnh đi cấp cứu, trẻ em dưới 06 tuổi, áp giải người có hành vi vi phạm pháp luật;\nk) quay đầu xe tại nơi không được quay đầu xe, trừ hành vi vi phạm quy định tại điểm d khoản 4 điều này.",
#         "reference_id": "2.1.7.2"
#     },
#     {
#         "id": "chunk_2_19",
#         "chuong": "chương 2. hành vi vi phạm, hình thức, mức xử phạt, mức trừ điểm giấy phép lái xe và biện pháp khắc phục hậu quả vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ",
#         "muc": "mục 1. vi phạm quy tắc giao thông đường bộ",
#         "dieu": "điều 7. xử phạt, trừ điểm giấy phép lái của người điều khiển xe mô tô, xe gắn máy, các loại xe tương tự xe mô tô và các loại xe tương tự xe gắn máy vi phạm quy tắc giao thông đường bộ",
#         "khoan": "khoản 3. phạt tiền từ 600.000 đồng đến 800.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:",
#         "content": "khoản 3. phạt tiền từ 600.000 đồng đến 800.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:\na) chuyển hướng không quan sát hoặc không bảo đảm khoảng cách an toàn với xe phía sau hoặc không giảm tốc độ hoặc không có tín hiệu báo hướng rẽ hoặc có tín hiệu báo hướng rẽ nhưng không sử dụng liên tục trong quá trình chuyển hướng (trừ trường hợp điều khiển xe đi theo hướng cong của đoạn đường bộ ở nơi đường không giao nhau cùng mức); điều khiển xe rẽ trái tại nơi có biển báo hiệu có nội dung cấm rẽ trái đối với loại phương tiện đang điều khiển; điều khiển xe rẽ phải tại nơi có biển báo hiệu có nội dung cấm rẽ phải đối với loại phương tiện đang điều khiển;\nb) chở theo từ 03 người trở lên trên xe;\nc) dừng xe, đỗ xe trên cầu;\nd) điều khiển xe không đi bên phải theo chiều đi của mình; đi không đúng phần đường, làn đường quy định (làn cùng chiều hoặc làn ngược chiều); điều khiển xe đi qua dải phân cách cố định ở giữa hai phần đường xe chạy;\nđ) vượt bên phải trong trường hợp không được phép;\ne) người đang điều khiển xe hoặc chở người ngồi trên xe bám, kéo, đẩy xe khác, vật khác, dẫn dắt vật nuôi, mang vác vật cồng kềnh; chở người đứng trên yên, giá đèo hàng hoặc ngồi trên tay lái của xe;\ng) điều khiển xe kéo theo xe khác, vật khác;\nh) chạy trong hầm đường bộ không sử dụng đèn chiếu sáng gần;\ni) không giữ khoảng cách an toàn để xảy ra va chạm với xe chạy liền trước hoặc không giữ khoảng cách theo quy định của biển báo hiệu \"cự ly tối thiểu giữa hai xe\";\nk) điều khiển xe chạy dàn hàng ngang từ 03 xe trở lên;\nl) xe được quyền ưu tiên lắp đặt, sử dụng thiết bị phát tín hiệu ưu tiên không đúng quy định hoặc sử dụng thiết bị phát tín hiệu ưu tiên mà không có giấy phép của cơ quan có thẩm quyền cấp hoặc có giấy phép của cơ quan có thẩm quyền cấp nhưng không còn giá trị sử dụng theo quy định.",
#         "reference_id": "2.1.7.3"
#     },
#     {
#         "id": "chunk_2_20",
#         "chuong": "chương 2. hành vi vi phạm, hình thức, mức xử phạt, mức trừ điểm giấy phép lái xe và biện pháp khắc phục hậu quả vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ",
#         "muc": "mục 1. vi phạm quy tắc giao thông đường bộ",
#         "dieu": "điều 7. xử phạt, trừ điểm giấy phép lái của người điều khiển xe mô tô, xe gắn máy, các loại xe tương tự xe mô tô và các loại xe tương tự xe gắn máy vi phạm quy tắc giao thông đường bộ",
#         "khoan": "khoản 4. phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:",
#         "content": "khoản 4. phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:\na) điều khiển xe chạy quá tốc độ quy định từ 10 km/h đến 20 km/h;\nb) dừng xe, đỗ xe trong hầm đường bộ không đúng nơi quy định;\nc) vượt xe trong những trường hợp không được vượt, vượt xe tại đoạn đường có biển báo hiệu có nội dung cấm vượt đối với loại phương tiện đang điều khiển, trừ các hành vi vi phạm quy định tại điểm đ khoản 3 điều này;\nd) quay đầu xe trong hầm đường bộ;\nđ) người đang điều khiển xe sử dụng ô (dù), thiết bị âm thanh (trừ thiết bị trợ thính), dùng tay cầm và sử dụng điện thoại hoặc các thiết bị điện tử khác.",
#         "reference_id": "2.1.7.4"
#     },
#     {
#         "id": "chunk_2_21",
#         "chuong": "chương 2. hành vi vi phạm, hình thức, mức xử phạt, mức trừ điểm giấy phép lái xe và biện pháp khắc phục hậu quả vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ",
#         "muc": "mục 1. vi phạm quy tắc giao thông đường bộ",
#         "dieu": "điều 7. xử phạt, trừ điểm giấy phép lái của người điều khiển xe mô tô, xe gắn máy, các loại xe tương tự xe mô tô và các loại xe tương tự xe gắn máy vi phạm quy tắc giao thông đường bộ",
#         "khoan": "khoản 5. phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:",
#         "content": "khoản 5. phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:\na) điều khiển xe có liên quan trực tiếp đến vụ tai nạn giao thông mà không dừng ngay phương tiện, không giữ nguyên hiện trường, không trợ giúp người bị nạn, trừ hành vi vi phạm quy định tại điểm c khoản 9 điều này;\nb) chuyển hướng không nhường quyền đi trước cho: người đi bộ, xe lăn của người khuyết tật qua đường tại nơi có vạch kẻ đường dành cho người đi bộ; xe thô sơ đang đi trên phần đường dành cho xe thô sơ;\nc) chuyển hướng không nhường đường cho: các xe đi ngược chiều; người đi bộ, xe thô sơ đang qua đường tại nơi không có vạch kẻ đường cho người đi bộ.",
#         "reference_id": "2.1.7.5"
#     },
#     {
#         "id": "chunk_2_22",
#         "chuong": "chương 2. hành vi vi phạm, hình thức, mức xử phạt, mức trừ điểm giấy phép lái xe và biện pháp khắc phục hậu quả vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ",
#         "muc": "mục 1. vi phạm quy tắc giao thông đường bộ",
#         "dieu": "điều 7. xử phạt, trừ điểm giấy phép lái của người điều khiển xe mô tô, xe gắn máy, các loại xe tương tự xe mô tô và các loại xe tương tự xe gắn máy vi phạm quy tắc giao thông đường bộ",
#         "khoan": "khoản 6. phạt tiền từ 2.000.000 đồng đến 3.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:",
#         "content": "khoản 6. phạt tiền từ 2.000.000 đồng đến 3.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:\na) điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn nhưng chưa vượt quá 50 miligam/100 mililít máu hoặc chưa vượt quá 0,25 miligam/1 lít khí thở;\nb) đi vào khu vực cấm, đường có biển báo hiệu có nội dung cấm đi vào đối với loại phương tiện đang điều khiển, trừ các hành vi vi phạm quy định tại điểm a, điểm b khoản 7 điều này và các trường hợp xe ưu tiên đang đi làm nhiệm vụ khẩn cấp theo quy định;\nc) không giảm tốc độ (hoặc dừng lại) và nhường đường khi điều khiển xe đi từ đường không ưu tiên ra đường ưu tiên, từ đường nhánh ra đường chính;\nd) không giảm tốc độ và nhường đường cho xe đi đến từ bên phải tại nơi đường giao nhau không có báo hiệu đi theo vòng xuyến; không giảm tốc độ và nhường đường cho xe đi đến từ bên trái tại nơi đường giao nhau có báo hiệu đi theo vòng xuyến.",
#         "reference_id": "2.1.7.6"
#     },
#     {
#         "id": "chunk_2_23",
#         "chuong": "chương 2. hành vi vi phạm, hình thức, mức xử phạt, mức trừ điểm giấy phép lái xe và biện pháp khắc phục hậu quả vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ",
#         "muc": "mục 1. vi phạm quy tắc giao thông đường bộ",
#         "dieu": "điều 7. xử phạt, trừ điểm giấy phép lái của người điều khiển xe mô tô, xe gắn máy, các loại xe tương tự xe mô tô và các loại xe tương tự xe gắn máy vi phạm quy tắc giao thông đường bộ",
#         "khoan": "khoản 7. phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:",
#         "content": "khoản 7. phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:\na) đi ngược chiều của đường một chiều, đi ngược chiều trên đường có biển \"cấm đi ngược chiều\", trừ hành vi vi phạm quy định tại điểm b khoản này và các trường hợp xe ưu tiên đang đi làm nhiệm vụ khẩn cấp theo quy định; điều khiển xe đi trên vỉa hè, trừ trường hợp điều khiển xe đi qua vỉa hè để vào nhà, cơ quan;\nb) điều khiển xe đi vào đường cao tốc, trừ xe phục vụ việc quản lý, bảo trì đường cao tốc;\nc) không chấp hành hiệu lệnh của đèn tín hiệu giao thông;\nd) không chấp hành hiệu lệnh, hướng dẫn của người điều khiển giao thông hoặc người kiểm soát giao thông;\nđ) không nhường đường hoặc gây cản trở xe được quyền ưu tiên đang phát tín hiệu ưu tiên đi làm nhiệm vụ.",
#         "reference_id": "2.1.7.7"
#     },{
#         "id": "chunk_3_26",
#         "chuong": "chương 3. thẩm quyền, thủ tục xử phạt, trừ điểm, phục hồi điểm giấy phép lái xe",
#         "muc": "mục 2. thủ tục xử phạt",
#         "dieu": "điều 47. thủ tục xử phạt, nguyên tắc xử phạt đối với chủ phương tiện, người điều khiển phương tiện vi phạm quy định liên quan đến trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ",
#         "khoan": "khoản 3. đối với những hành vi vi phạm mà cùng được quy định tại các điều khác nhau của chương ii của nghị định này, trong trường hợp đối tượng vi phạm trùng nhau thì xử phạt như sau:",
#         "content": "khoản 3. đối với những hành vi vi phạm mà cùng được quy định tại các điều khác nhau của chương ii của nghị định này, trong trường hợp đối tượng vi phạm trùng nhau thì xử phạt như sau:\na) các hành vi vi phạm quy định về biển số, chứng nhận đăng ký xe, chứng nhận đăng ký xe tạm thời quy định tại điều 13 (điểm a khoản 4; điểm a khoản 6; điểm a, điểm b khoản 7; điểm a khoản 8), điều 14 (điểm a, điểm b, điểm c khoản 2; điểm a khoản 3), điều 16 (điểm a khoản 1; điểm a, điểm c, điểm d, điểm đ khoản 2) và các hành vi vi phạm tương ứng quy định tại điều 32 (điểm đ, điểm e, điểm h khoản 8; điểm đ khoản 9; điểm a, điểm b khoản 12; điểm d khoản 13), trong trường hợp chủ phương tiện là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại các điểm, khoản tương ứng của điều 32 của nghị định này;\nb) các hành vi vi phạm quy định về giấy chứng nhận, tem kiểm định an toàn kỹ thuật và bảo vệ môi trường của xe quy định tại điều 13 (điểm a khoản 5; điểm a, điểm b khoản 6), điều 16 (điểm đ khoản 1; điểm b, điểm đ khoản 2) và các hành vi vi phạm tương ứng quy định tại điều 32 (điểm b, điểm đ khoản 9; điểm a khoản 11), trong trường hợp chủ phương tiện là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại các điểm, khoản tương ứng của điều 32 của nghị định này;\nc) các hành vi vi phạm quy định về thời gian lái xe, thời gian nghỉ giữa hai lần lái xe liên tục của người lái xe, phù hiệu quy định tại điều 20 (điểm d khoản 6, khoản 7), điều 21 (điểm b khoản 5, điểm c khoản 6) và các hành vi vi phạm tương ứng quy định tại điều 32 (điểm d khoản 9, điểm đ khoản 11), trong trường hợp chủ phương tiện là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại các điểm, khoản tương ứng của điều 32 của nghị định này;\nd) các hành vi vi phạm quy định về niên hạn sử dụng của phương tiện quy định tại điều 13 (điểm a khoản 9) và các hành vi vi phạm tương ứng quy định tại điều 32 (điểm c khoản 17), trong trường hợp chủ phương tiện là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại điểm c khoản 17 điều 32 của nghị định này;\nđ) các hành vi vi phạm quy định về niên hạn sử dụng của phương tiện quy định tại điều 13 (điểm c khoản 5) và các hành vi vi phạm tương ứng quy định tại điều 26 (điểm i khoản 7), trong trường hợp cá nhân kinh doanh vận tải là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại điểm i khoản 7 điều 26 của nghị định này;\ne) các hành vi vi phạm quy định về kích thước thùng xe, khoang chở hành lý (hầm xe), lắp thêm hoặc tháo bớt ghế, giường nằm trên xe ô tô quy định tại điều 13 (điểm d khoản 3, điểm b khoản 4) và các hành vi vi phạm tương ứng quy định tại điều 32 (điểm d khoản 11, điểm h khoản 14), trong trường hợp chủ phương tiện là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại các điểm, khoản tương ứng của điều 32 của nghị định này;\ng) các hành vi vi phạm quy định về lắp, sử dụng thiết bị giám sát hành trình, thiết bị ghi nhận hình ảnh người lái xe trên xe ô tô quy định tại điều 20 (điểm l khoản 5, điểm đ khoản 6), điều 21 (điểm b khoản 3, điểm c khoản 5), điều 27 (điểm c khoản 1, điểm a khoản 3) và các hành vi vi phạm tương ứng quy định tại điều 26 (điểm c, điểm g khoản 7), trong trường hợp cá nhân kinh doanh vận tải là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại các điểm, khoản tương ứng của điều 26 của nghị định này;\nh) các hành vi vi phạm quy định về lắp, sử dụng thiết bị giám sát hành trình, thiết bị ghi nhận hình ảnh người lái xe trên xe ô tô quy định tại điều 29 (khoản 1, khoản 3), điều 30 (khoản 1, khoản 2) và các hành vi vi phạm tương ứng quy định tại điều 32 (điểm m, điểm n khoản 7), trong trường hợp chủ phương tiện là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại các điểm, khoản tương ứng của điều 32 của nghị định này;\ni) các hành vi vi phạm quy định về dây đai an toàn, hướng dẫn cho hành khách về an toàn giao thông, thoát hiểm khi xảy ra sự cố trên xe quy định tại điều 20 (điểm h, điểm i khoản 3) và các hành vi vi phạm tương ứng quy định tại điều 26 (điểm c khoản 2, điểm đ khoản 4) trong trường hợp cá nhân kinh doanh vận tải là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại các điểm, khoản tương ứng của điều 26 của nghị định này;\nk) các hành vi vi phạm quy định về đón, trả khách; nhận, trả hàng quy định tại điều 20 (khoản 8), điều 21 (khoản 9) và các hành vi vi phạm tương ứng quy định tại điều 26 (điểm c khoản 8), trong trường hợp cá nhân kinh doanh vận tải là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại điểm c khoản 8 điều 26 của nghị định này;\nl) các hành vi vi phạm quy định về dụng cụ, thiết bị chuyên dùng để cứu hộ, hỗ trợ cứu hộ giao thông đường bộ quy định tại điều 29 (khoản 2) và hành vi vi phạm tương ứng quy định tại điều 32 (điểm o khoản 7), trong trường hợp chủ phương tiện là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại điểm o khoản 7 điều 32 của nghị định này;\nm) các hành vi vi phạm quy định về thiết bị ghi nhận hình ảnh trẻ em mầm non, học sinh và thiết bị có chức năng cảnh báo, chống bỏ quên trẻ em trên xe quy định tại điều 27 (điểm b khoản 3) và hành vi vi phạm tương ứng quy định tại điều 26 (điểm b khoản 6), trong trường hợp cá nhân kinh doanh vận tải là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại điểm b khoản 6 điều 26 của nghị định này;\nn) các hành vi vi phạm quy định về màu sơn, biển báo dấu hiệu nhận biết của xe chở trẻ em mầm non, học sinh quy định tại điều 27 (điểm c, điểm d khoản 3) và hành vi vi phạm tương ứng quy định tại điều 26 (điểm c, điểm d khoản 6), trong trường hợp cá nhân kinh doanh vận tải là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại các điểm, khoản tương ứng của điều 26 của nghị định này;\no) các hành vi vi phạm quy định về chở hàng siêu trường, siêu trọng, chở quá khổ, quá tải, quá số người quy định tại điều 20, điều 21, điều 22, điều 34 và các hành vi vi phạm tương ứng quy định tại điều 32, trong trường hợp chủ phương tiện là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại điều 32 của nghị định này;\np) các hành vi vi phạm quy định về vận chuyển hàng hóa là phương tiện vận tải, máy móc, thiết bị kỹ thuật, hàng dạng trụ quy định tại điều 21 (điểm a khoản 10) và hành vi vi phạm tương ứng quy định tại điều 32 (điểm đ khoản 13), trong trường hợp chủ phương tiện là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại điểm đ khoản 13 điều 32 của nghị định này;\nq) các hành vi vi phạm quy định về niêm yết thông tin (hành trình chạy xe) quy định tại điều 20 (điểm k khoản 3) và hành vi vi phạm tương ứng quy định tại điều 26 (điểm g khoản 4), trong trường hợp cá nhân kinh doanh vận tải là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại điểm g khoản 4 điều 26 của nghị định này;\nr) các hành vi vi phạm quy định về không thực hiện đúng các nội dung thông tin đã niêm yết (tuyến đường, lịch trình, hành trình vận tải) quy định tại điều 20 (điểm c khoản 3) và các hành vi vi phạm tương ứng quy định tại điều 26 (điểm b khoản 7), trong trường hợp cá nhân kinh doanh vận tải là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại điểm b khoản 7 điều 26 của nghị định này;\ns) các hành vi vi phạm quy định về lệnh vận chuyển, giấy vận tải quy định tại điều 20 (điểm e khoản 5), điều 21 (điểm đ khoản 2) và hành vi vi phạm tương ứng quy định tại điều 26 (điểm đ khoản 2), trong trường hợp cá nhân kinh doanh vận tải là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại điểm đ khoản 2 điều 26 của nghị định này;\nt) các hành vi vi phạm quy định về vận chuyển hàng hoá nguy hiểm mà không làm sạch hoặc không bóc (xóa) biểu trưng nguy hiểm trên phương tiện khi không tiếp tục vận chuyển loại hàng hóa đó quy định tại điều 23 (khoản 1) và hành vi vi phạm tương ứng quy định tại điều 26 (điểm e khoản 2), trong trường hợp cá nhân kinh doanh vận tải là người trực tiếp điều khiển phương tiện thì bị xử phạt theo quy định tại điểm e khoản 2 điều 26 của nghị định này.",
#         "reference_id": "3.2.47.3"
#     },
# ]
# def split_content_into_chunks(content, chunk_size=2):
#     """Tách nội dung dài thành các câu và chia thành các chunk."""
#     sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)
#     num_sentences = len(sentences)
#     if num_sentences <= chunk_size:
#         return sentences
#     else:
#         chunk1_sentences = sentences[:num_sentences // 2]
#         chunk2_sentences = sentences[num_sentences // 2:]
#         return [" ".join(chunk1_sentences).strip(), " ".join(chunk2_sentences).strip()]

# # Xử lý content của phần tử đầu tiên nếu nó dài
# if len(data[0]['content'].split()) > 256: # Ước lượng độ dài, bạn có thể điều chỉnh
#     original_content = data[0]['content']
#     chunks = split_content_into_chunks(original_content, chunk_size=2)
#     # Thay thế content ban đầu bằng danh sách các chunk
#     data[0]['content'] = chunks

# # Chuẩn bị danh sách content để encode
# all_contents = []
# original_indices = [] # Để theo dõi index ban đầu

# for i, item in enumerate(data):
#     if isinstance(item['content'], list):
#         for chunk in item['content']:
#             all_contents.append(chunk)
#             original_indices.append(i)
#     else:
#         all_contents.append(item['content'])
#         original_indices.append(i)

# # Define a maximum length for truncation
# max_length = 256  # Thay đổi thành 256

# # Encode all content chunks
# encoded_inputs = tokenizer(all_contents, padding=True, truncation=True, return_tensors='pt', max_length=max_length)

# with torch.no_grad():
#     model_outputs = model(**encoded_inputs)

# content_embeddings = mean_pooling(model_outputs, encoded_inputs['attention_mask'])

# # Query sentence
# query = "vận chuyển hàng hoá nguy hiểm mà không làm sạch hoặc không bóc"

# # Encode the query sentence
# encoded_query = tokenizer([query], padding=True, truncation=True, return_tensors='pt', max_length=max_length)

# with torch.no_grad():
#     output_query = model(**encoded_query)

# query_embedding = mean_pooling(output_query, encoded_query['attention_mask'])

# # Calculate cosine similarity
# similarity_scores = cosine_similarity(query_embedding, content_embeddings)

# # Get indices of top 3 most similar content
# top_indices = np.argsort(similarity_scores[0])[-3:][::-1]

# print("\nTop 3 nội dung tương đồng với câu query:")
# for index in top_indices:
#     original_data_index = original_indices[index]
#     content_chunk = all_contents[index]
#     print(f"Độ tương đồng: {similarity_scores[0][index]:.4f}")
#     print(f"Nội dung: {content_chunk}")
#     print(f"ID gốc: {data[original_data_index]['id']}")
#     print("-" * 50)