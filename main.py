import cv2
import numpy as np
import imutils
import model

from model import CNN_Model

# chuẩn hóa hình ảnh về kích thước 28x28 và chuyển đổi sang định dạng (1, 28, 28, 1); chuẩn hóa về khoảng [0, 1]
def preprocess_image(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img.reshape((1, 28, 28, 1))  # Thêm batch dimension
    img = img / 255.0  # Chuẩn hóa về khoảng [0, 1]
    return img


def convert_binary_image(image):
    # Chuyển ảnh về grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng threshold để làm nổi bật các điểm chấm đen
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Áp dụng Morphological Opening để loại bỏ nhiễu nhỏ
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh

# Tự động cắt lấy vùng chứa thông tin bằng 4 góc
def crop_4_goc(image):
    thresh = convert_binary_image(image)
    # Tìm các contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc và vẽ các ô vuông tìm thấy trên ảnh
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
    for contour in contours:
        # Tính diện tích của contour
        area = cv2.contourArea(contour)

        # Tìm hình chữ nhật bao quanh contour
        x, y, w, h = cv2.boundingRect(contour)

    # Tính bounding box cho từng contour
    contours_bound = [cv2.boundingRect(contour) for contour in contours]

    # Sắp xếp contours_bound dựa trên tổng giá trị x (hệ số vị trí x + y)
    contours_bound = sorted(contours_bound, key=lambda x: (x[0] + x[1]))
    tl, tr, bl, br = contours_bound

    # In ra kết quả
    tl = [tl[0] + tl[2] // 2, tl[1] + tl[3] // 2]
    tr = [tr[0] + tr[2] // 2, tr[1] + tr[3] // 2]
    bl = [bl[0] + bl[2] // 2, bl[1] + bl[3] // 2]
    br = [br[0] + br[2] // 2, br[1] + br[3] // 2]

    # Chuyển tọa độ các điểm thành float32 để sử dụng cho phép biến đổi perspective
    pts1 = np.float32([tl, tr, br, bl])

    # Tạo các điểm gốc của hình chữ nhật (các góc của ảnh cắt)
    width, height = br[0] - tl[0], br[1] - tl[1]  # Kích thước của vùng cắt (có thể thay đổi theo yêu cầu)
    pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    # Tính toán ma trận perspective transform
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Áp dụng phép biến đổi perspective
    image = cv2.warpPerspective(image, matrix, (width, height))
    thresh = cv2.warpPerspective(thresh, matrix, (width, height))
    return image, thresh

# phát hiện cạnh bằng Sobel trong ảnh --> Giúp tìm các vùng có cạnh nổi bật (ví dụ khung ô, chữ)
def sobel_edge_detection(image):
    # Chuyển ảnh sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng Sobel để tìm gradient theo cả hai hướng X và Y
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradient theo hướng X
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradient theo hướng Y

    # Tính tổng độ mạnh của gradient
    edges = cv2.magnitude(grad_x, grad_y)

    # Chuyển đổi sang kiểu uint8 để hiển thị
    edges = np.uint8(np.absolute(edges))

    # Trả về ảnh với các cạnh đã được phát hiện
    return edges

# Trả ra danh sách các vị trí chấm tròn để định vị từng phần trên phiếu
def find_local(image):
    # Cắt và chuyển ảnh sang chế độ nhị phân

    image, thresh = crop_4_goc(image)
    image_area = image.shape[0] * image.shape[1]
    # Áp dụng Sobel Edge Detection
    edges = sobel_edge_detection(image)

    # Áp dụng threshold để chuyển các giá trị độ mạnh thành ảnh nhị phân (black & white)
    _, thresh = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

    # Tìm contours trong ảnh đã threshold
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Vẽ các contours lên ảnh gốc (hoặc ảnh đã qua xử lý)
    local = []
    seen_y_values = set()  # Set để theo dõi giá trị x[1] đã gặp

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Tính bounding box của contour
        area = w * h  # Diện tích của contour

        # Kiểm tra nếu diện tích hợp lệ và có các điều kiện khác
        if 200 / (1902 * 2698) * image_area < area < 2000 / (1902 * 2698) * image_area and x + w // 2 < image.shape[
            1] // 50 and w / h > 0.2 and w / h < 1 / 0.2:
            if y not in seen_y_values:  # Nếu giá trị x[1] chưa gặp
                local.append(cv2.boundingRect(contour))
                seen_y_values.add(y)  # Thêm y vào set đã gặp

    # Sắp xếp các contour theo giá trị của y (theo trục y)
    local = sorted(local, key=lambda x: x[1])

    # Loại bỏ các bounding box có hiệu x[1] < 30 giữa các phần tử liên tiếp
    filtered_local = []
    previous_y = -float('inf')  # Giá trị y trước đó (bắt đầu với một giá trị rất nhỏ)
    for box in local:
        x, y, w, h = box
        if abs(y - previous_y) >= 30:  # Kiểm tra hiệu của y với giá trị y trước đó
            filtered_local.append(box)
            previous_y = y  # Cập nhật giá trị y cho lần lặp tiếp theo

    return filtered_local, image

# Trích xuất ảnh vùng Số Báo Danh.
def find_SBD_image(image):
    local, image = find_local(image)

    if len(local) < 2:  # Kiểm tra nếu có ít nhất 2 bounding box
        return None  # Không tìm thấy SBD

    height, width = local[1][1] - local[0][1], image.shape[1]

    # Tính toán x, y, w, h và làm tròn các giá trị
    x = round(1417 / 1902 * width)
    y = round(217 / 830 * height)
    w = round((1644 - 1417) / 1902 * width)
    h = round((756 - 217) / 830 * height)

    # Cắt vùng ảnh theo tọa độ đã tính toán

    return image[y:y + h, x:x + w], x, y, w, h

# Trích xuất ảnh mã đề thi.
def find_MDT_image(image):
    local, image = find_local(image)

    if len(local) < 2:  # Kiểm tra nếu có ít nhất 2 bounding box
        return None  # Không tìm thấy SBD

    height, width = local[1][1] - local[0][1], image.shape[1]

    # Tính toán x, y, w, h và làm tròn các giá trị
    x = round(1733 / 1902 * width)
    y = round(217 / 830 * height)
    w = round((1849 - 1733) / 1902 * width)
    h = round((756 - 217) / 830 * height)

    # Cắt vùng ảnh theo tọa độ đã tính toán

    return image[y:y + h, x:x + w], x, y, w, h


def find_Part1(image):
    local, image = find_local(image)

    if len(local) < 2:  # Kiểm tra nếu có ít nhất 2 bounding box
        return None  # Không tìm thấy SBD

    height, width = local[2][1] - local[1][1], image.shape[1]

    # Tính toán x, y, w, h và làm tròn các giá trị
    x = round(93 / 1902 * width)
    y = round(954 / (1465 - 830) * height)
    w = round((440 - 93) / 1902 * width)
    h = round((1417 - 954) / (1465 - 830) * height)
    of = round((567 - 93) / 1902 * width)
    # Cắt vùng ảnh theo tọa độ đã tính toán

    return image[y:y + h, x:x + w], image[y:y + h, x + of:x + w + of], image[y:y + h, x + of * 2:x + w + of * 2], image[
                                                                                                                  y:y + h,
                                                                                                                  x + of * 3:x + w + of * 3], x, y, w, h, of


def find_Part2(image):
    local, image = find_local(image)

    if len(local) < 2:  # Kiểm tra nếu có ít nhất 2 bounding box
        return None  # Không tìm thấy SBD

    height, width = local[3][1] - local[2][1], image.shape[1]

    # Tính toán x, y, w, h và làm tròn các giá trị
    x = round(95 / 1902 * width)
    y = round(1649 / (1868 - 1465) * height)
    w = round((240 - 95) / 1902 * width)
    h = round((1830 - 1649) / (1868 - 1465) * height)
    of1 = round((293 - 95) / 1902 * width)
    of2 = round((568 - 95) / 1902 * width)
    # Cắt vùng ảnh theo tọa độ đã tính toán

    image1 = image[y:y + h, x:x + w]
    image2 = image[y:y + h, x + of1:x + w + of1]
    image3 = image[y:y + h, x + of2:x + w + of2]
    image4 = image[y:y + h, x + of1 + of2:x + w + of1 + of2]
    image5 = image[y:y + h, x + of2 * 2:x + w + of2 * 2]
    image6 = image[y:y + h, x + of1 + of2 * 2:x + w + of1 + of2 * 2]
    image7 = image[y:y + h, x + of2 * 3:x + w + of2 * 3]
    image8 = image[y:y + h, x + of1 + of2 * 3:x + w + of1 + of2 * 3]
    return image1, image2, image3, image4, image5, image6, image7, image8, x, y, w, h, of1, of2


def find_Part3(image):
    local, image = find_local(image)

    if len(local) < 2:  # Kiểm tra nếu có ít nhất 2 bounding box
        return None  # Không tìm thấy SBD

    height, width = local[4][1] - local[3][1], image.shape[1]

    # Tính toán x, y, w, h và làm tròn các giá trị
    x = round(94 / 1902 * width)
    y = round(2070 / (2670 - 1868) * height)
    w = round((275 - 94) / 1902 * width)
    h = round((2625 - 2070) / (2670 - 1868) * height)
    of = round((410 - 94) / 1902 * width)

    return image[y:y + h, x:x + w], image[y:y + h, x + of:x + w + of], image[y:y + h, x + of * 2:x + w + of * 2], image[
                                                                                                                  y:y + h,
                                                                                                                  x + of * 3:x + w + of * 3], image[
                                                                                                                                              y:y + h,
                                                                                                                                              x + of * 4:x + w + of * 4], image[
                                                                                                                                                                          y:y + h,
                                                                                                                                                                          x + of * 5:x + w + of * 5], x, y, w, h, of


import os


def read_SBD(image):
    SBD_image, x, y, w, h = find_SBD_image(image)
    offx = SBD_image.shape[1] // 6  # Chia theo chiều rộng (width)
    offy = SBD_image.shape[0] // 10  # Chia theo chiều cao (height)
    SBD_map = {i: [] for i in range(6)} # danh sách chữ số đã tô
    SBD_rec = [] # tọa độ các ô được tô
    if not os.path.exists("dataCNN"):
        os.makedirs("dataCNN")

    model = CNN_Model('weight.h5').build_model(rt=True)
    list_image = []
    m = 0
    for i in range(6):
        for j in range(10):
            temp_img = SBD_image[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{10 * i + j}.jpg", temp_img)
            list_image.append(temp_img)

            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                SBD_map[i].append(j)
                # SBD_res.append()
                SBD_rec.append([x + offx * i, y + offy * j, offx, offy])
            m = m + 1

    return SBD_map, SBD_rec


def read_MDT(image):
    MDT_image, x, y, w, h = find_MDT_image(image)
    offx = MDT_image.shape[1] // 3  # Chia theo chiều rộng (width) thành 3 phần
    offy = MDT_image.shape[0] // 10  # Chia theo chiều cao (height) thành 10 phần
    MDT_map = {i: [] for i in range(3)}
    MDT_rec = []
    if not os.path.exists("dataCNN"):
        os.makedirs("dataCNN")

    model = CNN_Model('weight.h5').build_model(rt=True)
    list_image = []

    for i in range(3):
        for j in range(10):
            temp_img = MDT_image[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{10 * i + j + 60}.jpg", temp_img)
            list_image.append(temp_img)

            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                MDT_map[i].append(j)
                MDT_rec.append([x + offx * i, y + offy * j, offx, offy])

    return MDT_map, MDT_rec


def read_Part1(image):
    image1, image2, image3, image4, x, y, w, h, of = find_Part1(image)
    offx = image1.shape[1] // 4  # Chia theo chiều rộng (width) thành 3 phần
    offy = image1.shape[0] // 10  # Chia theo chiều cao (height) thành 10 phần
    ans = ['A', 'B', 'C', 'D']
    Part1_rec = []
    Part1_map = {i: [] for i in range(40)}

    if not os.path.exists("dataCNN"):
        os.makedirs("dataCNN")

    model = CNN_Model('weight.h5').build_model(rt=True)
    list_image = []

    for j in range(10):
        for i in range(4):
            temp_img = image1[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{j*4+i+250}.jpg", temp_img)
            list_image.append(temp_img)
            # print(j*4+i+90)
            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part1_map[j].append(ans[i])
                Part1_rec.append([x + offx * i, y + offy * j, offx, offy])
    for j in range(10):
        for i in range(4):
            temp_img = image2[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{j*4+i+250}.jpg", temp_img)
            list_image.append(temp_img)
            # print(j*4+i+90)
            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part1_map[j + 10].append(ans[i])
                Part1_rec.append([x + offx * i + of, y + offy * j, offx, offy])
    for j in range(10):
        for i in range(4):
            temp_img = image3[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{j*4+i+250}.jpg", temp_img)
            list_image.append(temp_img)
            # print(j*4+i+90)
            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part1_map[j + 20].append(ans[i])
                Part1_rec.append([x + offx * i + of * 2, y + offy * j, offx, offy])
    for j in range(10):
        for i in range(4):
            temp_img = image4[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{j*4+i+250}.jpg", temp_img)
            list_image.append(temp_img)
            # print(j*4+i+90)
            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part1_map[j + 30].append(ans[i])
                Part1_rec.append([x + offx * i + of * 3, y + offy * j, offx, offy])
    return Part1_map, Part1_rec


def read_Part2(image):
    image1, image2, image3, image4, image5, image6, image7, image8, x, y, w, h, of1, of2 = find_Part2(image)
    offx = image1.shape[1] // 2  # Chia theo chiều rộng (width) thành 3 phần
    offy = image1.shape[0] // 4  # Chia theo chiều cao (height) thành 10 phần
    Part2_map = {i: [] for i in range(32)}
    Part2_rec = []
    ans = [True, False]
    if not os.path.exists("dataCNN"):
        os.makedirs("dataCNN")

    model = CNN_Model('weight.h5').build_model(rt=True)
    list_image = []

    for j in range(4):
        for i in range(2):
            temp_img = image1[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{j*2+i+346}.jpg", temp_img)
            list_image.append(temp_img)
            # print(j*4+i+90)
            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part2_map[j].append(ans[i])
                Part2_rec.append([x + offx * i, y + offy * j, offx, offy])

    for j in range(4):
        for i in range(2):
            temp_img = image2[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{j*2+i+346}.jpg", temp_img)
            list_image.append(temp_img)
            # print(j*4+i+90)
            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part2_map[j + 4].append(ans[i])
                Part2_rec.append([x + offx * i + of1, y + offy * j, offx, offy])

    for j in range(4):
        for i in range(2):
            temp_img = image3[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{j*2+i+346}.jpg", temp_img)
            list_image.append(temp_img)
            # print(j*4+i+90)
            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part2_map[j + 8].append(ans[i])
                Part2_rec.append([x + offx * i + of2, y + offy * j, offx, offy])

    for j in range(4):
        for i in range(2):
            temp_img = image4[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{j*2+i+346}.jpg", temp_img)
            list_image.append(temp_img)
            # print(j*4+i+90)
            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part2_map[j + 12].append(ans[i])
                Part2_rec.append([x + offx * i + of1 + of2, y + offy * j, offx, offy])

    for j in range(4):
        for i in range(2):
            temp_img = image5[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{j*2+i+346}.jpg", temp_img)
            list_image.append(temp_img)
            # print(j*4+i+90)
            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part2_map[j + 16].append(ans[i])
                Part2_rec.append([x + offx * i + of2 * 2, y + offy * j, offx, offy])

    for j in range(4):
        for i in range(2):
            temp_img = image6[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{j*2+i+346}.jpg", temp_img)
            list_image.append(temp_img)
            # print(j*4+i+90)
            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part2_map[j + 20].append(ans[i])
                Part2_rec.append([x + offx * i + of1 + of2 * 2, y + offy * j, offx, offy])

    for j in range(4):
        for i in range(2):
            temp_img = image7[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{j*2+i+346}.jpg", temp_img)
            list_image.append(temp_img)
            # print(j*4+i+90)
            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part2_map[j + 24].append(ans[i])
                Part2_rec.append([x + offx * i + of2 * 3, y + offy * j, offx, offy])

    for j in range(4):
        for i in range(2):
            temp_img = image8[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{j*2+i+346}.jpg", temp_img)
            list_image.append(temp_img)
            # print(j*4+i+90)
            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part2_map[j + 28].append(ans[i])
                Part2_rec.append([x + offx * i + of1 + of2 * 3, y + offy * j, offx, offy])
    return Part2_map, Part2_rec


def read_Part3(image):
    image1, image2, image3, image4, image5, image6, x, y, w, h, of = find_Part3(image)
    offx = image1.shape[1] // 4  # Chia theo chiều rộng (width) thành 3 phần
    offy = image1.shape[0] // 12  # Chia theo chiều cao (height) thành 10 phần
    Part3_map = {i: [] for i in range(24)}
    Part3_rec = []
    ans = ['-', ',', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if not os.path.exists("dataCNN"):
        os.makedirs("dataCNN")

    model = CNN_Model('weight.h5').build_model(rt=True)
    list_image = []

    for i in range(4):
        for j in range(12):
            temp_img = image1[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{12 * i + j + 594}.jpg", temp_img)
            list_image.append(temp_img)

            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part3_map[i].append(ans[j])
                Part3_rec.append([x + offx * i, y + offy * j, offx, offy])

    for i in range(4):
        for j in range(12):
            temp_img = image2[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{12 * i + j + 594}.jpg", temp_img)
            list_image.append(temp_img)

            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part3_map[i + 4].append(ans[j])
                Part3_rec.append([x + offx * i + of, y + offy * j, offx, offy])

    for i in range(4):
        for j in range(12):
            temp_img = image3[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{12 * i + j + 594}.jpg", temp_img)
            list_image.append(temp_img)

            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part3_map[i + 8].append(ans[j])
                Part3_rec.append([x + offx * i + of * 2, y + offy * j, offx, offy])

    for i in range(4):
        for j in range(12):
            temp_img = image4[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{12 * i + j + 594}.jpg", temp_img)
            list_image.append(temp_img)

            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part3_map[i + 12].append(ans[j])
                Part3_rec.append([x + offx * i + of * 3, y + offy * j, offx, offy])

    for i in range(4):
        for j in range(12):
            temp_img = image5[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{12 * i + j + 594}.jpg", temp_img)
            list_image.append(temp_img)

            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part3_map[i + 16].append(ans[j])
                Part3_rec.append([x + offx * i + of * 4, y + offy * j, offx, offy])

    for i in range(4):
        for j in range(12):
            temp_img = image6[offy * j:offy * (j + 1), offx * i:offx * (i + 1)]
            # cv2.imwrite(f"dataCNN/img{12 * i + j + 594}.jpg", temp_img)
            list_image.append(temp_img)

            temp_img = preprocess_image(temp_img)
            scores = model.predict_on_batch(temp_img)

            if scores[0][1] > 0.9:
                Part3_map[i + 20].append(ans[j])
                Part3_rec.append([x + offx * i + of * 5, y + offy * j, offx, offy])
    return Part3_map, Part3_rec


import sys
import json
import cv2
import imutils


def solve(image):
    SBD_map, SBD_rec = read_SBD(image)
    MDT_map, MDT_rec = read_MDT(image)
    Part1_map, Part1_rec = read_Part1(image)
    Part2_map, Part2_rec = read_Part2(image)
    Part3_map, Part3_rec = read_Part3(image)

    image, _ = crop_4_goc(image)

    # Vẽ hình chữ nhật trên ảnh
    for recs in [SBD_rec, MDT_rec, Part1_rec, Part2_rec, Part3_rec]:
        for rec in recs:
            cv2.rectangle(image, (rec[0], rec[1]), (rec[0] + rec[2], rec[1] + rec[3]), (0, 255, 0), 2)

    # Tạo output dưới dạng JSON
    result = {
        "SBD": {k + 1: v for k, v in SBD_map.items()}, # items(): trả về một danh sách các tuple chứa key-value của dict
        "MDT": {k + 1: v for k, v in MDT_map.items()},
        "Part1": {k + 1: v for k, v in Part1_map.items()},
        "Part2": {k + 1: v for k, v in Part2_map.items()},
        "Part3": {k + 1: v for k, v in Part3_map.items()},
    }

    return image, result


def main():
    # code bẩn quá =))
    image_path = r"D:\prj 2\z6495321724781_d1d742784d6ff262dfdb9826d447b93f.jpg"
    output_image_path = r"D:\prj 2\output\result.jpg"  # Đảm bảo output có .jpg hoặc .png

    image = cv2.imread(image_path)

    processed_image, result = solve(image)

    # Lưu ảnh kết quả
    cv2.imwrite(output_image_path, processed_image)

    # Xuất kết quả dưới dạng JSON
    print(json.dumps(result))


if __name__ == "__main__":
    main()

