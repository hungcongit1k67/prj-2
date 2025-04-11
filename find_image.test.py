import cv2
import numpy as np
import imutils

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
    tl = [tl[0]+tl[2]//2, tl[1]+tl[3]//2]
    tr = [tr[0]+tr[2]//2, tr[1]+tr[3]//2]
    bl = [bl[0]+bl[2]//2, bl[1]+bl[3]//2]
    br = [br[0]+br[2]//2, br[1]+br[3]//2]
    
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
        if 200/(1902*2698)*image_area < area < 2000/(1902*2698)*image_area and x + w // 2 < image.shape[1] // 50 and w/h > 0.2 and w/h < 1/0.2:
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
    print(filtered_local)
    return filtered_local, image



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
    print(x, y, w, h)
    return image[y:y+h, x:x+w]

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
    print(x, y, w, h)
    return image[y:y+h, x:x+w]

def find_Part1(image):
    local, image = find_local(image)
    
    if len(local) < 2:  # Kiểm tra nếu có ít nhất 2 bounding box
        return None  # Không tìm thấy SBD
    
    height, width = local[2][1] - local[1][1], image.shape[1]
    
    # Tính toán x, y, w, h và làm tròn các giá trị
    x = round(93 / 1902 * width)
    y = round(954 / (1465-830) * height)
    w = round((440 - 93) / 1902 * width)
    h = round((1417 - 954) / (1465-830) * height)
    of = round((567-93)/1902*width)
    # Cắt vùng ảnh theo tọa độ đã tính toán
    print(x, y, w, h)
    return image[y:y+h, x:x+w], image[y:y+h, x+of:x+w+of], image[y:y+h, x+of*2:x+w+of*2], image[y:y+h, x+of*3:x+w+of*3]

def find_Part2(image):
    local, image = find_local(image)
    
    if len(local) < 2:  # Kiểm tra nếu có ít nhất 2 bounding box
        return None  # Không tìm thấy SBD
    
    height, width = local[3][1] - local[2][1], image.shape[1]
    
    # Tính toán x, y, w, h và làm tròn các giá trị
    x = round(95 / 1902 * width)
    y = round(1649 / (1868-1465) * height)
    w = round((240 - 95) / 1902 * width)
    h = round((1830 - 1649) / (1868-1465) * height)
    of1 = round((293-95) / 1902 * width)
    of2 = round((568-95) / 1902 * width)
    # Cắt vùng ảnh theo tọa độ đã tính toán
    print(x, y, w, h)
    image1 = image[y:y+h, x:x+w]
    image2 = image[y:y+h, x+of1:x+w+of1]
    image3 = image[y:y+h, x+of2:x+w+of2]
    image4 = image[y:y+h, x+of1+of2:x+w+of1+of2]
    image5 = image[y:y+h, x+of2*2:x+w+of2*2]
    image6 = image[y:y+h, x+of1+of2*2:x+w+of1+of2*2]
    image7 = image[y:y+h, x+of2*3:x+w+of2*3]
    image8 = image[y:y+h, x+of1+of2*3:x+w+of1+of2*3]
    return image1, image2, image3, image4, image5, image6, image7, image8

def find_Part3(image):
    local, image = find_local(image)
    
    if len(local) < 2:  # Kiểm tra nếu có ít nhất 2 bounding box
        return None  # Không tìm thấy SBD
    
    height, width = local[4][1] - local[3][1], image.shape[1]
    
    # Tính toán x, y, w, h và làm tròn các giá trị
    x = round(94 / 1902 * width)
    y = round(2070 / (2670-1868) * height)
    w = round((275 - 94) / 1902 * width)
    h = round((2625 - 2070) / (2670-1868) * height)
    of = round((410-94)/1902*width)
    # Cắt vùng ảnh theo tọa độ đã tính toán
    print(x, y, w, h)
    return image[y:y+h, x:x+w], image[y:y+h, x+of:x+w+of], image[y:y+h, x+of*2:x+w+of*2], image[y:y+h, x+of*3:x+w+of*3], image[y:y+h, x+of*4:x+w+of*4], image[y:y+h, x+of*5:x+w+of*5]
def solve(image):
    
    # Tìm SBD image
    SBD_image = find_SBD_image(image)
    MDT_image = find_MDT_image(image)
    Part1_image1, Part1_image2, Part1_image3, Part1_image4 = find_Part1(image)
    Part2_image1, Part2_image2, Part2_image3, Part2_image4, Part2_image5, Part2_image6, Part2_image7, Part2_image8 = find_Part2(image)
    Part3_image1, Part3_image2, Part3_image3, Part3_image4, Part3_image5, Part3_image6 = find_Part3(image)
    _, image= find_local(image)
    if SBD_image is not None:
        cv2.imshow("SBD Image", imutils.resize(Part1_image4, height=800))
    else:
        print("Không tìm thấy SBD")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def main():
    # Đọc ảnh
    image = cv2.imread("D:\prj 2\z6495321724781_d1d742784d6ff262dfdb9826d447b93f.jpg")
    #image = cv2.imread('./data/Trainning_set/Trainning_SET/Images/IMG_1581_iter_1.jpg')
    solve(image)

if __name__ == "__main__":
    main()
