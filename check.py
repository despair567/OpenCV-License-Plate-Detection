import cv2
import numpy as np

# 演示图像
def show_img(desc, image):
    cv2.imshow(desc, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#-------------------
# 加载图像和预处理
#-------------------
img = cv2.imread('test.jpg')

# 调整图片大小
img = cv2.resize(img, (1024, 800))

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 双边滤波                输入图像 邻域直径  颜色空间标准差  坐标空间标准差
bif = cv2.bilateralFilter(gray, 13, 15, 15)

#-------------------
# 边缘检测和轮廓处理
#-------------------
# 边缘检测         算法 低阈值 高阈值
edged = cv2.Canny(bif, 50, 200)

# 寻找轮廓
contours,_ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 最大的前10个轮廓                       按照面积排序         降序      取前10个
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

# 画出轮廓
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

#-------------------
# 找到符合车牌的轮廓
#-------------------
screenCnt = None
# -------------------
# 直接基于颜色提取车牌 (跳过 Canny 边缘检测)
# -------------------
# 1. 转换为 HSV 颜色空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 2. 设定蓝色的 HSV 范围
lower_blue = np.array([90, 80, 80])
upper_blue = np.array([130, 255, 255])

# 3. 创建蓝色掩膜 (蓝色区域变白，其他区域变黑)
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 4. 对掩膜进行形态学操作（闭运算），把车牌上的白色字体的空隙填满，连成一个整体
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 5. 在蓝色掩膜中寻找轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 6. 找出最像车牌的那个蓝色轮廓
screenCnt = None
for c in contours:
    # 计算面积，过滤掉太小的噪点
    area = cv2.contourArea(c)
    if area < 2000:
        continue

    # 获取最小外接矩形
    x, y, w, h = cv2.boundingRect(c)

    # 计算长宽比 (中国车牌蓝牌约为 2.5 到 3.5)
    aspect_ratio = float(w) / h

    if 2.0 < aspect_ratio < 4.0:
        screenCnt = c
        break

# -------------------
# 绘制结果
# -------------------
if screenCnt is not None:
    # 画出红色边框
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 2)
    print("成功检测到蓝色车牌！")
else:
    print("未检测到符合条件的蓝色车牌。")

# -------------------
# 图像裁剪
# -------------------
# 遮罩
mask = np.zeros(gray.shape,np.uint8)
cv2.drawContours(mask, [screenCnt],0, 255, -1)

# 图像位运算
mask_image = cv2.bitwise_and(img, img, mask=mask)

# 图像裁剪
(x,y) = np.where(mask == 255)
# 左上方
(topx,topy) = (np.min(x), np.min(y))
# 右下方
(bottomx,bottomy) = (np.max(x), np.max(y))

cropped = img[topx:bottomx,topy:bottomy]
# 将裁剪出来的车牌图片保存到本地
cv2.imwrite("cropped_plate.jpg", cropped)
print("车牌裁剪图已保存为 cropped_plate.jpg")

# -------------------
# 车牌字符识别 (CPU版 EasyOCR)
# -------------------
import easyocr

print("正在加载 EasyOCR 模型（首次运行会下载，请耐心等待）...")
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

print("开始识别...")
result = reader.readtext(cropped)

print("\n=== 识别结果 ===")
for res in result:
    print(f"车牌号: {res[1]}, 置信度: {res[2]:.2f}")