---
layout: post
title: "[Computer Vision] Color&Histogram"
categories: 
 - ComputerVision
use_math: true
---

### Basic Techinques in Digital Image Processing
- Image Representation&Storage
- Image Enhancement&Filtering
  - Contrast stretching, smoothing, sharpening
- Feature Extraction 
  - Color: color space, histogram
  - Texture: Wavelet transformation, Discrete cosine transformation
  - Shape: Edge detector, HOG, SIFT/SURF

### Color
사람의 시각에서 색은 중요한 기능을 한다. 컴퓨터에서는 pixel 단위로 색을 표현한다. 사람은 약 400nm에서 700nm 사이의 파장을 인식할 수 있지만, 기계는 X-ray나 적외선, 라디오 주파수같이 더 다양한 것을 인식할 수 있다. 

### 색의 표현 
- RGB: Red, Green, Blue로 표현되는 가산 시스템으로 하드웨어의 출력에 많이 사용된다.
- CMY: Cyan, Magenta, Yellow로 표현되는 감산 시스템으로 프린팅에 많이 사용된다. Black을 추가한 CMYK를 사용하기도 한다.
- HSI(Value): Hue, Saturation, Intensity(Value)로 표현하면 사람의 시각 체계와 가장 유사한 표현 방법이다.  
- YIQ: 압축률이 좋아 TV 방송 송출에 사용된다.

### Saturation 변환
<center>
<img src="/assets/img/saturation_result.jpg">
</center>

- 왼쪽: 원본 이미지, 중간: Saturation 1.8배, 오른쪽: Saturation 0.4배

Saturation을 높이면 이미지의 색감이 더 풍부해지고, 낮추면 전체적으로 물빠진 색이 된다. opencv2를 이용하여 간단하게 실습해볼 수 있다. 

```python
import cv2

img = cv2.imread('cat.jpg')                                 # 이미지 로드
saturation_up = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)        # BGR 색공간을 HSV로 변환
saturation_down = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)      # BGR 색공간을 HSV로 변환

saturation_up[:, :, 1] = saturation_up[:, :, 1] * 1.8       # Saturation 조정: 1.8배
saturation_down[:, :, 1] = saturation_down[:, :, 1] * 0.4   # Saturation 조정: 0.4배

saturation_up = cv2.cvtColor(saturation_up, cv2.COLOR_HSV2BGR)      # HSV 색공간을 다시 BGR로 변환
saturation_down = cv2.cvtColor(saturation_down, cv2.COLOR_HSV2BGR)  # HSV 색공간을 다시 BGR로 변환

cv2.imshow("image", img)                                    # 원본 이미지 
cv2.imshow("saturation_up", saturation_up)                  # Saturation * 1.8 이미지
cv2.imshow("saturation_down", saturation_down)              # Saturation * 0.4 이미지

result = cv2.hconcat([img, saturation_up, saturation_down]) # 결과 이미지: 원본, Saturation * 1.8, Saturation * 0.4 
cv2.imwrite('saturation_result.jpg', result)                # 이미지 저장 

cv2.waitKey(0)                                              # 키입력 대기 
cv2.destroyAllWindows()                                     # 모든 창 닫기
```

### Histogram 
- Color histogram으로 이미지를 표현할 수 있다.
- 히스토그램은 빠르고 쉽게 연산이 가능하다. 
- 정규화하여 다른 이미지의 히스토그램과 비교가 가능하다.
- 데이터베이스의 쿼리로 사용할 수 있고, classification에 사용할 수도 있다. 

<center>
<img src="/assets/img/histogram_hsv.png">
</center>

- HSV 이미지에 대한 histogram

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('cat.jpg')                                     # 이미지 로드
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                  # 이미지 변환: BRG->HSV

hist_h = cv2.calcHist([img_hsv], [0], None, [180], [0, 180])    # Histogram: H  [0, 180]
hist_s = cv2.calcHist([img_hsv], [1], None, [256], [0, 255])    # Histogram: S  [0, 255]
hist_v = cv2.calcHist([img_hsv], [2], None, [256], [0, 255])    # Histogram: V  [0, 255]

print(np.max(img_hsv[:,:,0]))
print(np.max(img_hsv[:,:,1]))
print(np.max(img_hsv[:,:,2]))

# 그래프 
plt.figure("Histogram: HSV")
plt.subplot(221)
plt.title('Image')
plt.imshow(img)
plt.subplot(222)
plt.title('Hue')

plt.plot(hist_h)
plt.subplot(223)
plt.title('Saturation')

plt.plot(hist_s)
plt.subplot(224)
plt.title('Intensity')
plt.plot(hist_v)
plt.tight_layout()
# figure 저장
plt.savefig('histogram_hsv.png', dpi=300)
plt.show()
```

### Reference
[Elephant Image](https://medium.com/@jovana.savic9494/image-contrast-increase-866b7eeac8c2)
