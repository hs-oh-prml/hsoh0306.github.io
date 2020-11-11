---
layout: post
title: "[Computer Vision] Contrast"
categories: 
 - ComputerVision
use_math: true
---

### Contrast
대비(Contrast)

- 가장 어두운 영역으로부터 가장 밝은 영역의 범위

> <center>$Contast = {I_{max}-I_{min} \over I_{max} + I_{min}}$</center>

- 지각 작용은 순수한 광도의 강도에 민감하기보다는 광도의 대비에 더 민감하다. 

<center>
<img src="/assets/img/contrast1.png">
</center>

- 두 이미지의 중앙의 색은 같지만 대비가 더 큰 왼쪽의 이미지가 부각되어 보인다.

Mach Band: 서로 다른 광도가 인접해 있는 경우 발생하는 효과
- 광도가 급격히 변화하는 것에 대한 시각 시스템의 반응은 경계 부분을 더 강조하여 보는 경향이 있다.

<center>
<img src="/assets/img/mach_band.jpg">
</center>


### Contrast Stretching
- Contrast stretching은 이미지의 intensity 범위를 늘려 원하는 값 범위에 걸쳐 이미지의 대비를 향상시키는 이미지 enhancement 기술이다.
- Basic contrast stretching
- Ends-in-search
- Simple transformation
- Histogram processing
  - Histogram equalization
  - Histogram specification
  


### Basic contrast stretching
- 특정부분, 중앙에 명암 값이 치우치는 히스토그램을 가진 영상에 가장 잘 적용
- 모든 범위의 화소 값을 포함하도록 영상을 확장

> <center>$new pixel = {old pixel - low \over high - low} * 255$</center>

<center>
<img src="/assets/img/basic_contrast_stretching.png">
</center>

- 낮은 명암대비를 가진 영상의 질을 향상시킬 수있는 유용한 도구로서 가우시안(Gaussian) 분포를 가질때 가장 잘 적용

<center>
<img src="/assets/img/basic_contrast_stretching_fail.png">
</center>

반면에 intensity가 전역에 퍼져있으면 전혀 효과가 없음을 알 수 있다.


### Ends-in search
- 모든범위의 명암값을 갖지만 히스토그램의 특정 부분에 화소들이 치우친 영상에 가장 잘 적용
- 일정한 양의 화소를 흰색 또는 검은색을 갖도록 지정
- 알고리즘: 2개의 임계값(low, high)을 사용
  - low : 낮은 범위에서 지정한 양 이후의 화소의 pixel intensity
  - high: 높은 범위에서 지정한 양 이후의 화소의 pixel intensity


<center>
<img src="/assets/img/ends_in_search.png">
</center>

low 값을 30으로, high 값을 220으로 정하고 ends-in search한 결과다. Basic contrast stretching으로 대비 개선 할 수 없었던 이미지에 대하여, ends-in search는 효과가 있는 것을 알 수 있다.

<center>
<img src="/assets/img/compare_result.jpg">
</center>


```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

def basic_contrast_stretching(img):                                 # Basic Contrast Stretching
    height, width = img.shape                                       # 이미지 크기 
    res = np.zeros(shape=(height, width), dtype=np.uint8)           # 결과 이미지
    low = np.min(img)                                               # 최소 값
    high = np.max(img)                                              # 최대 값

    for i in range(height):
        for j in range(width):
            res[i, j] = np.uint8(((img[i, j] - low)/(high - low)) * 255)
    return res

def ends_in_search(img, low, high):                                 # Ends-in Search
    height, width = img.shape                                       # 이미지 크기 
    res = np.zeros(shape=(height, width), dtype=np.uint8)           # 결과 이미지
    for i in range(height):
        for j in range(width):
            if img[i, j] < low:
                res[i, j] = 0
            elif img[i, j] > high:
                res[i, j] = 255
            else:
                res[i, j] = np.uint8(((img[i, j] - low)/(high - low)) * 255)
    return res
# img = cv2.imread('bean.jpg', cv2.IMREAD_GRAYSCALE)              # 이미지 로드
img = cv2.imread('elephant.jpeg', cv2.IMREAD_GRAYSCALE)              # 이미지 로드

bcs = basic_contrast_stretching(img)
eis = ends_in_search(img, 30, 220)

# Histogram 
hist_o = cv2.calcHist([img], [0], None, [256], [0, 255])    
hist_bcs = cv2.calcHist([bcs], [0], None, [256], [0, 255])    
hist_eis = cv2.calcHist([eis], [0], None, [256], [0, 255])    

# 그래프 
plt.figure("Histogram")
plt.subplot(321)
plt.title('Image')
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(322)
plt.title('Histogram')
plt.bar(range(len(hist_o)), hist_o[:, 0])
plt.subplot(323)
plt.title('Basic Contrast Stretching')
plt.imshow(bcs, cmap='gray', vmin=0, vmax=255)
plt.subplot(324)
plt.title('Basic Contrast Stretching Result Histogram')
plt.bar(range(len(hist_bcs)), hist_bcs[:, 0])
plt.subplot(325)
plt.title('Ends-in Search')
plt.imshow(eis, cmap='gray', vmin=0, vmax=255)
plt.subplot(326)
plt.title('Ends-in Search Result Histogram')
plt.bar(range(len(hist_eis)), hist_eis[:, 0])

plt.tight_layout()
plt.savefig('ends_in_search.png')
plt.show()

cv2.putText(img, "Image", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
cv2.putText(bcs, "Basic Contrast Stretching", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
cv2.putText(eis, "Ends-in Search", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)

result = cv2.hconcat([img, bcs, eis])
cv2.imwrite('compare_result.jpg', result)


```

### Histogram Equalization
Histogram Equalization은 두 가지를 충족시켜야한다.

- output image는 gray-level 모든 값을 사용해야한다.
- output image의 모든 픽셀은 gray-level의 값을 골고루 분포해야한다. 

#### Process
1. input image의 히스토그램을 만든다.
2. normalized sum of histogram을 계산하여 look-up table을 만든다.
3. look-up table을 기반으로 input image를 tranform한다.

#### Pseudo code
<pre>
<code>
/* clear histogram to 0 */
for(i = 0; i < 256; i++){
  histogram[i] = 0;
}
/* caculate histogram */
for(i = 0; # of pixel; i++){
  histogram[buffer[i]]++;
}
/* calculate normalized sum of histogram */
sum = 0;
scale_factor = 255.0
for(i = 0; i < 256; i++){
  sum += histogram[i];
  sum_hist[i] = (sum * scale_factor) + 0.5;
}
/* transform image using new sum_histogram as a LUT */
for(i = 0; i < 256; i++){
  buffer[i] = sum_hist[buffer[i]];
}
</code>
</pre>

<center>
<img src="/assets/img/histogram_equalization.png">
</center>

Color image에 대해서 histogram equalization을 하고 싶다면 HSV로 변환하여 intensity를 함수에 집어 넣으면 된다. 


<center>
<img src="/assets/img/histogram_equalization_color.png">
</center>

꽤 잘작동한다는 것을 확인할 수 있다. 

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

def histogram_equalization(img):                                      # Histogram Equalization
    height, width = img.shape                                         # 이미지 크기 
    res = np.zeros(shape=(height, width), dtype=np.uint8)             # 결과 이미지
    
    sum = 0
    scale_factor = 255 / (height * width)                           
    sum_hist = np.zeros(shape=(256, ), dtype=np.uint8)
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])

    for i in range(len(hist)):                                      # Normalizaed sum histogram 계산
        sum = sum + hist[i]
        sum_hist[i] = np.uint8(sum * scale_factor + 0.5)

    for i in range(height):                                         # Look-up table 기반으로 transform
        for j in range(width):
            res[i, j] = sum_hist[img[i, j]]
    return res

### RGB channel  
# img = cv2.imread('april_2.jpg')                                   # 이미지 로드
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                        # matplot에 이미지를 올리기 위해 RGB로 변환

# hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)                        # HSV로 변환
# hist_o = cv2.calcHist([hsv], [2], None, [256], [0, 255])          # Intensity histogram 계산

# hsv[:,:,2] = histogram_equalization(hsv[:,:,2])                   # Intensity channel만 histogram equalization
# hist_he = cv2.calcHist([hsv], [2], None, [256], [0, 255])         # 결과 Intensity histogram 계산

# he = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)                         # 다시 RGB로 변환

### Gray level 
img = cv2.imread('elephant.jpeg', cv2.IMREAD_GRAYSCALE)             # 이미지 로드
he = histogram_equalization(img)

### OpenCV function                                                 # OpenCV에서 제공하는 함수
# he = cv2.equalizeHist(img)

# Histogram 
hist_o = cv2.calcHist([img], [0], None, [256], [0, 255])
hist_he = cv2.calcHist([he], [0], None, [256], [0, 255])

# 그래프 
plt.figure("Histogram")
plt.subplot(221)
plt.title('Image')
# plt.imshow(img)                               # RGB Channel  
plt.imshow(img, cmap='gray', vmin=0, vmax=255)  # Gray level

plt.subplot(222)
plt.title('Histogram')
plt.bar(range(len(hist_o)), hist_o[:, 0])
plt.subplot(223)
plt.title('Histogram Equalization')
# plt.imshow(he)                                # RGB Channel
plt.imshow(he, cmap='gray', vmin=0, vmax=255)   # Gray level

plt.subplot(224)
plt.title('Histogram Equalization Result Histogram')
plt.bar(range(len(hist_he)), hist_he[:, 0])

plt.tight_layout()
# plt.savefig('histogram_equalization_color.png')
plt.savefig('histogram_equalization.png')
plt.show()

```

물론 opencv에서 같은 기능으로 equalizeHist라는 함수를 제공한다. 그냥 이거 쓰면 될거같다.
