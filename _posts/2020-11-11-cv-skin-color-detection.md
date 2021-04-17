---
layout: post
title: Skin Color Detection
tags: [Computer Vision]
---

강의자료에 재밌어 보이는 내용이 있어서 따로 포스팅한다. 사람의 피부색은 인종에 상관 없이 일정한 범위 내에 값을 가져서 색을 이용하여 피부색을 검출 할 수 있다고 한다. "Human Skin Detection Using RGB, HSV and YCbCr Color Models"는 색을 이용하여 사람의 피부를 검출하는 것을 연구한 논문이다. 기계학습을 이용하지 않고 threshold를 이용한 방법으로, 여러 color space를 조합한 방법을 제시했다. 구현하기 전에 Color space에 대해 간략하게 알아보자.

#### Red, Green, and Blue (RGB) Color Model

RGB는 가장 널리 사용되고 있는 color model이다. 모든 색을 빨강(R), 초록(G), 파랑(B)의 합으로 표현하며, 디지털 이미지의 저장에 사용된다. 보통 하나의 color당 8-bit를 할당하며, [0, 255] 사이 값을 가진다. normalized RGB는 이를 [0, 1]로 정규화한 표현으로 다음과 같이 쓰면 된다.

$$ 
\begin{aligned}
r = {R \over R + G + B} \\
g = {G \over R + G + B} \\
b = {B \over R + G + B}
\end{aligned}
$$

<center>
<img src="/assets/img/rgb_model.jpg">
</center>

#### YCbCr (Luminance, Chrominance) Color Model
YCbCr은 encoded non-linear RGB라고 한다. 압축할 때 많이 써서, TV 송신, 비디오,  JPEG, MPEG1, MPEG2, MPEG4 등에 사용된다. YCbCr에서 Y는 luminance, Cb는 blue값과 기준 값(Y)의 차이,  Cr은 red값과 기준 값(Y)의 차이를 나타낸다. 

$$
\begin{aligned}
Y = 0.299R + 0.287G + 0.11B \\
Cr = R - Y \\
Cb = B - Y
\end{aligned}
$$

<center>
<img src="/assets/img/ycbcr_model.jpg">
</center>

#### Hue Saturation Value (HSV) Color Model

HSV는 RGB보다 사람이 경험하는 색을 더 직관적으로 나타내는 model이다. HSV에서 H(Hue)는 [0, 1.0] 범위를 가지며 red, yellow, green, cyan, blue, magenta 그리고 다시 red를 나타낸다. S(Saturation)는 [0, 1.0]에 값을 가지면 색이 얼마나 saturate한지(0: gray, 1.0: no white componet)를 나타낸다.  V(Value)는 밝기 값을 의미하며 [0, 1.0]의 값을 가진다. H를 각도로 표현하여 [0, 360]도로 표현하는 방법도 있다. opencv에서는 HSV은 각각 H: [0, 180], S: [0, 255], V: [0, 255]로 표현된다.

<center>
<img src="/assets/img/hsv_model.jpg">
</center>

### Proposed Skin Detection Algorithm
논문에서는 skin color detection을 위한 2가지 방법을 제시했다. 하나는 ARGB와 HSV를 사용한 방법이고, 다른 하나는 ARGB와 YCbCr을 이용한 방법이다. ARGB는 RGB space에 투명도 A(alpha)를 추가한 model이다. 

<pre>
<code>
ARGB&HSV

if (0.0 <= H <= 50.0 & 0.23 <= S <= 0.68 
    & R > 95 & G > 40 & B > 20 & R > G & R > B
    & | R - G | > 15 & A > 15){
    skin color
    }
else {
    non-skin color
}
</code>
</pre>

<pre>
<code>
ARGB&YCbCr

if (R > 95 & G > 40 & B > 20 & R > G & R > B
    & | R - G | > 15 & A > 15 & Cr > 135 &
    Cb > 85 & Y > 80 & Cr <= (1.5862*Cb)+20 &
    Cr>=(0.3448*Cb)+76.2069 &
    Cr >= (-4.5652*Cb)+234.5652 &
    Cr <= (-1.15*Cb)+301.75 &
    Cr <= (-2.2857*Cb)+432.85) {
    skin color
    }
else {
    non-skin color
}
</code>
</pre>
> (H : Hue ; S: Saturation ; R : Red ; B: Blue ; G : Green ; Cr, Cb : Chrominance components ; Y : luminance component )

```python 
import cv2
import numpy as np

img = cv2.imread('face.jpg')

height, width, _ = img.shape

argb = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)        # ARGB 
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)          # HSV
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)     # YCrCb

norm_hsv = np.zeros((height, width, 3), np.float)   
norm_hsv[:,:,0] = hsv[:,:,0] * 2
norm_hsv[:,:,1] = hsv[:,:,1] / 255
norm_hsv[:,:,2] = hsv[:,:,1] / 255

res1 = np.zeros((height, width, 3), np.uint8)       # 첫번째 방법 결과
res2 = np.zeros((height, width, 3), np.uint8)       # 두번째 방법 결과

for i in range(height):
    for j in range(width):
        r = argb[i, j, 0]
        g = argb[i, j, 1]
        b = argb[i, j, 2]
        a = argb[i, j, 3]

        h = norm_hsv[i, j, 0]
        s = norm_hsv[i, j, 1]

        y = ycrcb[i, j, 0]
        cr = ycrcb[i, j, 1]
        cb = ycrcb[i, j, 2]

        # opencv에서 hsv의 범위는 h: [0, 180], s: [0, 255], v: [0, 255]이다.
        # ARGB
        if r > 95 and g > 40 and b > 20 and r > g and r > b and abs(r - g) > 15 and a > 15:
            # HSV
            if h >= 0.0 and h <= 50.0 and s >= 0.23 and s <= 0.68:
                res1[i,j,0] = b
                res1[i,j,1] = g
                res1[i,j,2] = r
            # YCrCb
            if cb > 85 and y > 80 and cr <= (1.5862*cb) + 20 and cr > (0.3448 * cb) + 76.2069 and cr >= (-4.5652 * cb) + 234.5652 and cr <= (-1.15 *cb) + 301.75 and cr <= (-2.2857 * cb) + 432.85:
                res2[i,j,0] = b
                res2[i,j,1] = g
                res2[i,j,2] = r


result = cv2.hconcat([img, res1, res2])
cv2.imshow('skin color detection', result)
cv2.imwrite('skin_color.png', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<center>
<img src="/assets/img/skin_color_1.jpg">
</center>
<center>
<img src="/assets/img/skin_color_2.jpg">
</center>
<center>
<img src="/assets/img/skin_color_3.jpg">
</center>
<center>
<img src="/assets/img/skin_color_4.jpg">
</center>



> 왼쪽부터 원본, ARGB&HSV 방법을 사용한 결과, ARGB&YCbCr 방법 사용한 결과

threshold만 사용한 방법이라 좀 애매할 거 같았는데 생각보다 꽤 괜찮은 결과를 내었다. 인종에 크게 상관이 없는 거 같고, HSV보다 YCbCr을 사용한 것이 결과가 더 잘나오는 거 같다. 


### Reference
[S. Kolkur, D. Kalbande, P. Shimpi, C. Bapat, and J. Jatakia,'Human Skin Detection Using RGB, HSV and YCbCr Color Models', *ICCASP/ICMMD*, 2016, Vol. 137, pp.324-332](https://arxiv.org/ftp/arxiv/papers/1708/1708.02694.pdf)
