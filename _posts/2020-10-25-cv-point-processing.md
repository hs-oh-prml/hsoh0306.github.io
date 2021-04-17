---
layout: post
title: "Point Processing: Arithmetic processing&Power-law transformation"
tags: [Computer Vision]
---

### Point Processing
$$G(x,y)=T(f(x,y))$$
- $$G()$$: 결과 값
- $$T()$$: 변환 함수
- $$f(x,y)$$:  (x,y) 위치의 픽셀 값

모든 픽셀에 대하여 수행하지만, 공간적인 정보를 사용하지 않음

Simple gray level transformation
- Image negative
- Log transformation
- Power-law transformation
- Thresholding
- Gray-level slicing, Bit-plane slicing
- Contrast stretching

Histogram processing
- Histogram equalization


### Basic Point Processing
<center>
<img src="/assets/img/point_processing.png">
</center>

Point Processing의 기본
- Negative: 반전
- Log&Inverse log: Log 함수 이용
- Root&Power: 지수 함수 이용
- Identity: 원본 

Arithmetic/logic operation
- +, -: 영상의 밝기를 밝게 하거나 어둡게 한다. 
- /, *: 영상의 명암 대비를 높이거나 낮춘다. 

<center>
<img src="/assets/img/arithmetic_result.jpg">
</center>

- 왼쪽부터 원본, +100, -100, x1.5, /2

``` python
import cv2

def plus(img, value):               # +, - 연산 함수
    for i, arr in enumerate(img):
        for j, v in enumerate(arr):
            if v + value < 0:       # 언더플로일 경우 0으로
                img[i,j] = 0
            elif v + value > 255:   # 오버플로일 경우 255로
                img[i,j] = 255
            else: 
                img[i,j] = v + value
    return img

def multiply(img, value):           # *, / 연산 함수
    for i, arr in enumerate(img):
        for j, v in enumerate(arr):
            if v * value > 255:     # 오버플로일 경우 255로
                img[i,j] = 255
            else: 
                img[i,j] = v * value
    return img

img = cv2.imread('april.jpg', cv2.IMREAD_GRAYSCALE)
pls = cv2.imread('april.jpg', cv2.IMREAD_GRAYSCALE)
mns = cv2.imread('april.jpg', cv2.IMREAD_GRAYSCALE)
mul = cv2.imread('april.jpg', cv2.IMREAD_GRAYSCALE)
div = cv2.imread('april.jpg', cv2.IMREAD_GRAYSCALE)

pls = plus(pls, 100)
mns = plus(mns, -100)
mul = multiply(mul, 1.5)
div = multiply(div, 0.5)

res = cv2.hconcat([img, pls, mns, mul, div])

cv2.imshow('arithmetic processing', res)
cv2.imwrite('arithmetic_result.jpg', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

이미지에서 산술 연산을 할 때, 오버플로랑 언더플로를 처리해줘야한다. 그렇지 않고

```python 
pls = pls + 100
mns = mns - 100
mul = mul * 2
div = div // 2
```
단순하게 이렇게 짜면 다음과 같은 결과가 나온다.

<center>
<img src="/assets/img/arithmetic_result_fail.jpg">
</center>

각 픽셀이 uint8 자료형으로 다루어지기 때문에 값이 255를 넘어가면 오버플로, 0 이하로 떨어지면 언더플로가 발생한다. 여기서는 간단하게 함수를 만들어서 실습했는데 opecv에서 산술 연산에 대한 함수를 제공한다. 

``` python
pls = cv2.add(pls, 100)
mns = cv2.subtract(mns, 100)
mul = cv2.multiply(mul, 1.5)
div = cv2.divide(div, 2)
```

제공되는 함수를 사용하면 오버플로랑 언더플로에 대한 처리는 물론이고 속도도 직접 만든 함수보다 훨씬 빠르다.

Power-Law Transformation

Gamma Transformation이라고도 하며 지수함수를 이용한 point processing 기법이다.


<center>
<img src="/assets/img/power_law_trans.png">
</center>

> <center> $$s = cr^{\gamma}$$  </center>
> <center> $$c = 1$$ </center>

power-law transformation을 하기 위해서 pixel의 값을 [0, 255]가 아닌 [0, 1]로 정규화한다. $$c$$는 1이고, $$\gamma$$ 값은 변수다. $$\gamma$$가 1보다 크면 영상은 어두워지고, 1보다 작으면 영상은 밝아진다. 

<center>
<img src="/assets/img/power_law_result.jpg">
</center>


``` python
import cv2
import numpy as np

img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)

hconcat = []
vconcat = []
for gamma in [1, 0.3, 0.5, 0.7, 2, 3, 4, 5]:
    res = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
    if gamma == 1:
        cv2.putText(res, "original", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 2, cv2.LINE_AA)
    elif gamma < 1:
        cv2.putText(res, "gamma: {}".format(gamma), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 20, 2, cv2.LINE_AA)
    else:
        cv2.putText(res, "gamma: {}".format(gamma), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 200, 2, cv2.LINE_AA)
    vconcat.append(res)

    if len(vconcat) == 2: 
        # print(vconcat)
        temp = cv2.vconcat(vconcat)
        hconcat.append(temp)
        vconcat = []

result = cv2.hconcat(hconcat)

cv2.imshow('image', result)
cv2.imwrite('power_law_result.jpg', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
```