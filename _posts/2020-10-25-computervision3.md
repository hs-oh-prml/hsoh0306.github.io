---
layout: post
title: "[Computer Vision] Mid Level Image Features: Shapes "
categories: 
 - ComputerVision
use_math: true
---

## Shapes 
Shape은 color랑 texture보다 더 나아간 단계의 feautre다. Color랑 texture는 둘 다 global attribute지만, shape은 특정 region에 대한 attribute이다. 

### Shape Descriptors
- Region Descriptor
- Boundary
- Interest points(corners)

### Region based Shape Descriptors 
#### Geometric and Shape Properties 
- area
- centroid
- perimeter
- perimeter length
- circularity, elongation
- mean and standard deviation of radial distance
- second order moments (row, column, mixed)
- bounding box
- extremal axis length from bounding box
- lengths and orientations of axes of best-fit ellipse

*Often want features independent of position, orientation, scale* 

#### Zero-order moment
Moment를 사용하는 이유: 
서로 다른 차수의 기하학적 모멘트는 이미지 분포의 공간적 특성을 나타낸다.
Zero-order moment란: 
- $A = \sum_i^n \sum_j^m {B[i,j]}$
- 전체 intensity를 의미
- binary image에서는 area를 의미

#### Centroid 
이미지 내의 object의 위치가 공간 위치에 의해 정해진다.

Center of area(centroid, center of mass): First order moment

- $\bar{x} = {\sum_i^n \sum_j^m {jB[i,j]} \over A}$ (object의 pixel간에 j 좌표의 평균)
- $\bar{y} = {\sum_i^n \sum_j^m {iB[i,j]} \over A}$ (object의 pixel간에 i 좌표의 평균)

- intensity centroid 
- binary image의 기하학적 중심
- Centroid는 분석과 매칭에서 interesting point가 될 수 있다.

#### Second moments

Second-order row moment
- $\mu_{rr} = {1 \over A} \sum_{(r,c) \in R}  {(r - \bar{r})^2}$ 

Second-order mixed moment
- $\mu_{rc} = {1 \over A} \sum_{(r,c) \in R}  {(r - \bar{r})(c - \bar{c})}$ 

Second-order column moment
- $\mu_{cc} = {1 \over A} \sum_{(r,c) \in R}  {(c - \bar{c})^2}$

#### Moment Invariants
기하 변형: translation, scale, mirroring, rotation

#### Perimeter and Perimeter Length

Perimeter

- $ P_4 = \\{ (r,c) \in R \| N_8(r,c) - R \neq \varnothing \\} $

- $ P_8 = \\{ (r,c) \in R \| N_4(r,c) - R \neq \varnothing \\} $

Perimeter Length
- $ \| P \| = \| \\{ k \| (r_{k+1}, c_{k+1}) \in N_4(r_k, c_k) \\} \| + \sqrt(2) \| \\{ k \| ( r_{k+1}, c_{k+1}) \in N_8(r_k, c_k) - N_4(r_k, c_k) \\} \|$

#### Circularity
간단하게 circularity를 측정하는 방법은 둘레의 제곱을 면적으로 나누는 것이다. 
- $C_1 = { \| P \| ^2 \over A}$
같은 면적일때, 원에 가까울수록 둘레가 작아지므로 $C_1$은 작아진다. 
그렇게 할 경우 정사각형이 원보다 circularity가 좋게 나오는 경우가 발생하기 때문에 다음과 같은 방법이 있다.

- $C_2 = { \mu_R \over \sigma_R}$
- $\mu_R = {1 \over K} \sum_{k=0}^{K-1} \parallel (r_k, c_k) - (\bar{r}, \bar{c}) \parallel$
- $\sigma_R^2 = {1 \over K} \sum_{k=0}^{K-1} [\parallel (r_k, c_k) - (\bar{r}, \bar{c}) \parallel - \mu_R]^2$

분산과 표준 편차를 이용하여 계산하는 방법이다. 원에 가까울 수록 표준편차가 작아지므로, $C_2$는 커진다. 

#### Orientation
object의 방향을 연장선의 축(axis) 방향으로 정의한다.
- second order moment(분산, 데이터의 퍼짐)를 최소화하는 부분
- $\min_{line} \chi^2 = \min_{line} \sum_{i=1}^n \sum_{j=1}^n {r_{ij}^2 B[i,j]}$


$r_{ij}^2$는 object 내의 점 $[i,j]$과 축의 수직 거리다. 

직선을 극좌표로 표현
- $y = ax + b$
- $(x,y) \centerdot (\cos{\theta}, \sin{\theta}) = \rho $
- $\centerdot$ is projection$
- $x\cos{\theta} + y\sin{\theta}=\rho$

Axis with Least Second Moment

- $\tan{2\alpha} = {2\sum(r-\bar{r})(c-\bar{c}) \over \sum(r-\bar{r})(r-\bar{r}) \sum(c-\bar{c})(c-\bar{c})}$
- $ = { {1 \over A} 2\sum(r-\bar{r})(c-\bar{c}) \over {1 \over A} \sum(r-\bar{r})(r-\bar{r}) {1 \over A} \sum(c-\bar{c})(c-\bar{c})} $
- $ = {2 \mu_{rc} \over \mu_{rr} - \mu_{cc}}$

#### Topological Region Descriptors
Hole Counting
- external corner has 3(1)s and 1(0)
- internal corner has 3(0)s and 1(1)
- Holes computed from only these patterns!

Algorithm
<pre>
<code>
Input a binary image and output the number of holes it contains 

- M is a binary image of R rows of C columns 
- 1 represents material through which light has not passed
- 0 represents absence of material indicated by light passing 
- Each region of 0s must be 4-connected and all image border pixels must be 1s 
- E is the count of *external corners* (3 ones and 1 zero) 
- I is the count of *internal corners* (3 zeros and 1 one) 

integer procedure Count_Holes(M)
{
    examine entire image, 2 rows at a time;
    count external corners E;
    count internal corners I;
    return (number_of_holes = (E-I)/4);
}
</code>
</pre>