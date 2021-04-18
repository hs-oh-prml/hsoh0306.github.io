---
layout: post
title: "Mid Level Image Feature: Texture"
tags: [Computer Vision]
---

## Texture 
- Texture는 이미지를 interest region으로 나누고 그런 region을 classify하기 위해 사용되는 feature다.
- Texture는 이미지의 color 또는 intensity의 공간적 정보를 제공한다.
- Texture는 인접 pixel들의 intensity level의 공간 분포로 특징된다.
- Texture는 이미지 intensity에서 local variation의 반복되는 pattern이다. 따라서 point에서 정의될 수 없다.

### Approach
1. Structural: texture는 규칙적이고 반복적인 primitive texel의 set이다.
2. Statistical: texture는 region에서 intensity arrangement의 정량적인 measure이다. 이런 측정의 집합을 feature vector라고 부른다.
3. Modeling: texture modeling 기술은 특정 texture를 구축하는 모델을 포함한다.

#### Aspects of Texture
- Size/Granularity
- Directionality/Orientation
- Random or regular

#### Structural Approach to Texture
- Texel을 정의해야함
    - real image에서 texel을 정의하고 추출하는 것이 어렵거나 불가능함

#### Statistical Approach to Texture
- gray-scale intensity(or color)에서 통계적 측정으로 texture를 characterize한다.
- 직관적이지 않지만, 모든 이미지에서 적용이 가능하고, 연산이 효율적이다.
- classification이나 segmentation에서도 모두 사용할 수 있다.

#### Texture Analysis
- Texture segmentation
    이미지 내에서 다양한 texture region사이의 boundary를 자동으로 결정
- Texture classification
    주어진 texture class로 이미지 내의 texture region을 identifying

## Simple Statistical Texture Measure 

#### Range
One of the simplest of the texture operator is the range or difference between maximum and minimum intensity values in a neighbor

인접 pixel사이에서 최대 값과 최소 값 차이 계산
- The range operator converts the original image to one in which brightness represents texture


#### Variance
Another estimator of texture is the variance in neighborhood regions

중심 pixel과 인접 pixel 사이의 차이의 합 계산

-  This is the sum of the squares of the differences between the intensity of the central pixel and its neighbors

### Quantitative Texture Measures
1. Local Binary Pattern(LBP)
2. Gray Level Co-occurence(GLCM)

#### Local Binary Pattern(LBP)
- For each pixel $$p$$, create an 8-bit number $$b_1$$ $$b_2$$ $$b_3$$ $$b_4$$ $$b_5$$ $$b_6$$ $$b_7$$ $$b_8$$, where $$b_i = 0$$ if neighbor $$i$$ has value less than or equal to $$p$$’s value and 1 otherwise.
- Represent the texture in the image (or a region) by the histogram of these numbers

중심 pixel과 인접 픽셀간 대소 비교하여 8-bit에 저장 크면 1, 작거나 같으면 0을 표기

Rotation invariant하여 10 level로 하는 방법도 있음

#### Gray Level Co-occurrence(GLCM)
- The statistical measures described so far are easy to calculate, but do not provide any information about the repeating nature of texture.
- A gray level co-occurrence matrix(GLCM) contains information about the positions of pixels having similar gray level values.

통계적 측정은 계산이 쉽지만, 반복되는 texture 본질에 대한 정보는 제공하지 않는다.

GLCM은 similar gray level value를 가지는 pixel의 좌표 정보도 포함한다. 

A co-occurrence matrix is a two-dimensional array, P, in which both the rows and the columns represent a set of possible image values

- A GLCM $$P_d[i,j]$$ is defined by first specifying a displacement vector $$d=(dx,dy)$$ and counting all pairs of pixels separated by d having gray levels $$i$$ and $$j$$.

- The GLCM is defined by: $$P_d[i,j] = n_{ij}$$

  - $$n_{ij}$$ is the number of occurrences of the pixel values $$(i,j)$$ lying at distance $$d$$ in the image
  - The co-occurrence matrix $$P_d$$has dimension n×n, where n is the number of gray levels in the image

displacement vector $$d=(dx,dy)$$를 정의하고, 모든 pair를 counting한다. 

##### Algorithm
1. Count all pairs of pixels in which the first pixel has a value $$i$$, and its matching pair displaced from the first pixel by d has a value of $$j$$
2. This count is entered in the ith row and jth column of the matrix $$P_d[i,j]$$
2. Note that $$P_d[i,j]$$ is not symmetric, since the number of pairs of pixels having gray levels $$[i,j]$$ does not necessarily equal the number of pixel pairs having gray levels $$[j,i]$$

#### Normalized GLCM
The elements of $$P_d[i,j]$$ can be normalized by dividing each entry by the total number of pixel pairs
- Normalized GLCM, $$N[i,j]$$ is defined by:

    $$N[i,j]= {P[i,j] \over \sum_i \sum_j  P[i,j]}$$

- It normalizes the co-occurrence values to lie between 0 and 1, and allows them to be thought of as probabilites

#### Numeric Features of GLCM
- Gray level co-occurrence matrices capture properties of a texture but they are not directly useful for further analysis, such as the comparison of two textures
- Numeric features are computed from the occurrence matrix that can be used to represent the texture more compactly
    - Maximum probability
    - Moments
    - Contrast
    - Homogeneity
    - Entropy
    - Correlation 

##### Maximum Probability
- This is simply the largest entry in the matrix, and corresponds to the strongest response
    - This could be the maximum in any of the matrices or the maximum overall 
    - $$C_m=\max_{i,j} P_d[i,j]$$

##### Moments
- The order k element difference moment can be defined as:
    - $$MOM_k = \sum_i \sum_j (i-j)^k P_d[i,j]$$
- This descriptor has small values in cases where the largest elements in $$P$$ are along the principal diagonal. The opposite effect can be achieved using the inverse moment 
    - $$MOM_k = \sum_i \sum_j {P_d[i,j] \over (i-j)^k}, i \neq j$$

##### Contrast
- Contrast is a measure of the local variations present in an image 

    - $$MOM_k = \sum_i \sum_j (i-j)^k P_d[i,j]^n$$

    - If there is a large amount of variation in an image the P[i,j]’s will be concentrated away from the main diagonal and contrast will be high
    - Typically, $$k=2$$ and $$n=1$$

##### Homogeneity
-  A homogeneous image will result in a co-occurrence matrix with a combination of high and low $$P[i,j]$$’s 
    - $$C_h = \sum_i \sum_j {P_d{i,j} \over 1 + \|i-j\|}$$
    - Where the range of gray levels is small, the P[i,j] will tend to be clustered around the main diagonal
    - A heterogeneous image will result in an even spread of $$P[i,j]$$’s 

##### Entrophy
- Entropy is a measure of information content 
- It measures the randomness of intensity distribution
    - $$C_e = - \sum_i \sum_j P_d[i, j]ln P_d[i,j]$$
- Entropy is highest when all entries in $$P[i,j]$$ are of similar magnitude, and small when the entries in $$P[i,j]$$ are unequal 

##### Correlation
- Correlation is a measure of image linearity
    - $$C_e = {\sum_i \sum_j ij P_d[i,j] - \mu_i \mu_j \over \sigma_i \sigma_j}, \mu_i = \sum i P_d[i,j], \sigma_i^2= \sum i^2 P_d[i,j] - \mu_i^2$$
- Correlation will be high if an image contains a considerable amount of linear structure

#### Problem with GLCM
- One problem with deriving texture measures from co-occurrence matrices is how to choose the displacement vector $$d$$

    - The choice of the displacement vector is an important parameter in the definition of the GLCM
    - Occasionally the GLCM is computed from several values of d and the one which maximizes a statistical measure computed from $$P[i,j]$$ is used
    - Zucker and Terzopoulos used a $$\chi^2$$ measure to select the values of d that have the most structure, i.e., to maximize the value 

        - $$\chi^2(d)= \sum_i \sum_j {P_d^2[i,j] \over P_d[i] P_d[j]} - 1$$

### Edges and Texture 
- It should be possible to locate the edges that result from the intensity transitions along the boundary of the texture
    - Since a texture will have large numbers of texels, there should be a property of the edge pixels that can be used to characterize the texture
- Compute the co-occurrence matrix of an edge-enhanced image 
- Edge Density and Direction
- Use an edge detector as the first step in texture analysis
- The number of edge pixels in a fixed-size region tells us how busy that region is
- The directions of the edges also help characterize the texture

#### Two Edge-based Texture Measures 
1. Edgeness per unit area for a region R

    - $$Fedgeness = \|{ p \| gradient_magnitude(p) ≥ threshold} \| / N$$ 
    - N is the size of the unit area

2. Histograms of edge magnitude and direction for a region R 
    - $$F_magdir = ( H_magnitude, H_direction )$$
    - These are the normalized histograms of gradient magnitudes and gradient directions, respectively

#### Energy and Texture 
- One approach to generate texture features is to use local kernels to detect various types of texture
- Laws developed a texture-energy approach that measures the amount of variation within a fixed size window

#### Law's Texture Energy
- Filter the input image using texture filters
- Compute texture energy by summing the absolute value of filtering results in local neighborhoods around each pixel
- Combine features to achieve rotational invariance 

-A set of convolution mask are used to compute texture energy
- The mask are computed from the following basic mask
    - L5 (Gaussian) gives a center-weighted local average
        - $$L5 = [1,4,6,4,2]$$
    - E5 (gradient) responds to row or col step edges
        - $$E5 = [-1,-2,0,2,1]$$
    - S5 (LoG) detectss spots
        - $$S5 = [-1,0,2,0,-1]$$
    - R5 (Gabor) detects ripples
        - $$R5 = [1,-4,6,-4,1]$$
    - W5(wave) detects waves 
        - $$W5 = [-1,2,0,-2,1]$$

- The 2D convolution mask are obtained by computing the outer product of a pair of vectors
    - For example, E5L5 is computed as the product of E5 and L5 as follows 
- Bias from the “directionality” of textures can be removed by combining symmetric pairs of features, making them rotationally invariant
    - For example, S5L5 (H) + L5S5 (V) = L5S5R 

- After the convolution with the specified mask, the texture energy measure (TEM) is computed by summing the absolute values in a local neighborhood
    - $$L_e = \sum_{i=1}^m \sum_{j=1}^n \|C(i,j)\|$$

- If n masks are applied, the result is an n-dimensional feature vector at each pixel of the image being analyzed 

#### Algorithm
1. Apply convolution masks
2. Calculate the texture energy measure (TEM) at each pixel. This is achieved by summing the absolute values in a local neighborhood
3. Normalize features – use L5L5 to normalize the TEM image

- Subtract mean neighborhood intensity from pixel (to reduce illumination effects)
- Filter the neighborhood with 16 masks
- Compute energy at each pixel by summing absolute value of filter output across neighborhood around pixel
- Define 9 features as follows (replace each pair with average)
- L5E5 / E5L5, L5R5 / R5L5
- E5S5 / S5E5, L5S5 / S5L5
- E5R5 / R5E5, S5R5 / S5R5
- R5R5, S5S5, E5E5

### Autocorrelation for texture 
- Autocorrelation function computes the dot product (energy) of original image with shifted image for different shifts
    - $$\rho (dr, dc) = {\sum_i \sum_j I[i, j]I[i+dr, j+dc] \over \sum_i \sum_j I^2[i, j]}= {I[i, j] \circ I_d[i ,j] \over I[i, j] \circ I[i, j]}$$

- It can detect repetitive patterns of texels
- Also it can captures fineness/coarseness of the texture 

- Regular textures : function will have peaks and valleys
- Random textures: only peak at [0,0] and breadth of peak gives the size of the texture
- Coarse texture: function drops off slowly
- Fine texture : function drops off rapidly
- Can drop differently for row and column 