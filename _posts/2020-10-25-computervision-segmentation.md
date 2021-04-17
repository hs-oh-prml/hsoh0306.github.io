---
layout: post
title: "Region Segmentation"
tags: [Computer Vision]
use_math: true
---

## Image Segmentation

Image segmentation이란 이미지를 여러 개의 픽셀 집합(region)으로 나누는 것이다. 

- 목표: 이미지 내에서 coherent region을 찾는 것
Coherent region은 비슷한 속성을 가지는 픽셀들을 포함한다.

- 장점: noise에 강하다.
- 단점: Oversegmented, Undersegmented 문제 발생

### Creteria
- $$\cup S_i = S$$ : 모든 region의 합은 원본 이미지와 같다. 
- $$S_i \cap S_j = \emptyset, i \neq j$$: 각 region은 overlap이 되지 않는다.  
- $$\forall S_i, P(S_i) = true$$: 같은 region에 있는 pixel들은 similarity property를 가진다. 
- $$P(S_i \cup S_j) = false$$: 다른 region의 pixel과는 dissimilarity하다. 

### Region Growing 
- region일 가능성이 높은 한 pixel에서 시작
- 인접 pixel과 비교하여 dissimilar한 pixel을 만나기 전까지 확장
- - seed pixel은 image 내에서 unlabelled pixel로 설정한다.
- 통계적 테스트를 통해서 어떤 pixel이 region이 될 가능성이  큰 지 결정한다.

### Split&Merge
- 영상을 4분할 하고, 모든 region $$R_i$$의 $$Q(R_i)=false$$이다.
- 더 이상 분할 할 수 없을 때까지 분할하고, $$Q(R_i \cup R_j) = true$$인 인접 region끼리 merge한다.
- 더 이상 merge할 수 없으면 종료

### Clustering
- object의 set을 grouping하는 것
- 같은 group의 object(cluster)는 서로 similar하다.
- 어떤 cluster object는 다른 cluster의 object와 다르다.
- ex) Connectivity model, centroid model, distribution model, density model, graph based model, hard clustering, soft clustering... 

### Clustering: Centroid Model
- 연산량이 작다
- 사용자가 classifying하기 전에 cluster의 수를 결정해줘야한다.
- ex) K-means method
- K 개의 클러스터
- Least-squares error: $$D=\sum_{k=1}^K \sum_{x_i \in C_k} \| x_i - m_k \| ^2$$
- K cluster가 가능한 모든 점에서 $$D$$를 최소화하는 곳 찾기

### K means clustering
n-dimensional vector에서 K-means cluster
1. ic(iteration count)를 1로 설정
2. random으로 K means set, means $$m_1(1),..., m_k(1)$$을 고른다.
3. 각 vector $$x_i$$에 대하여 $$D(x_i, m_k(ic)), k=1,...,K$$ 계산하고, $$x_i$$를 가장 가까운 mean의 cluster $$C_j$$에 할당
4. ic를 1증가 시키고, means $$m_1(ic),..., m_k(ic)$$로 업데이트
5. 모든 k에 대하여 $$C_k(ic) = C_k(ic+1)$$가 될 때까지, 3-4과정 반복

