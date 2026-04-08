# Encrypted 3D-Face-Recognition

암호화된 3D Face-Recognition

------
## A. Introduction
1. **프로젝트 배경**
3D 데이터(여기서는 point cloud 형태의 데이터)로 구성되어있는 얼굴 인식 시스템을 구현하고 암호화한 아키텍처를 제시함    
이게 얼마나 성능이 나올까?에 대해 알아보고 클라이언트-서버 간 구조를 제시   
*중앙대학교 2025학년도 동계 SW·AI학부연구생 프로그램 대상   
    
2. **참조한 논문**
_**PointFace: Point Cloud Encoder-Based Feature Embedding for 3D Face Recognition**_      
_Changyuan Jiang , Shisong Lin , Wei Chen , Feng Liu , Member, IEEE, and Linlin Shen , Senior Member, IEEE_      
_IEEE TRANSACTIONS ON BIOMETRICS, BEHAVIOR, AND IDENTITY SCIENCE, VOL. 4, NO. 4, OCTOBER 2022_      
해당 논문에서 제시하는 PointFace 아키텍처를 구현하고 이를 테스트함


## B. Preliminary

### - Cancelable Biometrics
생체 인식 데이터를 암호화 기술을 사용하여 의도적으로 변형하고, 이 변형된 템플릿을 저장하여 사용하는 기술      
원본 생체 정보를 그대로 저장하지 않으므로, 정보가 유출되더라도 해당 변형 함수를 변경하여 새로운 템플릿을 재발급할 수 있음


### - Homomorphic Encryption
동형 암호(Homomorphic Encryption)이란, 암호문(Ciphertext)인 상태로 연산이 가능한 암호를 말함
수식으로 표현하면 다음과 같음

$$
DEC(ENC(x) \ \Delta \ ENC(y)) = x \ \Delta \ y
$$

Encryption(암호화)시에 r(noise, error)를 넣음
→ 연산 시마다 error가 커지게 됨

그래서 연산 깊이(depth)를 정해놓는 경우가 있음
=> 이 횟수를 넘어가면 오차가 커져서 원래대로 복호화하지 못함

### - BioHashing
생체 정보와 토큰을 결합하여 보안성이 높고 Cancelable한 생체 템플릿을 생성하는 기술

1. **특징 추출**
얼굴, 지문 등에서 고정된 길이의 특징 벡터 $v$를 추출함

2. **토큰 생성**
토큰, 비밀번호로부터 시드(seed)를 얻음      
해당 시드를 이용하여 Pseudo Random Number를 생성함      

3. **직교 투영**
Gaussian Distribution을 이용하여 랜덤 Matrix를 생성함
Gram-Schmidt를 이용하여 Orthogonal Matrix $R$를 생성함      
해당 Matrix를 생체 템플릿에 내적함
$$
x = R \cdot v
$$

4. **이진화**
내적한 결과값이 0보다 크면 1, 작으면 0으로 변환함
$$
b_i = \begin{cases} 1 \ \ if \ \ x_i > \tau \\ 0 \ \ otherwise\end{cases}
$$

* 특이사항
FAR, FRR이 감소하는 효과가 있음
1. FAR(False Accept Rate)의 감소 이유
개인의 특징 + 개인의 비밀 키(시드)라는 두 개의 요소가 필요하므로      
비슷한 특징의 인물을 더이상 착각하지 않게 됨      

2. FRR(False Reject Rate)의 감소 이유
이진화를 거치게 되므로 인식 시의 노이즈가 흡수됨      
등록된 템플릿과 약간 다르더라도 임계값에 따라 0, 1로 분류되므로 FRR이 감소하게 됨     

## C. Overall Architecture
<img src="/docs/Overall_Architecture.jpg" width="100%" height="100%" title="Overall_Architecture" alt="Overall_Architecture"></img>
_**Figure 1.** Overall Architecture_        

Feature Extractor의 Relation Shape Convolution은 두 개의 모듈로 구성되어 있음(Set Abstraction, RSConv)
<img src="/docs/RSConv.jpg" width="100%" height="100%" title="RSConv" alt="RSConv"></img>
_**Figure 2.** Relation Shape Convolution_        

## D. Experiment

### - Dataset
**UMB-DB**    
143명 (남 98, 여 45; 쌍둥이 남자, 아기 포함)     
1473 total acquisitions (3D + colour 2D)      
120명의 데이터를 학습에 사용, 나머지 23명의 데이터를 추론에 사용함      

### - Encoder Training
120명의 데이터, 1251 쌍의 데이터 셋을 가지고 학습함      
PointFace에서 언급한 대로, Siamese Network를 이용함    
<img src="/docs/Training.jpg" width="100%" height="100%" title="Training" alt="Training"></img>
_**Figure 3.** Training Architecture_        

Loss Function은 다음과 같음 (Contrastive Loss, PointFace 참조)
$$L_{Total} = L_{softmax} + \lambda L_{sim}$$      

$L_{softmax} = loss(x, class) = -\log(\frac{ \exp(x_ {class}) } { \sum_{j} \exp (x_{j}) }) = -S_t + \log \sum_1^j \exp(x_i)$      
$L_{sim} = \sum_{i=1}^{N_s}[1 - \mathcal D(emb_i, emb_i^+) + \mathcal D(emb_i, emb_i^-) - m]$     
$\lambda = 1, margin = 0.35$      

```python
# 람다(lambda) 값 (두 loss 간의 비율)
lambda_factor = 1.0 

# 손실 함수 정의
criterion_softmax = nn.CrossEntropyLoss()
criterion_similarity = FeatureSimilarityLoss(margin=0.35).to(device)

# 옵티마이저 (Adam, lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)
```


**Unseen Test**
모델이 잘 학습되었는지를 확인하기 위해서 Unseen Test를 수행      
학습에 사용하지 않은 23명의 데이터를 이용해서 얼굴 쌍에 대해 유사도를 계산함
<img src="/docs/unseen_test_result_200.png" width="100%" height="100%" title="epoch_200_unseen_test_result" alt="epoch_200_unseen_test_result"></img>
_**Figure 4.** Unseen Test (200 Epoch)_        

부정 쌍에 대해서는 유사도가 낮고 긍정 쌍에 대해서는 유사도가 높음      
어느 정도 인물의 특징을 잘 구분할 수는 있으나, 긍정 쌍(동일 인물)을 구분해내는 능력이 부족하다고 판단됨      

**Rank-1 Accuracy**
학습에 사용하지 않은 23명의 데이터를 이용해서 Rank-1 Accuracy를 계산
Gallery: 23, Probes: 199
Rank-1 Accuracy: 84.92% (169/199)


### - Matching Result

**전체 Matching Dataset**     
23 Person, Total 222 Data      

동일 인물의 여러 얼굴 데이터를 넣고 템플릿을 추출한 후 평균을 구해서 DB에 저장함      

- Model Weight = 200 Epoch checkpoint
- Cosine Similarity Threshold = 0.7
- Hamming Distance Threshold = 0.3      

암호화는 1) **CKKS 스킴의 동형 암호(Homomorphic Encryption)**, 2) **BioHashing** 두 가지를 사용함      

#### Metric
보안 성능을 측정하기 위해 FAR(False Accept Rate), FRR(False Reject Rate), EER(Equal Error Rate)를 측정함

<img src="/docs/Cosine_similarity_result.png" width="100%" height="100%" title="Cosine_similarity_result" alt="Cosine_similarity_result"></img>
_**Figure 5.** FAR vs FRR Curve (Cosine Similarity)_        

<img src="/docs/Hamming_distance_result.png" width="100%" height="100%" title="Hamming_distance_result" alt="Hamming_distance_result"></img>
_**Figure 6.** FAR vs FRR Curve (Hamming Distance)_        


FAR을 낮게 유지하기 위해서 EER에서 옆으로 벗어난 값을 Threshold로 사용함      
-> FRR이 증가함에도 불구하고 FAR을 줄이기 위함

#### Biohashing Threshold
<img src="/docs/Hamming_Distance_Distribution.png" width="100%" height="100%" title="Hamming_Distance_Distribution" alt="Hamming_Distance_Distribution"></img>
_**Figure 7.** Hamming Distance Distribution_   
이론적으로 Threshold의 안전 마지노선은 0.35임
여기서는 Threshold로 0.3을 사용함


#### Matching Result

**[11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40 GHz (8 CPUs), 16GB Ram]**
* **Matching (without ENC)**
템플릿 암호화 없이 1:1 매칭시킨 결과 (Average)

    |  Preprocessing  |   Extracting Embedding   |  Matching  |  Total  |
    |:----------:|:-----------:|:-----------:|:-----------:|
    |  0.0047 sec  |   1.2199 sec   |   0.0003 sec   |  1.2254 sec   |

    정확도 - Total : 222, Correct : 190 (85.59%), Wrong : 32 (14.41%)  


* **Matching (with HE)**
템플릿 암호화 후 1:1 매칭시킨 결과 (Average)

    |  Preprocessing  |   Extracting Embedding   |  Encrypting  |  Matching  |  Total  |
    |:----------:|:-----------:|:-----------:|:-----------:|:-----------:|
    |  0.0037 sec  |   1.2779 sec   |   0.0065 sec   |  0.0228 sec   |  1.3115 sec   |

    정확도 - Total : 222, Correct : 190 (85.59%), Wrong : 32 (14.41%)        


* **Matching (with BioHashing)**
템플릿 암호화 후 1:1 매칭시킨 결과 (Average)

    |  Preprocessing  |   Extracting Embedding   |  Encrypting  |  Matching  |  Total  |
    |:----------:|:-----------:|:-----------:|:-----------:|:-----------:|
    |  0.0036 sec  |   1.2184 sec   |   0.0462 sec   |  0.0001 sec   |  1.2687 sec   |

    정확도 - Total : 222, Correct : 204 (92.79%), Wrong : 18 (7.21%)   


**[Ampere Altra Q80-30 (4 OCPU), 24GB Ram]**
* **Matching (without ENC)**
템플릿 암호화 없이 1:1 매칭시킨 결과 (Average)

    |  Preprocessing  |   Extracting Embedding   |  Matching  |  Total  |
    |:----------:|:-----------:|:-----------:|:-----------:|
    |  0.0033 sec  |   0.9967 sec   |   0.0002 sec   |  1.0004 sec   |

    정확도 - Total : 222, Correct : 187 (84.23%), Wrong : 35 (15.77%)      


* **Matching (with HE)**
템플릿 암호화 후 1:1 매칭시킨 결과 (Average)

    |  Preprocessing  |   Extracting Embedding   |  Encrypting  |  Matching  |  Total  |
    |:----------:|:-----------:|:-----------:|:-----------:|:-----------:|
    |  0.0031 sec  |   0.9818 sec   |   0.0084 sec   |  0.0385 sec   |  1.0320 sec   |

    정확도 - Total : 222, Correct : 190 (85.59%), Wrong : 32 (14.41%)       


* **Matching (with BioHashing)**
템플릿 암호화 후 1:1 매칭시킨 결과 (Average)

    |  Preprocessing  |   Extracting Embedding   |  Encrypting  |  Matching  |  Total  |
    |:----------:|:-----------:|:-----------:|:-----------:|:-----------:|
    |  0.0030 sec  |   0.9724 sec   |   0.0239 sec   |  0.0000 sec   |  0.9995 sec   |

    정확도 - Total : 222, Correct : 208 (93.69%), Wrong : 14 (6.31%)   

## E. Client-Server Architecture
현재 제안하는 클라이언트 - 서버 구조는 아래 그림과 같음      
<img src="/docs/client-server1.jpg" width="100%" height="100%" title="client-server1" alt="client-server1"></img>
_**Figure 8.** Client-Server Architecture_   


현재 구조는 클라이언트에서 특징 추출 -> 서버에서 연산 후 Threshold를 넘었는지 0 1로만 제공 -> 클라이언트가 복호화해서 결과 확인하는 구조

1. Client #1
    |  Preprocessing  |   Extracting Embedding   |  Encrypting  |  Total  |
    |:----------:|:-----------:|:-----------:|:-----------:|
    |  0.0034 sec  |   1.1521 sec   |   0.0055 sec   |  1.1610 sec   |

2. Server
    |   Matching   |  Total  |
    |:-----------:|:-----------:|
    |   0.0232 sec   |   0.0232 sec   |

3. Client #1
    |  Decrypting  |  Total  |
    |:----------:|:-----------:|
    |  0.0006 sec  |   0.0006 sec   |

    정확도 - Total : 222, Correct : 186 (83.78%), Wrong : 37 (16.22%)  
    Total Client = 1.1616 sec, Total Server = 0.0232 sec

이 구조는 Client에서의 연산이 많아지는 구조임 (Client에서 Feature Extractor를 전부 수행하므로)     
Client에서의 연산을 최대한 줄일 수 있는가?


Feature Extractor의 맨 마지막 FC Layer를 서버로 옮기는 구조를 생각중    
이렇게 되면 클라이언트에서 Global Feature까지만 추출 후 암호화 -> 서버에서 DB와 비교해서 Dot 계산 후 클라이언트로 Norm과 함께 반환(나눗셈 연산 코스트 크므로) -> 클라이언트에서 복호화 후 코사인 유사도를 계산하는 방식이 될 듯


**(이론상 걸리는 시간)**
1. Client #1
    |  Preprocessing  |   Global Feature Extraction   |  Encrypting  |  Total  |
    |:----------:|:-----------:|:-----------:|:-----------:|
    |  0.0020 sec  |   1.0457 sec   |   0.0065 sec   |  1.0543 sec   |

2. Server
    |  FC Layer  |   Matching   |  Total  |
    |:----------:|:-----------:|:-----------:|
    |  1.3322 sec  |   0.0102 sec   |   1.3424 sec   |

3. Client #1
    |  Decryption  |   Similarity Score   |  Total  |
    |:----------:|:-----------:|:-----------:|
    |  0.0056 sec  |   0.0004 sec   |   0.0060 sec   |

    Total Client = 1.0603 sec, Total Server = 1.3424 sec     

이 구조는 현재 불가능한게 FC 레이어를 옮겨서 그런지 현재 파라미터로 복호화가 안됨      
아마 암호문 크기를 더 키워야 할 듯 한데 그러면 속도가 더 느려질 것임      
추가로 서버에서의 연산 시간도 매우 증가함(암호화된 상태로 FC 레이어를 처리하므로)        
클라이언트의 시간을 조금 줄이는 대신에 리턴이 너무 큰 구조인 듯함
-> 폐기 



## F. Analysis


## G. Conclusion


------
## Reference
1. _C. Jiang, S. Lin, W. Chen, F. Liu and L. Shen, "PointFace: Point Cloud Encoder-Based Feature Embedding for 3-D Face Recognition," in IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 4, no. 4, pp. 486-497, Oct. 2022, doi: 10.1109/TBIOM.2022.3197437._

2. _Colombo, A., Cusano, C., & Schettini, R. (2011). "UMB-DB: A database of partially occluded 3D faces." 2011 IEEE International Conference on Computer Vision Workshops (ICCV Workshops), 2113-2119._

3. _Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J, "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space,", arXiv preprint arXiv:1706.02413, 2017_

4. _Andrew Teoh Beng Jin, David Ngo Chek Ling, Alwyn Goh, "Biohashing: two factor authentication featuring fingerprint data and tokenised random number," Pattern Recognition, Volume 37, Issue 11, 2004, Pages 2245-2255_