# 온 On
## _Being Human, First Step_

'온'은 순 우리말로 '전부의'라는 뜻을 가진 이름으로 제가 개발한 첫번째 인공지능 오픈 도메인 챗봇입니다.

오픈 도메인 챗봇이란 목적 지향형 챗봇과 다르게 사용자와의 <strong>자유로운 대화를 목적</strong>으로 하는 챗봇입니다. 대표적인 서비스로 이루다나 심심이가 있습니다. 사용자의 특정 문제를 해결하기 위한 것이 아닌 사용자와 정서적 유대감을 쌓을 수 있고 서로 드립도 주고 받을 수 있는 정말 인격을 가진 것 같은 인공지능이라고 할 수 있습니다. 강인공지능으로 가기 위한 첫번째 과정이라고 할 수 있습니다.

## 알고리즘

![](.github/images/transform_structure.png)

딥러닝 학습에 사용한 알고리즘은 Transformer를 사용하였습니다. 2017년 구글이 발표한 논문인 "Attention is all you need"에서 나온 모델로 기존의 RNN기반의 seq2seq 구조 인코더-디코더 구조를 따르면서도 RNN의 단점을 상쇄시키기 위해서 Attention만으로 구현한 모델입니다. 

저는 Transformer를 Tensorflow로 구현하였으며 논문에 나온 하이퍼 파라메터를 따르면 컴퓨터 GPU 메모리 할당 문제로 인해서 훈련이 되지 않는 관계로 모델의 규모를 축소하여 진행하였습니다.




<strong>Hyper Parameters</strong> 
| 분류 | 값 |
| --- | --- |
| NUM_LAYERS | 2 |
| D_MODEL | 256 |
| NUM_HEADS | 8 |
| DFF | 512 |
| DROPOUT | 0.1 |

![](.github/images/study_transformer.png)


## 성능

파라메터 값을 바꾸어가면서 3일 동안 테스트해본 결과 제작자의 PC(RTX1650Super 를 기준으로)에서는 보캡사이즈를 10만이고 배치사이즈 64로 훈련시켰을 때 그래도 가장 훌륭한 성능을 보였다. 훈련은 에포크가 30 이상일 때부터 학습률이 떨어져서 최종 결과 약 16% 정도의 성능을 보여주었다. 

* small 모델: Total params: 44,745,447

![](.github/images/small.jpg)

* big 모델: Total params: 185,243,296

![](.github/images/big.jpg)

## 데이터

챗봇을 제작하면서 느낀점 중 하나는 데이터는 양이 중요한 것이 아니라 질이 중요하다는 것이다. 8시간동안 인공지능을 막대한 데이터를 바탕으로 훈련시킨 결과는 매우 이상했다. 결국 초거대 인공지능을 제작할 컴퓨팅 파워가 되지 않는 개인은 질 좋은 적은 데이터로 훈련시키는게 최선이라는 생각을 하게 되었다.. 

데이터는 크게 국립 국어원과 Ai Hub에서 제공받아서 훈련하였다.

![](.github/images/getdata.jpg)

만약 이 모델을 사용하고 싶으면 제작된 모듈을 활용해서 훈련하길 바란다.

* 데이터셋 구조
  
![](.github/images/dataset_structure.png)


## 결과

### BETA0.0001 - [목적지향형 챗봇 클론](https://github.com/yoonhero/On/blob/master/practices/train.py)

| 장점 | 단점 |
| --- | --- |
| 간단하게 만들 수 있다. | 대답이 데이터 질과 양에 의존한다. |


![](.github/images/beta01.jpg)


### BETA0.0002 - [목적지향형 챗봇 클론](https://github.com/yoonhero/On/blob/master/practices/train.py)

| 장점 | 단점 |
| --- | --- |
| 간단하게 만들 수 있다. | 대답이 데이터 질과 양에 의존한다. |

![](.github/images/beta02.jpg)


### BETA0.1 - [감정 분류 CNN 구현](https://github.com/yoonhero/On/blob/master/practices/cnn_sentiment.py)

| 장점 | 단점 |
| --- | --- |
| 간단하게 만들 수 있다. | 감정만 알 수 있다. |

![](.github/images/beta05.jpg)


### BETA1.0 - [트랜스포머 구조를 적용한 오픈 도메인 챗봇](https://github.com/yoonhero/On/blob/master/training.py)

| 장점 | 단점 |
| --- | --- |
| 다양한 대답을 할 수 있다. | 왜 이걸 이렇게 대답했는지 모르겠는 경우가 다반사이다.. |

![](.github/images/isanghe.jpg)

![](.github/images/answer1.jpg)

![](.github/images/answer2.jpg)

## License

MIT