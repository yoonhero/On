# 온 On
## _Being Human, First Step_

'온'은 순 우리말로 '전부의'라는 뜻을 가진 이름으로 제가 개발한 첫번째 인공지능 오픈 도메인 챗봇이다.

오픈 도메인 챗봇이란 목적 지향형 챗봇과 다르게 사용자와의 <strong>자유로운 대화를 목적</strong>으로 하는 챗봇이다. 대표적인 서비스로 이루다나 심심이가 있다. 사용자의 특정 문제를 해결하기 위한 것이 아닌 사용자와 정서적 유대감을 쌓을 수 있고 서로 드립도 주고 받을 수 있는 정말 인격을 가진 것 같은 인공지능이라고 할 수 있다. 강인공지능으로 가기 위한 첫번째 과정이라고 할 수 있다.

## Installation


```bash
git clone https://github.com/yoonhero/On
```

깃허브 레포지토리를 클론한다.

```bash
conda env create -f requirements.yml 
```

Anaconda를 사용해서 패키지들을 설치한다.

## Getting Started

크게 데이터 전처리 -> 훈련 -> 모델 사용 단계로 나눌 수 있다.

이 중 훈련과 모델 단계를 모듈화해두었으니 데이터 전처리를 한 다음에 사용하기를 바란다.

데이터 전처리는 질문과 대답으로 이루어진 csv파일을 Q, A  열로 만들면 된다.

### Training

```bash
python training.py \
        --verbose=False \
        --dataset=dataset.csv \
        --tokenizer=tokenizer \
        --create-tokenizer=False \
        --target-vocab-size=2**15 \
        --checkpoint=training/cp-{epoch:04d}.ckpt \
        --batch-size=64 \
        --save-best-only=False \
        --epochs=20 \
```

인자들에 대한 더 자세한 설명이 필요하다면

```bash
python training.py -h
```

를 실행하기를 바란다.


### Use Model

```bash
python main.py \
        --tokenizer=tokenizer \
        --checkpoint=training \
```

인자들에 대한 더 자세한 설명이 필요하다면

```bash
python main.py -h
```

를 실행하기를 바란다.




## 알고리즘

![](https://github.com/yoonhero/On/blob/master/images/transform_structure.png?raw=true)

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

![](https://github.com/yoonhero/On/blob/master/images/study_transformer.jpg?raw=true)


## 성능

파라메터 값을 바꾸어가면서 3일 동안 테스트해본 결과 제작자의 PC(RTX1650Super 를 기준으로)에서는 보캡사이즈를 10만이고 배치사이즈 64로 훈련시켰을 때 그래도 가장 훌륭한 성능을 보였다. 훈련은 에포크가 30 이상일 때부터 학습률이 떨어져서 최종 결과 약 16% 정도의 성능을 보여주었다. 

* small 모델: Total params: 44,745,447

![](https://github.com/yoonhero/On/blob/master/images/small.jpg?raw=true)

* big 모델: Total params: 185,243,296

![](https://github.com/yoonhero/On/blob/master/images/big.jpg?raw=true)

## 데이터

챗봇을 제작하면서 느낀점 중 하나는 데이터는 양이 중요한 것이 아니라 질이 중요하다는 것이다. 8시간동안 인공지능을 막대한 데이터를 바탕으로 훈련시킨 결과는 매우 이상했다. 결국 초거대 인공지능을 제작할 컴퓨팅 파워가 되지 않는 개인은 질 좋은 적은 데이터로 훈련시키는게 최선이라는 생각을 하게 되었다.. 

데이터는 크게 국립 국어원과 Ai Hub에서 제공받아서 훈련하였다.

만약 이 모델을 사용하고 싶으면 제작된 모듈을 활용해서 훈련하길 바란다. 이용 가이드 참고

* 데이터셋 구조
  
![](https://github.com/yoonhero/On/blob/master/images/dataset_structure.png?raw=true)


## 결과

### BETA0.0001 - [목적지향형 챗봇 클론](https://github.com/yoonhero/On/blob/master/practices/train.py)

| 장점 | 단점 |
| --- | --- |
| 간단하게 만들 수 있다. | 대답이 데이터 질과 양에 의존한다. |


### BETA0.0002 - [목적지향형 챗봇 클론](https://github.com/yoonhero/On/blob/master/practices/train.py)

| 장점 | 단점 |
| --- | --- |
| 간단하게 만들 수 있다. | 대답이 데이터 질과 양에 의존한다. |

![](https://github.com/yoonhero/On/blob/master/images/beta02.jpg?raw=true)


### BETA0.1 - [감정 분류 CNN 구현](https://github.com/yoonhero/On/blob/master/practices/cnn_sentiment.py)

| 장점 | 단점 |
| --- | --- |
| 간단하게 만들 수 있다. | 감정만 알 수 있다. |


### BETA1.0 - [트랜스포머 구조를 적용한 오픈 도메인 챗봇](https://github.com/yoonhero/On/blob/master/training.py)

| 장점 | 단점 |
| --- | --- |
| 다양한 대답을 할 수 있다. | 왜 이걸 이렇게 대답했는지 모르겠는 경우가 다반사이다.. |

![](https://github.com/yoonhero/On/blob/master/images/isanghe.jpg?raw=true)

![](https://github.com/yoonhero/On/blob/master/images/answer1.JPG?raw=true)

![](https://github.com/yoonhero/On/blob/master/images/answer2.JPG?raw=true)


## Contribute

자유롭게 Pull Request 보내주시면 코드 검토하고 허락해드리겠습니다.

기타 문의사항이나 모델에 관한 질문은 yoonhero06@naver.com 으로 해드리면 답해드리겠습니다.


## License

MIT