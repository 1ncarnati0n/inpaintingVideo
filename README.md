# Inpainting in Video

<b>Inpainterz</b>팀은 2023년 이어드림스쿨3기에서 진행된 스타트업기업 연계 프로젝트입니다. <br>
<b>연계기업</b>(커넥트브릭)으로부터 <b>비디오 인페인팅기술</b>에 대한 <b>선행연구개발</b> 주제를 받아 진행하였습니다.

🚀 Team Members [강도성](https://github.com/kang952175), [경소현](https://github.com/SohyeonGyeong), [변웅진](https://github.com/1ncarnati0n), [손수진](https://github.com/Soosembly), [지경호](https://github.com/zkhshub)

🎷 Supported by [(주)**커넥트브릭**](https://connectbrick.com/) 

<img src="assets/connectbrick.png" width="320">

<br>

<p align='center'><i> 그 결과를 오픈소스 프로젝트로 정리했습니다.</i></p>

<br>

## 01. Project Background

### 📽 영상 인페인팅 기술

<p align='center'><img src="assets/A01.png" width="620"></p>

본 프로젝트의 주요 키워드는 **inpainting** 입니다.
지우고자 하는 대상을 마스킹을 하고 마스킹한 부분을 이미지상에서 누락된 부분으로 인식, 이를 자연스럽게 채우는 것을 **'인페인팅'** 이라고 합니다.

특히 단순히 이미지가 아닌 **영상 인페인팅**은 영상에 있는 시공간상의 구멍(spatio-temporal holes)을 메우고 비디오의 누락된 부분을 채우는 것을 목표로 합니다. 이미지에 비교하여 영상물 특성상 콘텐츠의 정확한 공간적, 시간적 일관성을 유지하기 어렵기 때문에 여전히 어려운 과제입니다.

<br>
<p align='center'><img src="assets/A02.gif" width="520"></p>
2022년에 개봉한 영화, 'Everything everywhere all at once'의 시각효과는 기존 시각효과산업의 방식이아닌 새로운 AI기술을 적용한 방법으로 구현되었습니다. 영상인페인팅기술 또한 위 장면에서 확인 할 수 있으며 시각효과 방식에 대한 패러다임이 AI기술에 의해 전환되고 있습니다.

<br>

### 🎯 프로젝트 목표 및 비즈니스 배경 

목표는 영상에서 **특정로고**를 **인페인팅**하여 로고가 보이지 않도록 삭제하는 모델 혹은 어플리케이션을 구축하는 것입니다. 

이는 방송에서 노출되는 로고나 ppl제품을 대상으로 하는 인페인팅을 할 수 있습니다. 방송을 시청하다 보면 가끔 상표가 가려진 모습을 볼 때가 있습니다. 이는 주로 두 가지 이유로 이루어집니다. 먼저, *지적 재산권과 상표권 보호* 그리고 *광고 규정을 준수하기 위한 방침*에 따른 것입니다. 
<p align='center'><img src="assets/A03.jpeg" width="320"></p>

특정 브랜드의 제품이나 로고가 무단으로 노출될 경우, 해당 브랜드의 지적 재산권을 침해할 수가 있고 무단 사용된 상표는 이미지의 손상을 줄 수도 있어 위 이미지와 같이 상표가리기 및 편집을 합니다.

방송에서 상업광고를 제한하기 위한 규정이 있으므로 프로그램 내 제품과 브랜드 노출을 피하고 광고와 프로그램을 분리하여 방영해야 하는 경우가 있어 상표를 가리거나 편집을 합니다. 또한, 나라별로 이러한 규정이 다르기 때문에 편집작업이 추가됩니다.

<br>

## 02. 방법론 및 기술탐색

### 🪜 영상 인페인팅 단계별 과정

🎥 영상 인페인팅을 수행하기 위해 세 단계로 과정을 분리했습니다.

1. **영상 객체 마스킹**: Segmentation & Masking <br>
영상내 한 프레임에서 인페인팅할 객체를 선택, 정확하게 분리하기 위해 Segmentation 기법을 사용. 분할된 객체를 Masking하여, 인페인팅 알고리즘이 수행할 수 있게 한다.

2. **영상 마스킹 추적**: Tracking, use Long-term Memory <br>
Long-term Memory으로 마스킹된 객체가 특정 프레임 내에서 따라 움직이는 것을 연속적으로 추적하고 추가 마스킹을 수행하여 영상 내 선택 객체의 모든 마스킹 이미지를 생성한다.

3. **영상 인페인팅**: Inpainting <br>
Input 값으로 마스킹된 모든 영상프레임을 넣으면 마스킹된 영역을 인식, 이 과정에서 알고리즘은 주변의 픽셀정보로 마스킹 부분의 색상과 텍스처 등을 추정하고 채운다.

<br>

### 🧑🏻‍💻 기술탐색

위 3가지 단계에 맞는 모델들을 탐색했고 **inpainterz 파이프라인**을 기획했습니다. <br>

⭐️ **주요 알고리즘**으로는 제로샷러닝 및 비젼에서의 파운데이션 모델로 선보인 Meta의 [**SAM**(Segment Anything Models)](https://github.com/facebookresearch/segment-anything)과 효율적인 Multi-Object Track 그리고 Propagation를 위한 [**DeAOT**(Decoupling features in Associating Objects with Transformers)](https://github.com/yoxu515/aot-benchmark) 그리고 [**E2FGVI** (End-to-End Framework for Flow-Guided Video Inpainting)](https://github.com/MCG-NKU/E2FGVI)등을 선별하여 적용했습니다.

<p align="center"> <img src="assets/readme00.png" width="1080"> </p>

이를 통합한 **inpainterz 파이프라인**🔧에서 

**SAM**은 새로운 오브젝트를 동적으로 자동감지하고 세분화할 수 있도록 지원하며, **DeAOT**는 식별된 모든 오브젝트를 추적하는 역할을 담당합니다. 결과적으로 **E2FGVI**는 영상내 모든 마스킹된 대상을 인페인팅합니다. 

그리고 최종으로 gradio 라이브러리를 이용해 GUI를 구성했습니다.

<br>

## Summary of Used Algorithms
inpainterz에서 사용한 알고리즘에 대한 내용을 요약했습니다.

### SAM 
<details>
<summary> Segment Anything Model </summary> 

**[Paper](https://ai.meta.com/research/publications/segment-anything/)**

대규모 데이터셋이 구축되지 않았던 기존의 Segmentation 작업은 매번 학습에 소모되는 시간과 비용이 너무 크다는 문제가 있었습니다. NLP 분야의 LLM처럼, **Zero-shot**이 가능한 모델을 만들수 없을까 했고, 2023년 4월 Meta에서는 Image Segmentation계의 **Foundation** 모델을 만드는 것을 목표로 이 모델을 발표했습니다.

Meta는 다음 세 가지를 새롭게 선보였습니다. **Task**, **Model**, **Data**.
1. **Task** ( Promptable Segmentation Task )\
	Segment Anything Task의 핵심은 **프롬프팅이 가능**하다는 것.\
	원하는 영역의 **Point**나 **Box** 또는 **자연어**, (+ **Mask**)로 구성된 프롬프트를 입력하면, 아무리 모호한 정보일지라도 유효한 Segmentation Mask를 출력한다.
	<p align="center">
	<img src="assets/readme01.png" width="420">
	</p>
 
2. **Model** ( Segment Anything Model, SAM )\
	이를 위한 모델인 SAM은 **두 개의 인코더**와 **하나의 디코더**로 구성.
	Image Encoder와 Prompt Encoder로부터 온 임베딩 정보를 매핑해 Mask Decoder가 예측된 Segmentation Mask를 출력하는 구조다.\
	Mask Decoder는 Transformer의 Decoder를 조금 수정한 것으로, 이미지 임베딩과 프롬프트 임베딩을 모두 업데이트 하기 위해 **Self-Attention**과 **Cross-Attention**을 양방향으로 활용한다.\
	SAM의 Prompt Encoder와 Mask Decoder는 **가볍고 빠르다**.\
	같은 이미지 임베딩이 여러 개의 프롬프트와 함께 재사용되기 때문에, CPU 환경의 웹 상에서 50ms 이하의 속도로 Mask를 예측할 수 있다.
	<p align="center">	
	<img src="assets/readme02.gif" width="360">
	</p>

3. **Data** ( Segment Anythin Data Engine, SA-1B Dataset )\
	Foundation 모델 개발에 있어 가장 중요한 것은 대규모 데이터셋이다.\
	Segment Anything은 자체적인 **Data Engine**을 개발했고, 그 결과 10억 개의 Mask를 가진 **SA-1B** 데이터셋이 탄생했다.

	<p align="center">
	<img src="assets/readme03.png" width="420">
	</p>
</details>

### DeAOT

<details>
<summary> Decoupling features in Associating Objects with Transformers </summary> 

[**Paper**](https://arxiv.org/abs/2210.09782)

비디오 내의 객체들을 세밀하게 구분하는 'semi-supervised 비디오 객체 세분화(VOS)'에 관한 모델입니다.\
특히,'비전 트랜스포머'라는 최신 기술을 사용, 'AOT(Associating Objects with Transformers)'라는 방법을 통해 VOS 문제를 해결하는 데 집중하고 있습니다. 이전 프레임에서 현재 프레임으로 정보를 차례대로 전달하는 '계층적 전파 hierarchical propagation' 방식을 사용하며, 이 방식은 각 객체의 정보를 점진적으로 전달하지만, 깊은 층에서는 일부 시각적 정보가 손실될 수 있는 단점이 있습니다. 이 문제를 해결하기 위해, 연구자들은 'DeAOT'라는 새로운 접근 방식을 제안합니다. DeAOT는 객체별 정보와 무관한 정보를 분리하여 처리함으로써 보다 효율적인 정보 전달을 가능하게 합니다. 또한, 이 방법은 추가적인 계산 부담을 줄이기 위해 특별히 설계된 '게이트 전파 모듈 Gated Propagation Module(GPM)'을 사용합니다. 결과적으로, DeAOT는 기존 AOT 및 다른 방식의 모델인 XMem보다 뛰어난 정확도 및 효율성을 보여줍니다.

<p align="center">	
<img src="assets/readme05.png" width="480">
</p>

<p align="center">	
<img src="assets/readme05.gif" width="480">
</p>

1. **VOS의 정의와 배경**\
	VOS는 주어진 비디오에서 하나 또는 여러 객체를 인식하고 분할하는 중요한 비디오 이해 작업입니다. 이 연구는 알고리즘이 초기 프레임에서 주어진 객체의 마스크를 기반으로 전체 비디오 시퀀스에 걸쳐 객체를 추적하고 분할해야 하는 semi-supervised VOS에 중점을 둡니다​​.

2. **DeAOT의 주요 구성**\
	DeAOT는 두 가지 분기, 즉 시각적 분기와 ID 분기로 구성. 시각적 분기는 객체를 일치시키고 과거의 시각 정보를 수집하며 객체 특징을 정제하는 역할. ID 분기는 시각적 분기에서 계산된 일치 맵(attention map)을 재사용하여 과거 프레임에서 현재 프레임으로 ID 임베딩을 전파​​.

3. **Gated Propagation Module (GPM)**\
	DeAOT에서는 효율성을 높이기 위해 Single Head attention을 기반으로 설계된 GPM을 사용합니다. GPM은 자체전파, 장기전파, 단기전파의 세 가지 종류의 게이트 전파를 포함.

4. **네트워크 세부사항, 트레이닝**\
   DeAOT는 다양한 인코더와 동일한 FPN 디코더를 사용합니다. GPM 모듈은 시각적 및 ID 임베딩의 차원을 지정하고, 학습은 정적 이미지 데이터셋에서 생성된 합성 비디오 시퀀스와 VOS 벤치마크에서 수행됩니다​​.

5. **결론**\
   DeAOT는 계층적 VOS 전파를 위한 효율적인 프레임워크를 제공. 이는 계층적 전파에서 시각적 및 ID 임베딩을 분리하여 깊은 전파계층에서의 시각 정보 손실을 방지.

</details>

### E2FGVI
<details>
<summary> End-to-End Framework for Flow-Guided Video Inpainting </summary> 

[**Paper**](https://arxiv.org/abs/2204.02663)

**비디오 인페인팅**
목표는 비디오 클립 전체에서 ‘손상된’ 영역을 그럴듯하고 일관된 콘텐츠로 채우는 것, 하지만 남은 과제로 복잡한 비디오 시나리오와 저하된 비디오 프레임에 관한 문제가 있으며. 이는 고품질 비디오 인페인팅을 위해서는 **공간적 구조**와 **시간적 일관성**을 모두 고려해야 함을 의합니다.

1. **기존방법: Flow-based methods**
- 이런 일반적인 흐름기반 방법(flow-based method)는 인페인팅을 **pixel propagation** 문제로 생각하여 시간적 일관성을 자연스럽게 보존

	1. flow completion : 
	   손상된 영역에 flow field가 없으면 후자의 프로세스에 영향을 미치므로 먼저 추정된 optical flow가 먼저 완료 되어야 한다
	2. pixel propagation : 
	   앞서 완성된 optical flow의 가이드에 따라 가시영역의 픽셀을 양방향으로 전파, 손상된 비디오 영역을 채움 
	3. content hallucination : 
	   픽셀 전파 후, 나머지 누락된 영역은 사전 학습된 이미지 인페인팅 네트워크로 환각으로 채움
	
	-  인페인팅의 방법은 전체 인페인팅 파이프라인을 구성하기 위해 개별적으로 적용, 인상적인 결과를 얻을 수 있지만, 처음 두 단계에서는 많은 수작업이 필요해서, **각 프로세스는 별도로 수행**해야 하는 **단점**이 있다.
	
	- 따라서, 두 가지 주요한 문제를 야기한다.
	    a. **이전 단계에서 발생한 오류가** 누적 후속 단계에서 증폭, **최종 성능에 큰 영향을 미침**
	    b. **복잡한 연산**을 해야하지만, GPU acceleration 처리불가, **많은 시간이 소요**

	<p align="center">	
	<img src="assets/readme04.png" width="480">
	</p>

2. **개선모델: E2FGVI** (Fig.Ours) 
- 이전 모델을 보완, 이전 방법과 다르게 **End-to-End**로 최적화 할 수 있어 보다 효율, 효과적인 인페인팅 프로세스 구현.
	
	1. **Flow-Completion** 모듈: 	
	   복잡한 단계 대신 원-스텝 완성을 위해 마스킹 된 비디오에 직접 적용
	2. **Feature Propagation** 모듈: 
	   pixel-level propagation과 달리, flow-guided propagation 프로세스는 (변형이 가능한 convolution의 도움을 받아서) feature space 수행됨. 
	   → 학습 가능한 sampling offset과 feature-level 연산 통해 **정확하지 않은 flow추정 부담을 덜어줌**
	3. **Content Hallucination** 모듈: 
	   공간과 시간적 차원에서 장거리 종속성을 효과적으로 모델링하기 위해 temporal focal transformer를 제안.
	   →이 모듈에서 local and non-local temporal neighbors을 모두 고려, **보다 시간적으로 일관된 인페인팅 결과** 도출

- 70개의 프레임 기준으로 이 크기의 비디오 하나를 완성하는 데에 약 4분 소요. E2FGVI는 프레임당 0.12초로 약 8.4초 소요.
 	<p align="center">	
	<img src="assets/readme06.png" width="480">
	</p>

</details>

<br>

## 04. GUI 구성

<br>

## 05. 결과

<br>

## 06. Review

### 구성한 App의 한계점

- 빠르게 움직이는 대상과 대상에 간섭이 지속적으로 이루어지는 경우 memory를 놓친다. (e.g. 댄스 영상)

- 경계가 뚜렷하지 않은 객체(벽의 균열)등을 inpainting하고자 하는 경우 잘 동작되지 않다.

### 회고 및 개선 가능한 방향들

- 이미지의 첫 단에 등장하는 객체가 아닌 중간이나 끝에 삽입되는 객체를 기억하는 알고리즘이다.

- SOTA inpainting 알고리즘을 적용하여 더 자연스러운 객체 제거를 할 수 있다.

<br>

## Demo & Tutorial 

- Colab Demo: 👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cQLFl2z5iOI9iZDcn4hSZ4zMU7bUu9NX)

- Tutorial: 👉 [Link]()

- Project PPT: 👉 [![Static Badge](https://img.shields.io/badge/report_ppt-pdf)](https://drive.google.com/file/d/1QtrXoP2Ny8CYVx314VeXlj5KS7QdsAx9/view?usp=drive_link)

<br>

## 🎮 Getting Started

<details>
<summary> <i> click and open "install guide" </i>  </summary> 

<br>

**1. Conda Default Environment** 🎾

```shell
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
pip install gradio==3.39
pip install av
pip install gdown
```

**2. Requirements** 🗝️
- The Segment-Anything repository has been cloned and renamed as sam, <br> and the aot benchmark repository has been cloned and renamed as aot.

- Please check the dependency requirements in SAM and DeAOT and E2FGVI.

- The *implementation is tested under **python 3.9***, as well as ***pytorch 1.11.0*** + ***cu113*** <br> and torchvision 0.12.0 + cu113 We recommend equivalent or higher pytorch version.
  
-  Use the install.sh to install the necessary libs for **Inpainterz**
	```
	bash script/install.sh
	```

**3. Model Preparation** ⚙️
- Download **SAM** model to **ckpt**, the default model is SAM-VIT-B (sam_vit_b_01ec64.pth).
  
- Download **DeAOT/AOT** model to **ckpt**, 
  the default model is R50-DeAOT-L (R50_DeAOTL_PRE_YTB_DAV.pth).

- Download **Grounding-Dino** model to **ckpt**, 
  the default model is GroundingDINO-T (groundingdino_swint_ogc).

- Download **E2fgvi** model to **ckpt**, 
  the default model is E2FGVI-CVPR22 (E2FGVI-CVPR22.pth)

- You can download the default weights using the command line as shown below.

	```
	bash script/download_ckpt.sh
	```
**Overall packages**

```
\inpainterz
	⎣ \E2FGVI
	⎣ \Pytorch-Correlation-extension
	⎣ \aot
	⎣ \assets
	⎣ \ckpt
	⎣ \groundingdino
		⎣ \config
		⎣ \datasets
		⎣ \models
		⎣ \util	
	⎣ \sam
	⎣ \script
	⎣ \sam
	⎣ \script
	⎣ \src
	⎣ \tool
	⎣ \tutorial
```

</details> 

<br>

## License❗️

오픈소스를 지향합니다. SAM, DeAOT는 상업적 이용까지 가능한 오픈소스입니다.

하지만 E2FGVI는 상업적으로는 이용할 수 없기에 추가 확인을 하시기 바랍니다.

<br>

## Acknowledgement🧐
This repository is maintained by **Inpainterz** [강도성](https://github.com/kang952175) and  [경소현](https://github.com/SohyeonGyeong) ,[변웅진](https://github.com/1ncarnati0n), [손수진](https://github.com/Soosembly),  [지경호](https://github.com/zkhshub)

This code is based on [SAM](https://github.com/facebookresearch/segment-anything), [DeAOT](https://github.com/z-x-yang/AOT), [SAMTrack](https://github.com/z-x-yang/Segment-and-Track-Anything), and [E2FGVI](https://github.com/MCG-NKU/E2FGVI).

Inspired by [Facebookresearch](https://github.com/facebookresearch/segment-anything), [z-x-yang](https://github.com/z-x-yang/Segment-and-Track-Anything), and [MCG-NKU](https://github.com/MCG-NKU/E2FGVI).
