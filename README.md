
## Getting Started

### Installation

> [!IMPORTANT]
> Installation is mandatory.

#### Install from Source

```bash
git clone --depth 1 https://github.com/MLAI-Yonsei/Distill_MoE.git
cd Distill_MoE
pip install -e ".[torch,metrics]" --no-build-isolation
```

## Distill MoE
utills_distill_moe 폴더에 LLaMA Factory에서 사용할 수 있는 코드를 저장해두었음.
`split_moe_studentMoE.py`: 일반 TinyLLaMA_v.1모델을 활용하여 MoE construction을 수행하는 코드임 (LLaMA MoE와 같은 방식으로 스플릿 + Student, Teacher 분할만 다름)
`/distill_moe_tr`: transformer 패키지에 저장해야할 모델 코드 (`/data1/choins18/anaconda3/envs/distill_moe/lib/python3.10/site-packages/transformers/models/` 폴더 안에 해당 모델 코드 폴더를 복붙하고, models 폴더 안에 `__init__.py`에 모델 클래스 import하는 코드를 삽입하면 완료)
`loader.py`: `/Distill_MoE/src/llamafactory/model/adapter.py`에 있는 loader 파일에 custom model인 distill moe 모델을 인식할 수 있도록 import한 코드임.
transformer 패키지에 모델 코드를 추가해도 해당 import를 해야 model class를 인식하고 해당 모델을 사용할 수 있음
`config.json`: model의 config 파일, 가끔 model_type이 distill_llama 로 되어 있지 않아서 일반 tinyllama를 사용하게 되는 경우가 있음, 해당 파일을 보고 config가 제대로 구성되어 있는지 확인 필수