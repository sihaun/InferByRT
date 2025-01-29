import torch
import models.efficientnet as eff


class EffiToOnnx:
    def __init__(self):
        self.model = eff.tf_efficientnet_b7(5)
        self.model.load_state_dict(torch.load("only_weight.pt", weights_only=True))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

    def to_onnx(self, onnx_path):
        torch.onnx.export(
            self.model,
            self.dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }
        )
# 모델 로드 (EfficientNet B7 모델 사용)
model = eff.tf_efficientnet_b7(5)
model.load_state_dict(torch.load("only_weight.pt", weights_only=True))
model.eval()

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델을 GPU로 이동
model.to(device)

# 더미 입력 (배치 크기: 1, 채널: 3, 이미지 크기: 224x224)
dummy_input = torch.randn(1, 3, 224, 224).to(device)  # GPU로 이동

ONNX_PATH = "trt.onnx"
# ONNX 변환
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,       # 저장할 ONNX 파일 이름
    export_params=True,             # 모델 파라미터 저장
    opset_version=11,               # ONNX Opset 버전
    do_constant_folding=True,       # 상수 폴딩 최적화
    input_names=["input"],          # 입력 이름 정의
    output_names=["output"],        # 출력 이름 정의
    dynamic_axes={                  # 동적 입력 크기 허용
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }
)

if __name__ == "__main__":
    effi_to_onnx = EffiToOnnx()
    effi_to_onnx.to_onnx("trt.onnx")