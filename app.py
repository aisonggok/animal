import pickle
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms

st.set_page_config(page_title="AI 동물 분류기", page_icon="🐾")
st.title("🐾 카메라 AI 동물 분류기")


# 1. 가벼운 PyTorch SqueezeNet 임베더 및 모델 로드
@st.cache_resource
def load_resources():
    # 오렌지3 기본 임베더와 동일한 SqueezeNet 1.1 로드
    weights = models.SqueezeNet1_1_Weights.DEFAULT
    embedder = models.squeezenet1_1(weights=weights)
    embedder.eval()  # 추론 모드 설정

    # 이미지 전처리 파이프라인
    preprocess = weights.transforms()

    # 저장된 모델 파일 로드 (파일명을 본인의 깃허브 파일명과 맞추세요)
    with open("animal.pkcls", "rb") as f:
        # Orange3 모듈 없이 scikit-learn 모델만 안전하게 추출하기 위한 Unpickler
        class SafeUnpickler(pickle.Unpickler):

            def find_class(self, module, name):
                if module.startswith("Orange"):
                    return type(name, (object,), {"__module__": module})
                return super().find_class(module, name)

        orange_obj = SafeUnpickler(f).load()

    return embedder, preprocess, orange_obj


try:
    embedder, preprocess, model_obj = load_resources()
except Exception as e:
    st.error(f"모델 로드 오류: {e}")
    st.stop()

# 2. 카메라 입력
img_file = st.camera_input("동물을 촬영해 주세요")

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="촬영된 이미지", use_container_width=True)

    with st.spinner("분석 중..."):
        # 이미지 특징 벡터 추출 (1 x 1000)
        input_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = embedder(input_tensor).numpy()

        # 내부 scikit-learn 모델로 예측
        skl_model = model_obj.skl_model
        class_names = model_obj.domain.class_var.values

        pred_idx = skl_model.predict(features)[0]
        probs = skl_model.predict_proba(features)[0]

        st.success(
            f"### 예측 결과: **{class_names[int(pred_idx)]}** ({probs[int(pred_idx)]*100:.1f}%)"
        )

        # 확률 차트
        prob_dict = {name: float(p) for name, p in zip(class_names, probs)}
        st.bar_chart(prob_dict)
