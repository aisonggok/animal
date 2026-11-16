import pickle
import numpy as np
from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
from PIL import Image
import streamlit as st

# 1. 페이지 기본 설정
st.set_page_config(page_title="AI 동물 분류기", page_icon="🐾")
st.title("🐾 카메라 AI 동물 분류기")
st.write("카메라로 동물을 촬영하면 Orange3 신경망 모델이 종류를 분류합니다.")


# 2. 모델 및 이미지 임베더 로드 (캐싱으로 속도 최적화)
@st.cache_resource
def load_model_and_embedder():
    with open("animal_model.pkcls", "rb") as f:
        model = pickle.load(f)
    # 오렌지3 Image Embedding 위젯의 기본 모델인 'squeezenet' 사용
    # (만약 오렌지3에서 inception-v3 등을 쓰셨다면 해당 이름으로 변경하세요)
    embedder = ImageEmbedder(model="squeezenet")
    return model, embedder


try:
    model, embedder = load_model_and_embedder()
except Exception as e:
    st.error(f"모델을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 3. 카메라 입력 위젯
img_file = st.camera_input("동물을 프레임에 맞추고 촬영 버튼을 누르세요")

if img_file is not None:
    # 이미지 표시
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="촬영된 사진", use_container_width=True)

    with st.spinner("AI가 동물을 분석하고 있습니다..."):
        # 임시 파일로 저장 후 임베딩 추출
        temp_path = "temp_capture.jpg"
        image.save(temp_path)

        # Orange3 ImageEmbedder를 통해 특징 벡터 추출
        embeddings = embedder([temp_path])

        if embeddings and embeddings[0] is not None:
            # 1차원 벡터를 모델 입력 형태(2D array)로 변환
            X = np.array(embeddings[0]).reshape(1, -1)

            # 모델 예측 수행
            pred_idx = model.predict(X)[0]
            probs = model.predict_proba(X)[0]

            # 클래스 이름 가져오기
            class_names = model.domain.class_var.values
            predicted_animal = class_names[int(pred_idx)]
            confidence = probs[int(pred_idx)] * 100

            # 결과 출력
            st.success(
                f"### 예측 결과: **{predicted_animal}** ({confidence:.1f}%)"
            )

            # 전체 클래스별 확률 막대그래프 표시
            st.write("#### 클래스별 예측 확률")
            prob_dict = {
                name: float(prob) for name, prob in zip(class_names, probs)
            }
            st.bar_chart(prob_dict)
        else:
            st.error("이미지에서 특징을 추출하지 못했습니다. 다시 촬영해주세요.")