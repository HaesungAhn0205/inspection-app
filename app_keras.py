import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras

# ── 1. 페이지 설정 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="가죽 불량 탐지 AI 시스템",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 가죽 이상 탐지 AI 시스템")
st.caption("가죽 이미지를 업로드하거나 카메라로 촬영하여 정상/불량 여부를 실시간으로 판정합니다.")

# ── 설정 및 상수 ──────────────────────────────────────────────────
MODEL_PATH     = "./weights/leather_model.keras"
INPUT_IMG_SIZE = (224, 224)
CLASSES        = ["정상", "불량"]


# ── 2. 모델 로드 (캐싱 적용) ────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        # Streamlit 화면에 에러를 표시하고 프로세스를 중단합니다.
        st.error(f"❌ 모델 파일을 찾을 수 없습니다. 경로를 확인해주세요: {MODEL_PATH}")
        st.stop()
    
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# ── 3. 이미지 전처리 및 추론 로직 (기존 로직 유지) ──────────────────────
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = keras.applications.vgg16.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict(model, pil_img):
    arr   = preprocess(pil_img)
    prob  = float(model.predict(arr, verbose=0)[0][0])
    label = CLASSES[1 if prob > 0.5 else 0]
    return label, prob


# ── 메인 앱 실행 영역 ───────────────────────────────────────────────
def main():
    # 모델 로드 (최초 1회만 로드됨)
    model = load_model()

    # 사이드바 또는 메인 화면에 입력 방식 선택
    st.subheader("⚙️ 이미지 입력 설정")
    input_mode = st.radio(
        "입력 방식을 선택하세요",
        ("파일 업로드", "카메라 촬영"),
        horizontal=True
    )

    pil_img = None

    # 입력 방식에 따른 컴포넌트 렌더링
    if input_mode == "파일 업로드":
        uploaded_file = st.file_uploader(
            "가죽 이미지를 업로드하세요", 
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)

    elif input_mode == "카메라 촬영":
        camera_file = st.camera_input("웹캠을 통해 가죽을 촬영하세요")
        if camera_file is not None:
            pil_img = Image.open(camera_file)

    # 이미지 미리보기
    if pil_img is not None:
        st.image(pil_img, caption="입력된 이미지 크기 (원본)", use_column_width=True)

    st.write("---")

    # ── 4. 검사 실행 및 결과 표시 ────────────────────────────────────
    if st.button("🔍 검사 시작", type="primary"):
        if pil_img is None:
            st.warning("⚠️ 먼저 이미지를 업로드하거나 카메라로 촬영해 주세요.")
        else:
            with st.spinner("AI가 이미지를 분석 중입니다..."):
                # 추론 실행
                label, prob = predict(model, pil_img)
            
            st.subheader("📊 판정 결과")
            
            # 정상이면 success(녹색), 불량이면 error(빨간색)로 분기 처리
            if label == "정상":
                st.success(f"✅ 판정: **{label} 제품**입니다.")
            else:
                st.error(f"🚨 판정: **{label} 제품**입니다. 주의가 필요합니다.")

            # st.metric으로 확률을 나란히 표시
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="정상 확률", value=f"{(1 - prob):.1%}")
            with col2:
                st.metric(label="불량 확률", value=f"{prob:.1%}")

            # st.progress로 불량 확률 시각화
            st.write("**불량 위험도 위험 수치**")
            st.progress(prob)


if __name__ == "__main__":
    main()