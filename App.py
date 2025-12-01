import os
import io
import base64
from typing import Tuple, Optional

import streamlit as st
from PIL import Image
from google.cloud import aiplatform
from google.protobuf.json_format import MessageToDict  # <-- added


# =========================
# CONFIG
# =========================
PROJECT_ID = os.getenv("PROJECT_ID", "mlops-project-479512")
LOCATION = os.getenv("LOCATION", "europe-west3")
ENDPOINT_ID = os.getenv("ENDPOINT_ID", "6025044444259024896")

ENDPOINT_NAME = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
)

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


# =========================
# STREAMLIT PAGE
# =========================
st.set_page_config(
    page_title="Image Classifier — Vertex AI Demo",
    layout="wide",
)


# =========================
# VERTEX AI CLIENT
# =========================
@st.cache_resource
def get_prediction_client():
    return aiplatform.gapic.PredictionServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
    )


def predict_image(image: Image.Image):
    """Send image to Vertex AI endpoint."""
    try:
        buf = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        b64 = base64.b64encode(img_bytes).decode("utf-8")
        instance = {"data": {"b64": b64}}

        client = get_prediction_client()
        response = client.predict(
            endpoint=ENDPOINT_NAME,
            instances=[instance],
            parameters=None,
        )

        if not response.predictions:
            return None, response, "Empty prediction."

        pred = response.predictions[0]
        if hasattr(pred, "number_value"):
            idx = int(pred.number_value)
        else:
            idx = int(pred)

        if idx not in range(len(CLASS_NAMES)):
            return None, response, f"Invalid class index {idx}"

        return idx, response, None

    except Exception as e:
        return None, None, str(e)


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Settings")

    st.write("**Project:**")
    st.write(PROJECT_ID)

    st.write("**Location:**")
    st.write(LOCATION)

    st.markdown("---")

    st.header("Classes:")
    for i, name in enumerate(CLASS_NAMES):
        st.write(f"{i}. {name.capitalize()}")


# =========================
# MAIN PAGE
# =========================
st.title("Intel Image Classifier")
st.write("Vertex AI")

left, right = st.columns([2.2, 1], gap="large")

# -------- Image Upload --------
with left:
    st.subheader("Upload an image")
    uploaded_file = st.file_uploader(
        "Choose a JPG/JPEG file",
        type=["jpg", "jpeg"],
    )

    image_obj = None
    if uploaded_file:
        try:
            image_obj = Image.open(uploaded_file)
            st.image(image_obj, caption=uploaded_file.name, use_column_width=True)
        except:
            st.error("Could not open image.")


# -------- Prediction --------
with right:
    st.subheader("Prediction")

    if not uploaded_file:
        st.info("Upload an image first.")
    else:
        if st.button("Predict with Vertex AI"):
            with st.spinner("Sending to Vertex AI..."):
                idx, raw_response, error = predict_image(image_obj)

            if error:
                st.error(error)
            else:
                label = CLASS_NAMES[idx].capitalize()
                st.success(f"Prediction: **{label}**")
                st.caption(f"Predicted class index: {idx}")

                with st.expander("Show raw Vertex AI response"):
                    # ---- only line changed here ----
                    st.json(MessageToDict(raw_response._pb))


# Footer
st.write("---")
st.write("Developed by Gourangkumar & Nishant")