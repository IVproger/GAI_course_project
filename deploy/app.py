# python -m streamlit run deploy/app.py

import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os

from dotenv import load_dotenv
load_dotenv()

from src.enums import AnimalType

USE_BACKEND = True
DOCKER = False  # Changed to False since we're running locally

DUMMY_IMAGE_PATH = "images/styled_generated_image_2.png" 

if DOCKER:
    API_URL = "http://api:8000/predict"
else:
    API_URL = "http://localhost:8000/predict"

# Mapping of animal types to their unique tokens and descriptions
ANIMAL_OPTIONS = {
    "dog": {"token": "xon", "example": "A photo of a xon dog on the beach", "enum": AnimalType.DOG},
    "duck": {"token": "miff", "example": "A painting of a miff duck in a fantasy forest", "enum": AnimalType.DUCK},
}

st.set_page_config(page_title="Animal Image Generator", layout="centered")
st.title("🧠 Animal Prompt Generator")

# --- Dropdown menu ---
animal = st.selectbox("Choose an animal type:", list(ANIMAL_OPTIONS.keys()))
animal_info = ANIMAL_OPTIONS[animal]
unique_token = animal_info["token"]
example_prompt = animal_info["example"]
animal_type = animal_info["enum"]

# --- Highlight token in red inside the example prompt ---
highlighted_prompt = example_prompt.replace(
    unique_token, f"<span style='color:red;font-weight:bold'>{unique_token}</span>"
)

# --- Display token and prompt ---
st.markdown(f"**Unique Token for '{animal}':** `{unique_token}`")
st.markdown("**Example Prompt:**", unsafe_allow_html=True)
st.markdown(f"{highlighted_prompt}", unsafe_allow_html=True)

# --- Prompt input field ---
prompt = st.text_input("Write your custom prompt (include the unique token!)", value=example_prompt)

# --- Generate button ---
if st.button("Generate Image"):
    with st.spinner("Generating..."):
        try:
            if USE_BACKEND:
                response = requests.post(API_URL, json={
                    "prompt": prompt,
                    "animal_type": animal_type.value
                })

                if response.status_code == 200:
                    image_path = response.json()["image_path"]

                    if os.path.exists(image_path):
                        image = Image.open(image_path)
                        st.image(image, caption="Generated Image")
                    else:
                        st.warning("Image path returned, but file not found on disk.")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            else:
                if os.path.exists(DUMMY_IMAGE_PATH):
                    image = Image.open(DUMMY_IMAGE_PATH)
                    st.image(image, caption="Generated Image (Mock)")
                else:
                    st.error(f"Dummy image not found at `{DUMMY_IMAGE_PATH}`. Please add one.")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
