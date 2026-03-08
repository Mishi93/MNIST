import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import joblib
from streamlit_drawable_canvas import st_canvas

# Load trained Logistic Regression model
model = joblib.load("mnist_model.pkl")  # replace with your trained CSV model

st.title("Draw a Digit (0-9)")

# Manage canvas key for clearing
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas"

if st.button("Clear Canvas"):
    st.session_state.canvas_key = str(np.random.randint(0, 1e6))  # new key forces reset

# Canvas settings
canvas_result = st_canvas(
    fill_color="#FFFFFF",        # white background
    stroke_width=20,             # thick stroke for visibility
    stroke_color="#000000",      # black digit
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=st.session_state.canvas_key,
)

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess canvas image to match MNIST format for Logistic Regression."""
    
    # Convert to grayscale
    img = img.convert("L")
    
    # Apply small blur to reduce noise
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Threshold: remove faint pixels and convert to uint8
    arr = np.array(img)
    arr = np.where(arr > 128, 255, 0).astype(np.uint8)
    img = Image.fromarray(arr)
    
    # Crop to bounding box of the digit
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # Resize cropped digit to 20x20
    img = img.resize((20,20), resample=Image.Resampling.LANCZOS)
    
    # Pad to 28x28
    img = ImageOps.expand(img, border=4, fill=255)
    
    # Invert colors: MNIST uses black=0, white=255
    img = ImageOps.invert(img)
    
    # Flatten and normalize to [0,1]
    img_array = np.array(img).reshape(1,-1) / 255.0
    return img_array

# Predict button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8'))
        img_array = preprocess_image(img)
        
        # Predict
        prediction = model.predict(img_array)[0]
        probs = model.predict_proba(img_array)[0]
        
        st.success(f"Predicted Digit: {prediction}")
        st.write(f"Confidence: {np.max(probs)*100:.2f}%")
        
        # Show top 3 predictions
        top3_idx = probs.argsort()[-3:][::-1]
        st.write("Top 3 predictions:")
        for i in top3_idx:
            st.write(f"Digit {i}: {probs[i]*100:.2f}%")
