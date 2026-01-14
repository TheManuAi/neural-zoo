import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Configuration
MODEL_PATH = 'doodle_cnn_best.keras'

# 50 animal classes (must match training order)
CLASSES = [
    "ant", "bat", "bear", "bee", "bird", "butterfly", "camel", "cat", "cow",
    "crab", "crocodile", "dog", "dolphin", "dragon", "duck", "elephant", "fish",
    "flamingo", "frog", "giraffe", "hedgehog", "horse", "kangaroo", "lion",
    "lobster", "mermaid", "monkey", "mosquito", "mouse", "octopus", "owl",
    "panda", "parrot", "penguin", "pig", "rabbit", "raccoon", "rhinoceros",
    "scorpion", "sea turtle", "shark", "sheep", "snail", "snake", "spider",
    "squirrel", "swan", "tiger", "whale", "zebra"
]

# Emojis for display
ANIMAL_EMOJIS = {
    "ant": "🐜", "bat": "🦇", "bear": "🐻", "bee": "🐝", "bird": "🐦",
    "butterfly": "🦋", "camel": "🐫", "cat": "🐱", "cow": "🐄", "crab": "🦀",
    "crocodile": "🐊", "dog": "🐕", "dolphin": "🐬", "dragon": "🐉", "duck": "🦆",
    "elephant": "🐘", "fish": "🐟", "flamingo": "🦩", "frog": "🐸", "giraffe": "🦒",
    "hedgehog": "🦔", "horse": "🐴", "kangaroo": "🦘", "lion": "🦁", "lobster": "🦞",
    "mermaid": "🧜", "monkey": "🐒", "mosquito": "🦟", "mouse": "🐭", "octopus": "🐙",
    "owl": "🦉", "panda": "🐼", "parrot": "🦜", "penguin": "🐧", "pig": "🐷",
    "rabbit": "🐰", "raccoon": "🦝", "rhinoceros": "🦏", "scorpion": "🦂", "sea turtle": "🐢",
    "shark": "🦈", "sheep": "🐑", "snail": "🐌", "snake": "🐍", "spider": "🕷️",
    "squirrel": "🐿️", "swan": "🦢", "tiger": "🐯", "whale": "🐋", "zebra": "🦓"
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_canvas(canvas_result):
    """
    Preprocesses the canvas drawing to match QuickDraw format.
    
    Steps:
    1. Converts to grayscale.
    2. Inverts colors (QuickDraw is white-on-black).
    3. Centers the drawing using a bounding box.
    4. Resizes to 28x28.
    """
    if canvas_result.image_data is None:
        return None
    
    img = canvas_result.image_data
    img_gray = np.mean(img[:, :, :3], axis=2)
    
    # Invert colors
    img_gray = 255 - img_gray
    
    # Check if canvas is empty
    if np.max(img_gray) < 25:
        return None
    
    # Find bounding box
    rows = np.any(img_gray > 25, axis=1)
    cols = np.any(img_gray > 25, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Crop to bounding box
    cropped = img_gray[rmin:rmax+1, cmin:cmax+1]
    
    # Add padding to verify square shape
    h, w = cropped.shape
    max_dim = max(h, w)
    margin = max(int(max_dim * 0.1), 4)
    final_size = max_dim + 2 * margin
    
    # Center image
    centered = np.zeros((final_size, final_size), dtype=np.float32)
    y_offset = (final_size - h) // 2
    x_offset = (final_size - w) // 2
    centered[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    
    # Resize to 28x28
    img_pil = Image.fromarray(centered.astype(np.uint8))
    img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Normalize pixel values
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

def main():
    st.set_page_config(
        page_title="Neural Zoo - Animal Doodle Classifier",
        page_icon="🦁",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # CSS for styling and mobile responsiveness
    st.markdown("""
    <style>
        /* Base styles */
        .main-header {
            text-align: center;
            padding: 1rem 0;
        }
        .prediction-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            color: white;
            text-align: center;
            margin: 1rem 0;
        }
        .prediction-animal {
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        .prediction-name {
            font-size: 1.5rem;
            font-weight: bold;
            text-transform: uppercase;
        }
        .confidence-text {
            font-size: 1rem;
            opacity: 0.9;
        }
        .stButton > button {
            border-radius: 0.5rem;
            font-weight: 600;
            min-height: 50px;
        }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .prediction-animal {
                font-size: 2.5rem;
            }
            .prediction-name {
                font-size: 1.2rem;
            }
            .stButton > button {
                min-height: 60px;
                font-size: 1.1rem;
            }
            /* Make canvas smaller on mobile */
            canvas {
                max-width: 100% !important;
                height: auto !important;
            }
            /* Stack columns vertically on mobile */
            [data-testid="column"] {
                width: 100% !important;
            }
        }
        
        @media (max-width: 480px) {
            .prediction-animal {
                font-size: 2rem;
            }
            .prediction-name {
                font-size: 1rem;
            }
            h1 {
                font-size: 1.8rem !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    
    # Header
    st.markdown("<div class='main-header'>", unsafe_allow_html=True)
    st.title("🦁 Neural Zoo")
    st.caption("AI-powered animal doodle recognition • 50 animal classes • CNN trained on QuickDraw dataset")
    st.warning("⚠️ Note: Model accuracy is ~79%. Predictions may vary based on drawing style.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.subheader("✏️ Draw an Animal")
        
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=10,
            stroke_color="#000000",
            background_color="#FFFFFF",
            width=450,
            height=450,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}"
        )
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔮 Predict", use_container_width=True, type="primary"):
                img = preprocess_canvas(canvas_result)
                if img is not None and np.sum(img) > 0.01:
                    model = load_model()
                    preds = model.predict(img, verbose=0)[0]
                    top_idx = np.argsort(preds)[::-1][:5]
                    st.session_state.prediction = {
                        "top_class": CLASSES[top_idx[0]],
                        "top_conf": preds[top_idx[0]],
                        "top_5": [(CLASSES[i], preds[i]) for i in top_idx]
                    }
                else:
                    st.toast("Please draw something first!", icon="⚠️")
        
        with btn_col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.canvas_key += 1
                st.session_state.prediction = None
                st.rerun()
    
    with col2:
        st.subheader("🎯 Prediction")
        
        if st.session_state.prediction:
            pred = st.session_state.prediction
            emoji = ANIMAL_EMOJIS.get(pred["top_class"], "🐾")
            
            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-animal">{emoji}</div>
                <div class="prediction-name">{pred["top_class"]}</div>
                <div class="confidence-text">{pred["top_conf"]*100:.1f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### Possible Matches")
            for name, conf in pred["top_5"][1:]:
                emoji = ANIMAL_EMOJIS.get(name, "🐾")
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(float(conf))
                with col_b:
                    st.caption(f"{emoji} {name}")
        else:
            st.info("👈 Draw an animal on the canvas and click **Predict**")
            
            with st.expander("💡 Drawing Tips"):
                st.markdown("""
                - Draw clear, simple shapes.
                - Use the full canvas space.
                - Focus on distinctive features (ears, tails, trunks).
                """)
    
    st.divider()
    
    with st.expander("🐾 Supported Animals", expanded=False):
        cols = st.columns(10)
        for i, animal in enumerate(CLASSES):
            emoji = ANIMAL_EMOJIS.get(animal, "🐾")
            cols[i % 10].markdown(f"{emoji} {animal}")

if __name__ == "__main__":
    main()
