import streamlit as st
import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import glob
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import from src
from src.utils import (
    get_class_names,
    calibrate_predictions,
    preprocess_image,
    get_top_predictions
)

# ==================== DOWNLOAD MODEL FROM GOOGLE DRIVE ====================
import requests

MODEL_PATH = 'models/lego_model_final.h5'

def download_model():
    """Download model from Google Drive if not exists"""
    if os.path.exists(MODEL_PATH):
        return
    
    st.info("📥 Downloading AI model from Google Drive... This may take 3-5 minutes.")
    
    # Your File ID
    file_id = "1hM7JwSTDoVmWRIEflgDh80s-rdpCMrIX"
    url = f"https://drive.google.com/file/d/1hM7JwSTDoVmWRIEflgDh80s-rdpCMrIX/view?usp=sharing"
    
    try:
        # Create models folder
        os.makedirs('models', exist_ok=True)
        
        # Download the file
        response = requests.get(url, stream=True)
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress
        with open(MODEL_PATH, 'wb') as f:
            progress_bar = st.progress(0)
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress_bar.progress(min(downloaded / total_size, 1.0))
        
        st.success("✅ Model downloaded successfully!")
        
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.info("💡 Please check your internet connection")

# Download model BEFORE loading it
download_model()
# ====================================================
# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="LEGO Brick Recognizer",
    page_icon="🧱",  # Use emoji or URL: "https://www.lego.com/favicon.ico"
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CONFIG ====================
IMG_SIZE = (128, 128)
MODEL_PATH = 'models/lego_model_final.h5'
DATASET_PATH = 'dataset'
TEMPERATURE = 1.5
# ================================================

# ==================== CUSTOM CSS WITH LEGO BACKGROUND ====================
st.markdown("""
    <style>
    /* Main background with LEGO image */
    .stApp {
        background-image: url('https://w0.peakpx.com/wallpaper/104/436/HD-wallpaper-lego-toy-block-colorfulness-kids.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Dark overlay for readability */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        z-index: -1;
    }
    
    /* Main content background - semi-transparent */
    .main > div {
        background: rgba(30, 30, 40, 0.85);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Headers with LEGO colors */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.8) !important;
    }
    
    h1 {
        background: linear-gradient(135deg, #FF6B35, #FFD700, #FF6B35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        text-shadow: none !important;
    }
    
    /* Labels and text */
    p, label, .stMarkdown, .stCaption {
        color: #FFFFFF !important;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.9) !important;
    }
    
    /* Buttons with LEGO theme */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B35, #E53E3E) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: bold !important;
        padding: 0.75rem 2rem !important;
        border: 2px solid #FFD700 !important;
        transition: all 0.3s ease !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5) !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.05) !important;
        background: linear-gradient(135deg, #FF8C5A, #FF6B35) !important;
        box-shadow: 0 0 20px rgba(255, 107, 53, 0.5) !important;
    }
    
    /* Prediction cards */
    .prediction-card {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border: 2px solid rgba(255, 215, 0, 0.3) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        color: white !important;
    }
    
    .success {
        border-left: 5px solid #00FF00 !important;
        background: rgba(0, 255, 0, 0.1) !important;
    }
    
    .warning {
        border-left: 5px solid #FFD700 !important;
        background: rgba(255, 215, 0, 0.1) !important;
    }
    
    .error {
        border-left: 5px solid #FF0000 !important;
        background: rgba(255, 0, 0, 0.1) !important;
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px dashed #FFD700 !important;
        border-radius: 15px !important;
        color: white !important;
    }
    
    .stFileUploader > div > div:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        border-color: #FF6B35 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FF6B35, #FFD700) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: white !important;
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-12oz5g7 {
        background: rgba(0, 0, 0, 0.7) !important;
        backdrop-filter: blur(10px);
    }
    
    .sidebar-content {
        color: white !important;
    }
    
    /* Dataframe */
    .dataframe {
        background: rgba(0, 0, 0, 0.5) !important;
        color: white !important;
    }
    
    .dataframe th {
        background: rgba(255, 215, 0, 0.3) !important;
        color: white !important;
    }
    
    .dataframe td {
        color: white !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: white !important;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 215, 0, 0.3) !important;
        color: #FFD700 !important;
    }
    
    /* LEGO Logo styling in title */
    .lego-title-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 1rem 0;
    }
    
    .lego-logo {
        width: 180px;
        height: auto;
        filter: invert(16%) sepia(88%) saturate(7207%) hue-rotate(1deg) brightness(102%) contrast(115%);
        margin-bottom: 0.5rem;
    }
    
    .lego-title {
        font-size: 3.0rem;
        color: #FFFFFF;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.8);
        margin: 0;
    }
    
    .lego-subtitle {
        font-size: 1.2rem;
        color: #FFD700;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.9);
    }
    </style>
""", unsafe_allow_html=True)

# ==================== TITLE SECTION WITH LOGO ====================
st.markdown(
    """
    <div class="lego-title-container">
        <img src="https://upload.wikimedia.org/wikipedia/commons/2/24/LEGO_logo.svg"
             alt="LEGO Logo"
             class="lego-logo">
        <h1 class="lego-title">Automated LEGO Brick Recognition System</h1>
        <p class="lego-subtitle">Upload LEGO bricks for instant identification</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model_func():
    """Load model with caching"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at {MODEL_PATH}")
        alternatives = [
            'models/lego_model_improved.h5',
            'models/lego_model.h5'
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                st.warning(f"Using alternative model: {alt}")
                model = load_model(alt, compile=False)
                return model
        return None
    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def get_classes():
    """Get class names with caching"""
    class_names = get_class_names(DATASET_PATH)
    if not class_names:
        st.error(f"❌ No classes found in {DATASET_PATH}")
    return class_names

# Load model and classes
model = load_model_func()
class_names = get_classes()

if model is None or not class_names:
    st.stop()

# ==================== FUNCTIONS ====================
def predict_single_image(img_path):
    """Predict a single image"""
    try:
        img_array, original_img = preprocess_image(img_path, target_size=IMG_SIZE)
        raw_predictions = model.predict(img_array, verbose=0)
        calibrated_predictions = calibrate_predictions(raw_predictions, TEMPERATURE)
        
        confidence = float(np.max(calibrated_predictions) * 100)
        predicted_class = class_names[np.argmax(calibrated_predictions)]
        top_predictions = get_top_predictions(calibrated_predictions, class_names, top_k=3)
        
        return {
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'calibrated_predictions': calibrated_predictions,
            'image': original_img
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def predict_batch_images(image_paths):
    """Predict multiple images"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, img_path in enumerate(image_paths):
        status_text.text(f"Processing {idx+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        result = predict_single_image(img_path)
        result['filename'] = os.path.basename(img_path)
        result['filepath'] = img_path
        results.append(result)
        progress_bar.progress((idx + 1) / len(image_paths))
    
    status_text.empty()
    progress_bar.empty()
    return results

def display_prediction_result(result):
    """Display single prediction result"""
    if not result['success']:
        st.error(f"Error: {result['error']}")
        return
    
    confidence = result['confidence']
    predicted_class = result['predicted_class']
    top_predictions = result['top_predictions']
    
    # Display image
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(result['image'], caption="Analyzed Image", width=200)
    
    with col2:
        # Confidence-based display
        if confidence > 70:
            st.markdown(f"""
            <div class="prediction-card success">
                <h3>✅ Predicted Brick: {predicted_class}</h3>
                <p>Confidence: <b>{confidence:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
        elif confidence > 50:
            st.markdown(f"""
            <div class="prediction-card warning">
                <h3>⚠️ Maybe: {predicted_class}</h3>
                <p>Confidence: <b>{confidence:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card error">
                <h3>❌ Not a LEGO brick</h3>
                <p>Confidence: <b>{confidence:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Top predictions with progress bars
        st.markdown("**Top Predictions:**")
        for i, pred in enumerate(top_predictions, 1):
            col_bar, col_label = st.columns([3, 1])
            with col_bar:
                progress_value = float(pred['confidence'] / 100)
                st.progress(progress_value)
            with col_label:
                st.write(f"{float(pred['confidence']):.1f}%")
            st.caption(f"{i}. {pred['class']}")

# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs(["📸 Single Image", "📚 Batch Processing", "📊 Dashboard"])

# ==================== TAB 1: SINGLE IMAGE ====================
with tab1:
    st.markdown("### Upload a single LEGO brick image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "png", "jpeg"],
        help="Upload a clear photo of a single LEGO brick"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Predict
        with st.spinner("🔍 Analyzing image..."):
            result = predict_single_image(temp_path)
            display_prediction_result(result)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ==================== TAB 2: BATCH PROCESSING ====================
with tab2:
    st.markdown("### 📚 Batch Processing - Upload multiple images at once")
    st.info("💡 Upload a folder of LEGO brick images for batch prediction")
    
    # Option 1: Upload multiple files
    uploaded_files = st.file_uploader(
        "Choose multiple images...", 
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        help="Upload multiple LEGO brick images for batch processing"
    )
    
    # Option 2: Process folder path
    folder_path = st.text_input(
        "Or enter folder path (optional):",
        placeholder="C:/path/to/lego/images",
        help="Enter the full path to a folder containing LEGO brick images"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        process_batch = st.button("🚀 Process Batch", use_container_width=True)
    
    if process_batch:
        image_paths = []
        
        # Collect images from uploaded files
        if uploaded_files:
            # Create temp folder for uploaded files
            temp_folder = "temp_batch"
            os.makedirs(temp_folder, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_folder, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image_paths.append(temp_path)
        
        # Collect images from folder path
        if folder_path and os.path.exists(folder_path):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if not image_paths:
            st.warning("⚠️ No images found. Please upload files or provide a valid folder path.")
        else:
            st.info(f"📊 Found {len(image_paths)} images to process")
            
            # Show images grid
            cols = st.columns(4)
            for idx, path in enumerate(image_paths[:8]):  # Show first 8 images
                with cols[idx % 4]:
                    try:
                        img = Image.open(path)
                        st.image(img, caption=os.path.basename(path), use_container_width=True)
                    except:
                        pass
            
            if len(image_paths) > 8:
                st.caption(f"... and {len(image_paths) - 8} more images")
            
            # Process batch
            with st.spinner("🔍 Processing batch..."):
                results = predict_batch_images(image_paths)
            
            # Display results
            st.markdown("### 📊 Batch Results")
            
            # Create dataframe
            df_data = []
            for result in results:
                if result['success']:
                    df_data.append({
                        'Filename': result['filename'],
                        'Prediction': result['predicted_class'],
                        'Confidence': f"{result['confidence']:.2f}%",
                        'Status': '✅'
                    })
                else:
                    df_data.append({
                        'Filename': result['filename'],
                        'Prediction': 'Error',
                        'Confidence': 'N/A',
                        'Status': '❌'
                    })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Summary statistics
            successful = sum(1 for r in results if r['success'])
            st.markdown(f"""
            <div class="prediction-card">
                <h4>📊 Summary</h4>
                <p>Total Images: <b>{len(results)}</b></p>
                <p>Successful: <b style="color: #00FF00;">{successful}</b></p>
                <p>Failed: <b style="color: #FF0000;">{len(results) - successful}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show detailed results
            with st.expander("📋 Detailed Results"):
                for result in results:
                    if result['success']:
                        st.markdown(f"""
                        **{result['filename']}** → ✅ {result['predicted_class']} ({result['confidence']:.2f}%)
                        """)
                    else:
                        st.markdown(f"""
                        **{result['filename']}** → ❌ Error: {result['error']}
                        """)
            
            # Download results as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Clean up temp folder
            if uploaded_files and os.path.exists(temp_folder):
                import shutil
                shutil.rmtree(temp_folder)

# ==================== TAB 3: DASHBOARD ====================
with tab3:
    st.markdown("### 📊 Recognition Dashboard")
    
    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🏗️ Total Classes",
            value=len(class_names)
        )
    
    with col2:
        st.metric(
            label="📸 Model Input Size",
            value=f"{IMG_SIZE[0]}x{IMG_SIZE[1]}"
        )
    
    with col3:
        st.metric(
            label="🎯 Confidence Threshold",
            value="70%"
        )
    
    with col4:
        st.metric(
            label="⚡ Status",
            value="✅ Active"
        )
    
    # Class distribution
    st.markdown("### 🏗️ Available Classes")
    
    # Count images per class
    class_counts = {}
    for class_name in class_names:
        class_path = os.path.join(DATASET_PATH, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_name] = count
    
    if class_counts:
        # Create a simple bar chart using matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        bars = ax.bar(list(class_counts.keys()), list(class_counts.values()), 
                     color='#FF6B35', edgecolor='#FFD700', linewidth=2)
        
        # Customize chart
        ax.set_xlabel('Classes', color='white')
        ax.set_ylabel('Number of Images', color='white')
        ax.set_title('Images per Class', color='white')
        ax.tick_params(axis='x', colors='white', rotation=45)
        ax.tick_params(axis='y', colors='white')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', color='white')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Also show as a simple table
        st.dataframe(
            pd.DataFrame(list(class_counts.items()), columns=['Class', 'Images']),
            use_container_width=True,
            hide_index=True
        )

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## LEGO Brick AI")
    st.markdown("---")
    
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Single Image**: Upload one LEGO brick photo
    2. **Batch Processing**: Upload multiple images
    3. **Dashboard**: View dataset statistics
    
    ### 💡 Tips
    - Use good lighting
    - Plain background
    - Center the brick
    - Avoid blurry images
    
    ### 🎯 Confidence Levels
    - 🟢 >70%: High confidence
    - 🟡 50-70%: Low confidence
    - 🔴 <50%: Not recognized
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Info")
    st.markdown(f"""
    - **Architecture**: ResNet50
    - **Input Size**: {IMG_SIZE[0]}x{IMG_SIZE[1]}
    - **Classes**: {len(class_names)}
    - **Temperature**: {TEMPERATURE}
    - **Status**: 🟢 Online
    """)
    
    st.markdown("---")
    st.caption("Built using TensorFlow & Streamlit")

# ==================== FOOTER ====================
st.markdown("---")
st.caption("LEGO Brick Recognition System | Powered by AI")