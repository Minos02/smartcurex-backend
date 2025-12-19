from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import tempfile
import os
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from datetime import timedelta
from datetime import datetime 
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Google Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
jwt = JWTManager(app)

CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})

# ========== LOAD MODELS ==========
print("🔄 Loading models...")

try:
    heart_disease_model = pickle.load(open('../Piyush Gupta/SmartCureX/models/New Datasets/heart_disease_xgb_model.pkl', 'rb'))
    print("✅ Heart Disease loaded")
except:
    heart_disease_model = None
    print("⚠️ Heart Disease missing")

try:
    diabetes_model = pickle.load(open('../Piyush Gupta/SmartCureX/models/New Datasets/diabetes_xgb_model.pkl', 'rb'))
    print("✅ Diabetes loaded")
except:
    diabetes_model = None
    print("⚠️ Diabetes missing")

try:
    breast_cancer_model = pickle.load(open('../Piyush Gupta/SmartCureX/models/New Datasets/breast_cancer_xgb_model.pkl', 'rb'))
    print("✅ Breast Cancer loaded")
except:
    breast_cancer_model = None
    print("⚠️ Breast Cancer missing")

try:
    alzheimers_model = keras.models.load_model('../Piyush Gupta/SmartCureX/models/New Datasets/alzheimer_cnn_20251005_004009.h5')
    print("✅ Alzheimer's loaded")
except:
    alzheimers_model = None
    print("⚠️ Alzheimer's missing")

try:
    brain_tumor_model = keras.models.load_model('../Piyush Gupta/SmartCureX/samplez/BrainTumor/brain_tumor_model_cpu.h5')
    print("✅ Brain Tumor loaded")
except:
    brain_tumor_model = None
    print("⚠️ Brain Tumor missing")

try:
    pneumonia_model = keras.models.load_model('../Piyush Gupta/SmartCureX/models/New Datasets/pneumonia_binary_best_model.h5')
    print("✅ Pneumonia loaded")
except:
    pneumonia_model = None
    print("⚠️ Pneumonia missing")

try:
    covid_model = keras.models.load_model('../Piyush Gupta/SmartCureX/models/New Datasets/covid_binary_best_model.h5')
    print("✅ COVID loaded")
except:
    covid_model = None
    print("⚠️ COVID missing")

# ========== IMAGE VALIDATION WITH GEMINI VISION ==========
def validate_medical_image(image_file, scan_type="medical scan"):
    """
    Validate if uploaded image is a medical scan by checking if it's grayscale.
    Medical scans (X-rays, MRIs, CTs) are typically grayscale.
    Returns (is_valid: bool, message: str)
    """
    try:
        print(f"🔍 Validating image as {scan_type}...")
        
        # Read image bytes
        image_file.seek(0)
        image_bytes = image_file.read()
        image_file.seek(0)
        
        # Open image with PIL
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img)
        
        print(f"📦 Image shape: {img_array.shape}")
        print(f"🎨 Image mode: {img.mode}")
        
        # Check if image is grayscale or close to grayscale
        is_grayscale = False
        
        # Method 1: Check PIL mode
        if img.mode in ['L', 'LA', 'I', 'F']:  # Grayscale modes
            is_grayscale = True
            print("✅ Image is grayscale (PIL mode)")
        
        # Method 2: Check if RGB channels are similar (grayscale in RGB format)
        elif img.mode == 'RGB' or img.mode == 'RGBA':
            # Convert to numpy array
            if len(img_array.shape) == 3:
                r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
                
                # Calculate variance between channels
                rg_diff = np.abs(r.astype(float) - g.astype(float)).mean()
                rb_diff = np.abs(r.astype(float) - b.astype(float)).mean()
                gb_diff = np.abs(g.astype(float) - b.astype(float)).mean()
                
                avg_diff = (rg_diff + rb_diff + gb_diff) / 3
                
                print(f"📊 Average channel difference: {avg_diff:.2f}")
                
                # If channels are very similar, it's grayscale (threshold: 15)
                if avg_diff < 15:
                    is_grayscale = True
                    print("✅ Image is grayscale (RGB but channels match)")
                else:
                    print(f"❌ Image has color (channel difference: {avg_diff:.2f})")
            elif len(img_array.shape) == 2:
                is_grayscale = True
                print("✅ Image is grayscale (2D array)")
        
        if is_grayscale:
            print(f"✅ VALIDATION PASSED: Grayscale {scan_type} detected")
            return True, "Valid medical scan"
        else:
            print(f"❌ VALIDATION FAILED: Image has colors (not a medical scan)")
            return False, f"Invalid image. Medical scans must be grayscale (black & white). Please upload a valid {scan_type}, not a color photograph."
            
    except Exception as e:
        print(f"⚠️ Validation error: {e}")
        import traceback
        traceback.print_exc()
        
        # On error, REJECT for safety
        print(f"⚠️ REJECTING due to validation error")
        return False, "Image validation failed. Please upload a valid PNG or JPG medical scan."




# ========== IMAGE PREPROCESSING ==========
def preprocess_image(image_file, target_size=(224, 224)):
    """Generic image preprocessing"""
    img = Image.open(image_file.stream).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


# ========== AUTH WITH SUPABASE ==========
@app.route('/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    email, password, name = data.get('email'), data.get('password'), data.get('name')
    
    if not all([email, password, name]):
        return jsonify({'success': False, 'error': 'All fields required'}), 400
    
    try:
        # Check if user exists
        existing = supabase.table('users').select('email').eq('email', email).execute()
        if existing.data:
            return jsonify({'success': False, 'error': 'Email exists'}), 400
        
        # Hash password and insert into Supabase
        hashed_pw = generate_password_hash(password)
        result = supabase.table('users').insert({
            'email': email,
            'password_hash': hashed_pw,
            'name': name
        }).execute()
        
        user = result.data[0]
        user_id = user['id']
        
        # Create JWT token
        token = create_access_token(identity={'id': user_id, 'email': email, 'name': name})
        
        return jsonify({
            'success': True, 
            'token': token, 
            'user': {'id': user_id, 'email': email, 'name': name}
        })
        
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    email, password = data.get('email'), data.get('password')
    
    try:
        # Get user from Supabase
        result = supabase.table('users').select('*').eq('email', email).execute()
        
        if not result.data:
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
        
        user = result.data[0]
        
        # Verify password
        if not check_password_hash(user['password_hash'], password):
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
        
        # Create JWT token
        token = create_access_token(identity={'id': user['id'], 'email': user['email'], 'name': user['name']})
        
        return jsonify({
            'success': True, 
            'token': token, 
            'user': {'id': user['id'], 'email': user['email'], 'name': user['name']}
        })
        
    except Exception as e:
        print(f"❌ Login error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 401


# ========== PREDICTIONS WITH IMAGE VALIDATION ==========
@app.route('/predict/heart-disease', methods=['POST', 'OPTIONS'])
def predict_heart_disease():
    if request.method == 'OPTIONS':
        return '', 200
    
    print("=== HEART DISEASE PREDICTION ===")
        
    if heart_disease_model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        print(f"📊 Received data: {data}")
        
        features = np.array([[
            float(data.get('age', 43)),
            1.0, float(data.get('tcp', 0)), float(data.get('nmv', 130)), 200.0, 0.0, 0.0,
            float(data.get('mhra', 150)), float(data.get('eia', 0)), float(data.get('op', 0.0)),
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(data.get('thal', 2)), 0.0, 0.0, 0.0
        ]], dtype=np.float32)
        
        prediction = heart_disease_model.predict(features)
        proba = heart_disease_model.predict_proba(features)[0]
        result = 'Healthy' if int(prediction[0]) == 0 else 'Heart Disease Risk'
        
        print(f"✅ Heart disease prediction: {result} ({max(proba)*100:.2f}%)")
        
        # Save to Supabase
        try:
            supabase.table('predictions').insert({
                'user_id': None,
                'disease_type': 'heart',
                'prediction_result': result,
                'confidence_score': float(max(proba)) * 100,
                'input_data': data
            }).execute()
            print(f"✅ Heart Disease saved to Supabase")
        except Exception as db_err:
            print(f"❌ Supabase INSERT ERROR: {db_err}")
        
        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': round(float(max(proba)) * 100, 2),
            'all_probabilities': {
                'Healthy': round(float(proba[0]) * 100, 2), 
                'Heart Disease Risk': round(float(proba[1]) * 100, 2)
            },
            'patient': {'name': data.get('name'), 'age': data.get('age')}
        })
    except Exception as e:
        print(f"❌ Heart disease error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict/diabetes', methods=['POST', 'OPTIONS'])
def predict_diabetes():
    if request.method == 'OPTIONS':
        return '', 200
    
    print("=== DIABETES PREDICTION ===")
        
    if diabetes_model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        print(f"📊 Received data: {data}")
        
        features = np.array([[
            float(data.get('pregnancies', 0)), 
            float(data.get('glucose', 100)),
            float(data.get('bp', 70)), 
            float(data.get('skin_thickness', 0)),
            float(data.get('insulin', 0)), 
            float(data.get('bmi', 25)),
            float(data.get('diabetes_pedigree', 0)), 
            float(data.get('age', 30))
        ]], dtype=np.float32)
        
        prediction = diabetes_model.predict(features)
        proba = diabetes_model.predict_proba(features)[0]
        result = 'Diabetic' if int(prediction[0]) == 1 else 'Non-Diabetic'
        
        print(f"✅ Diabetes prediction: {result} ({max(proba)*100:.2f}%)")
        
        # Save to Supabase
        try:
            supabase.table('predictions').insert({
                'user_id': None,
                'disease_type': 'diabetes',
                'prediction_result': result,
                'confidence_score': float(max(proba)) * 100,
                'input_data': data
            }).execute()
            print(f"✅ Diabetes saved to Supabase")
        except Exception as db_err:
            print(f"❌ Supabase INSERT ERROR: {db_err}")
        
        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': round(float(max(proba)) * 100, 2),
            'all_probabilities': {
                'Non-Diabetic': round(float(proba[0]) * 100, 2), 
                'Diabetic': round(float(proba[1]) * 100, 2)
            },
            'patient': {'name': data.get('name'), 'age': data.get('age')}
        })
    except Exception as e:
        print(f"❌ Diabetes error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict/breast-cancer', methods=['POST', 'OPTIONS'])
def predict_breast_cancer():
    if request.method == 'OPTIONS':
        return '', 200
    
    print("=== BREAST CANCER PREDICTION ===")
        
    if breast_cancer_model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        print(f"📊 Received data: {data}")
        
        features = np.array([[
            float(data.get('mean_radius', 14.0)),
            float(data.get('mean_texture', 19.0)),
            float(data.get('mean_perimeter', 92.0)),
            float(data.get('mean_area', 655.0)),
            float(data.get('mean_smoothness', 0.096))
        ]], dtype=np.float32)
        
        print(f"✅ Features shape: {features.shape}")
        
        prediction = breast_cancer_model.predict(features)
        proba = breast_cancer_model.predict_proba(features)[0]
        result = 'Malignant' if int(prediction[0]) == 1 else 'Benign'
        
        print(f"✅ Prediction: {result} ({max(proba)*100:.2f}%)")
        
        # Save to Supabase
        try:
            supabase.table('predictions').insert({
                'user_id': None,
                'disease_type': 'breast_cancer',
                'prediction_result': result,
                'confidence_score': float(max(proba)) * 100,
                'input_data': data
            }).execute()
            print(f"✅ Breast Cancer saved to Supabase")
        except Exception as db_err:
            print(f"❌ Supabase ERROR: {db_err}")
        
        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': round(float(max(proba)) * 100, 2),
            'all_probabilities': {
                'Benign': round(float(proba[0]) * 100, 2), 
                'Malignant': round(float(proba[1]) * 100, 2)
            },
            'patient': {'name': data.get('name'), 'age': data.get('age')}
        })
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict/alzheimers', methods=['POST', 'OPTIONS'])
def predict_alzheimers():
    if request.method == 'OPTIONS':
        return '', 200
        
    print("=== ALZHEIMERS PREDICTION ===")
    
    if alzheimers_model is None:
        return jsonify(success=False, error="Model not loaded"), 500
    
    name = request.form.get('name', 'Unknown')
    age = request.form.get('age', '0')
    
    if 'image' not in request.files:
        return jsonify(success=False, error="No image provided"), 400
    
    image_file = request.files['image']
    
    try:
        # ✅ VALIDATE IMAGE WITH GEMINI VISION
        is_valid, validation_message = validate_medical_image(image_file, "Brain MRI scan")
        
        if not is_valid:
            print(f"❌ VALIDATION FAILED: {validation_message}")
            return jsonify(success=False, error=validation_message), 400
        
        print("✅ Validation passed, proceeding with prediction...")
        
        image_array = preprocess_image(image_file, target_size=(224, 224))
        print(f"✅ Image shape: {image_array.shape}")
        
        prediction = alzheimers_model.predict(image_array, verbose=0)
        classes = ["Non-Demented", "Very Mild Dementia", "Mild Dementia", "Moderate Dementia"]
        predicted_class = classes[np.argmax(prediction[0])]
        confidence = round(float(np.max(prediction[0])) * 100, 2)
        
        print(f"✅ Prediction: {predicted_class} ({confidence}%)")
        
        # Save to Supabase
        try:
            supabase.table('predictions').insert({
                'user_id': None,
                'disease_type': 'alzheimer',
                'prediction_result': predicted_class,
                'confidence_score': confidence,
                'image_url': image_file.filename
            }).execute()
            print(f"✅ Alzheimer's saved to Supabase")
        except Exception as db_err:
            print(f"❌ Supabase INSERT ERROR: {db_err}")
        
        return jsonify(
            success=True,
            prediction=predicted_class,
            confidence=confidence,
            name=name,
            age=age,
            image_filename=image_file.filename,
            all_probabilities={classes[i]: round(float(prediction[0][i]) * 100, 2) for i in range(len(classes))}
        )
        
    except Exception as e:
        print(f"❌ Alzheimer's error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500


@app.route('/predict/brain-tumor', methods=['POST', 'OPTIONS'])
def predict_brain_tumor():
    if request.method == 'OPTIONS':
        return '', 200
    
    print("=== BRAIN TUMOR PREDICTION ===")
    
    if brain_tumor_model is None:
        return jsonify(success=False, error="Brain tumor model not loaded"), 500
    
    name = request.form.get('name', 'Unknown')
    age = request.form.get('age', '0')
    
    if 'image' not in request.files:
        return jsonify(success=False, error="No image provided"), 400
    
    image_file = request.files['image']
    
    try:
        # ✅ VALIDATE IMAGE WITH GEMINI VISION
        is_valid, validation_message = validate_medical_image(image_file, "Brain MRI scan")
        
        if not is_valid:
            print(f"❌ VALIDATION FAILED: {validation_message}")
            return jsonify(success=False, error=validation_message), 400
        
        print("✅ Validation passed, proceeding with prediction...")
        
        image_array = preprocess_image(image_file, target_size=(150, 150))
        print(f"✅ Image shape: {image_array.shape}")
        
        prediction = brain_tumor_model.predict(image_array, verbose=0)
        classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        predicted_class = classes[np.argmax(prediction)]
        confidence = round(float(np.max(prediction)) * 100, 2)
        
        print(f"✅ Prediction: {predicted_class} ({confidence}%)")
        
        # Save to Supabase
        try:
            supabase.table('predictions').insert({
                'user_id': None,
                'disease_type': 'brain_tumor',
                'prediction_result': predicted_class,
                'confidence_score': confidence,
                'image_url': image_file.filename
            }).execute()
            print(f"✅ Brain Tumor saved to Supabase")
        except Exception as db_err:
            print(f"❌ Supabase INSERT ERROR: {db_err}")
        
        return jsonify(
            success=True,
            prediction=predicted_class,
            confidence=confidence,
            name=name,
            age=age,
            image_filename=image_file.filename,
            all_probabilities={classes[i]: round(float(prediction[0][i]) * 100, 2) for i in range(len(classes))} 
        )
        
    except Exception as e:
        print(f"❌ Brain Tumor error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500


@app.route('/predict/pneumonia', methods=['POST', 'OPTIONS'])
def predict_pneumonia():
    if request.method == 'OPTIONS':
        return '', 200
        
    print("=== PNEUMONIA PREDICTION ===")
    
    if pneumonia_model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    name = request.form.get('name', 'Unknown')
    age = request.form.get('age', '0')
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'Image required'}), 400
    
    image_file = request.files['image']
    
    try:
        is_valid, validation_message = validate_medical_image(image_file, "Chest X-ray")
        
        if not is_valid:
            print(f"❌ VALIDATION FAILED: {validation_message}")
            return jsonify({'success': False, 'error': validation_message}), 400
        
        print("✅ Validation passed, proceeding with prediction...")
        
        image_array = preprocess_image(image_file, target_size=(224, 224))
        print(f"✅ Image shape: {image_array.shape}")
        
        prediction = pneumonia_model.predict(image_array, verbose=0)
        raw_value = float(prediction[0][0])
        print(f"🔍 Raw prediction: {raw_value}")
        
        # ✅ FLIPPED: High = Normal, Low = Pneumonia
        predicted_class = 'Normal' if prediction[0][0] > 0.5 else 'Pneumonia'
        confidence = round(max(raw_value, 1 - raw_value) * 100, 2)
        
        print(f"✅ FINAL: {predicted_class} ({confidence}%)")
        
        try:
            supabase.table('predictions').insert({
                'user_id': None,
                'disease_type': 'pneumonia',
                'prediction_result': predicted_class,
                'confidence_score': confidence,
                'image_url': image_file.filename
            }).execute()
            print(f"✅ Pneumonia saved to Supabase")
        except Exception as db_err:
            print(f"❌ Supabase INSERT ERROR: {db_err}")
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'name': name,
            'age': age,
            'image_filename': image_file.filename,
            'all_probabilities': {
                'Normal': round(raw_value * 100, 2),           # ✅ High = Normal
                'Pneumonia': round((1 - raw_value) * 100, 2)  # ✅ Low = Pneumonia
            }
        })
    except Exception as e:
        print(f"❌ Pneumonia error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/predict/covid', methods=['POST', 'OPTIONS'])
def predict_covid():
    if request.method == 'OPTIONS':
        return '', 200
        
    print("=== COVID PREDICTION ===")
    
    if covid_model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    name = request.form.get('name', 'Unknown')
    age = request.form.get('age', '0')
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'Image required'}), 400
    
    image_file = request.files['image']
    
    try:
        # ✅ VALIDATE IMAGE WITH GEMINI VISION
        is_valid, validation_message = validate_medical_image(image_file, "Chest X-ray")
        
        if not is_valid:
            print(f"❌ VALIDATION FAILED: {validation_message}")
            return jsonify({'success': False, 'error': validation_message}), 400
        
        print("✅ Validation passed, proceeding with prediction...")
        
        image_array = preprocess_image(image_file, target_size=(224, 224))
        print(f"✅ Image shape: {image_array.shape}")
        
        prediction = covid_model.predict(image_array, verbose=0)
        predicted_class = 'Normal' if prediction > 0.5 else 'COVID-19'
        confidence = round(float(max(prediction[0][0], 1 - prediction[0][0])) * 100, 2)
        
        print(f"✅ COVID prediction: {predicted_class} ({confidence}%)")
        
        # Save to Supabase
        try:
            supabase.table('predictions').insert({
                'user_id': None,
                'disease_type': 'covid',
                'prediction_result': predicted_class,
                'confidence_score': confidence,
                'image_url': image_file.filename
            }).execute()
            print(f"✅ COVID saved to Supabase")
        except Exception as db_err:
            print(f"❌ Supabase INSERT ERROR: {db_err}")
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'name': name,
            'age': age,
            'image_filename': image_file.filename,
            'all_probabilities': {
                'COVID-19 Negative': round(float(1 - prediction[0][0]) * 100, 2), 
                'COVID-19 Positive': round(float(prediction[0][0]) * 100, 2)
            }
        })
    except Exception as e:
        print(f"❌ COVID error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chatbot', methods=['POST', 'OPTIONS'])
def chatbot():
    if request.method == 'OPTIONS':
        return '', 200
    
    print("=== CHATBOT REQUEST ===")
    
    try:
        data = request.json
        user_message = data.get('message', '')
        
        print(f"User message: {user_message}")
        
        system_prompt = """You are SmartCureX AI Assistant 🤖 - a friendly healthcare companion!

SmartCureX Platform Features:
- Heart Disease screening ❤️
- Diabetes assessment 🩺
- Breast Cancer detection 🎗️
- Alzheimer's MRI analysis 🧠
- Brain Tumor detection 🧬
- Pneumonia X-ray screening 🫁
- COVID-19 detection 😷

How our Alzheimer's detection works:
- Users upload brain MRI scans
- AI analyzes images for dementia patterns
- Classifies into: Non-Demented, Very Mild, Mild, or Moderate Dementia
- Results include confidence scores

Your role:
- Answer health questions naturally and helpfully
- Explain our screening tools and how they work
- Use emojis to be friendly 😊💙✨
- Keep responses 2-4 sentences
- Be warm, informative, and supportive

Just chat naturally about health and our platform! 💙"""
        
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE", 
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
        }
        
        generation_config = {
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 300,
        }
        
        model = genai.GenerativeModel(
            'gemini-2.0-flash-lite',
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        full_prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
        response = model.generate_content(full_prompt)
        
        try:
            response_text = response.text
            print(f"✅ Response generated: {response_text[:100]}...")
            
        except ValueError as ve:
            print(f"⚠️ ValueError: {ve}")
            response_text = "I'm here to help! 😊 Ask me about our disease screening tools (Heart ❤️, Diabetes 🩺, Alzheimer's 🧠, and more) or any health questions. What would you like to know? 💙"
        
        # Save to Supabase chatbot_history (optional)
        try:
            supabase.table('chatbot_history').insert({
                'user_id': None,
                'message': user_message,
                'response': response_text
            }).execute()
            print(f"✅ Chatbot history saved to Supabase")
        except Exception as db_err:
            print(f"❌ Chatbot history save error: {db_err}")
        
        return jsonify({
            'success': True,
            'response': response_text,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': True,
            'response': "I'm here to help! 😊 Ask me about our disease screening tools (Heart ❤️, Diabetes 🩺, Alzheimer's 🧠, and more) or any health questions. What would you like to know? 💙",
            'timestamp': datetime.utcnow().isoformat()
        })


@app.route('/predictions', methods=['GET'])
@jwt_required()
def get_predictions():
    try:
        current_user = get_jwt_identity()
        
        # Get predictions from Supabase
        result = supabase.table('predictions')\
            .select('*')\
            .eq('user_id', current_user['id'])\
            .order('created_at', desc=True)\
            .limit(50)\
            .execute()
        
        predictions = [{
            'id': p['id'],
            'disease': p['disease_type'],
            'result': p['prediction_result'],
            'confidence': p['confidence_score'],
            'date': p['created_at']
        } for p in result.data]
        
        return jsonify({
            'success': True, 
            'predictions': predictions
        })
    except Exception as e:
        print(f"❌ Get predictions error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 SmartCureX Backend API with Supabase")
    print("="*50)
    print("📍 Server: http://localhost:5000")
    print("💾 Database: Supabase (PostgreSQL)")
    print("🔐 Auth: JWT + Supabase")
    print("🤖 AI: Google Gemini + Vision Validation")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)

@app.route('/auth/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    email = data.get('email')
    
    try:
        # Check if user exists
        result = supabase.table('users').select('*').eq('email', email).execute()
        
        if not result.data:
            # Don't reveal if email exists or not (security)
            return jsonify({'success': True, 'message': 'If email exists, reset link sent'}), 200
        
        user = result.data[0]
        
        # Generate reset token (valid for 1 hour)
        reset_token = create_access_token(
            identity={'id': user['id'], 'email': user['email'], 'reset': True},
            expires_delta=timedelta(hours=1)
        )
        
        # In production, send email with reset link
        # For now, just log it (you can add email service later)
        reset_link = f"http://localhost:5173/reset-password?token={reset_token}"
        print(f"\n🔐 PASSWORD RESET LINK: {reset_link}\n")
        
        return jsonify({
            'success': True,
            'message': 'Reset link sent',
            'reset_link': reset_link  # Remove this in production!
        }), 200
        
    except Exception as e:
        print(f"❌ Forgot password error: {e}")
        return jsonify({'success': False, 'error': 'Something went wrong'}), 500
