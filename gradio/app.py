import os
import sys
# Add the project root to the sys.path to resolve imports correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import psycopg2
import gradio as gr
import cv2
import json
import datetime
import time
import numpy as np
from PIL import Image

from engine.header import *

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
root_path = os.path.dirname(dir_path)
# root_path = dir_path

print('\t Face SDK Lite version')

g_activation_result = -1
MATCH_THRESHOLD = 0.82
css = """
/* Modern UI/UX Upgrade */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&display=swap');

body, .gradio-container {
    font-family: 'Outfit', sans-serif !important;
}

.example-image img {
    display: flex; justify-content: center; align-items: center;
    height: 300px; object-fit: contain;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}
.example-image img:hover {
    transform: scale(1.03);
}

.face-row {
    display: flex; justify-content: space-around; align-items: center; width: 100%;
}

.face-image img {
    height: 160px; object-fit: cover;
    border-radius: 50%;
    border: 4px solid #4facfe;
    box-shadow: 0 0 20px rgba(79, 172, 254, 0.3);
    transition: transform 0.3s ease;
}
.face-image img:hover {
    transform: scale(1.1) rotate(5deg);
}

.markdown-success-container {
    background: rgba(46, 204, 113, 0.1);
    backdrop-filter: blur(10px);
    padding: 20px; margin: 20px;
    border-radius: 12px;
    border: 1px solid rgba(46, 204, 113, 0.4);
    box-shadow: 0 4px 15px rgba(46, 204, 113, 0.2);
    text-align: center;
}
.markdown-success-container p { color: #2ecc71 !important; font-weight: bold; }

.markdown-fail-container {
    background: rgba(231, 76, 60, 0.1);
    backdrop-filter: blur(10px);
    padding: 20px; margin: 20px;
    border-radius: 12px;
    border: 1px solid rgba(231, 76, 60, 0.4);
    box-shadow: 0 4px 15px rgba(231, 76, 60, 0.2);
    text-align: center;
}
.markdown-fail-container p { color: #e74c3c !important; font-weight: bold; }

.markdown-attribute-container {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 12px; padding: 15px; margin: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    border: 1px solid rgba(0,0,0,0.05);
}

table { width: 100%; border-collapse: collapse; }
th { border-bottom: 2px solid rgba(0,0,0,0.1); padding-bottom: 10px; color: #4facfe; font-size: 1.1em;}
td { padding: 10px 0; border-bottom: 1px solid rgba(0,0,0,0.05); font-weight: 500;}

.block-background {
    background: var(--background-fill-secondary) !important;
    border-radius: 16px !important;
    border: 1px solid var(--border-color-primary) !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08) !important;
    padding: 20px;
}

.enrollment-group {
    background: var(--background-fill-primary) !important;
    padding: 25px !important;
    border-radius: 15px !important;
    border: 1px solid var(--border-color-primary) !important;
    margin-top: 20px !important;
}

button.primary {
    background: linear-gradient(45deg, #00f2fe 0%, #4facfe 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4) !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(79, 172, 254, 0.6) !important;
}

.user-cards-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 25px;
    padding: 20px;
}

.user-card {
    background: var(--background-fill-primary);
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    overflow: hidden;
    border: 1px solid var(--border-color-primary);
    transition: all 0.3s ease;
    display: flex;
    flex-direction: column;
}

.user-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 30px rgba(79, 172, 254, 0.2);
    border-color: #4facfe;
}

.user-card img {
    width: 100%;
    height: 250px;
    object-fit: cover;
    border-bottom: 2px solid rgba(0,0,0,0.05);
}

.user-info {
    padding: 20px;
    text-align: left;
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.user-info h4 {
    margin: 0;
    color: #4facfe;
    font-size: 1.4rem;
    font-weight: 700;
}

.user-info p {
    margin: 0;
    font-size: 0.95rem;
    color: var(--body-text-color);
    display: flex;
    justify-content: space-between;
}
.user-info p strong {
    color: var(--body-text-color-subdued);
}

.custom-admin-table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 1rem;
    background: var(--background-fill-primary);
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

.custom-admin-table th {
    background: linear-gradient(45deg, #00f2fe 0%, #4facfe 100%);
    color: white;
    padding: 15px;
    text-align: left;
    font-weight: 600;
}

.custom-admin-table td {
    padding: 15px;
    border-bottom: 1px solid var(--border-color-primary);
    vertical-align: middle;
}

.custom-admin-table tr:hover td {
    background: rgba(79, 172, 254, 0.05);
}

.table-wrapper {
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
}

.table-avatar {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #4facfe;
}

/* Mobile Responsiveness */
@media (max-width: 768px) {
    .face-row {
        flex-direction: column !important;
        gap: 20px;
    }
    .face-image img {
        height: 120px;
    }
    .gradio-container {
        padding: 5px !important;
    }
    .block-background {
        padding: 15px !important;
    }
    .login-card {
        background: var(--background-fill-secondary);
        border: 1px solid var(--border-color-primary);
        border-radius: 30px;
        padding: 50px;
        max-width: 480px;
        margin: 100px auto;
        box-shadow: 0 30px 60px rgba(0,0,0,0.3);
        animation: fadeIn 0.8s ease-out;
    }
    .login-card h1 {
        background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 5px;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
}
"""


def activate_sdk():
    ret = init_sdk()

    if ret == 0:
        print("Successfully init SDK!")
    else:
        print(f"Falied to init SDK, Error code {ret}")

    return ret

def convert_fun(input_str):
    # Remove line breaks and extra whitespaces
    return ' '.join(input_str.split())

def analyze_face_clicked(frame):    
    global g_activation_result
    if g_activation_result != 0:
        gr.Warning("SDK Activation Failed!")
        return None, None, None

    try:
        image = open(frame, 'rb')
    except:
        raise gr.Error("Please select images file!")

    image_mat = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    start_time = time.time()
    ret, face_result = detect_face(image_mat, 1, ENGINE_MODE.M_IDENTIFY.value)
    end_time = time.time()
    process_time = (end_time - start_time) * 1000

    if ret <= 0:
        if ret == ENGINE_CODE.E_NO_FILE.value:
            result = "Can't open file"
        elif ret == ENGINE_CODE.E_NO_FACE.value:
            result = "NO FACE"
        else:
            result = "ENGINE ERROR"
        gr.Warning(result)
        return None, None, None

    attribute = face_result[0]
    face_crop, one_line_attribute = None, ""
    try:
        image = Image.open(frame)

        face = Image.new('RGBA',(150, 150), (80,80,80,0))
        
        x1 = attribute.x1
        y1 = attribute.y1
        x2 = attribute.x2
        y2 = attribute.y2

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 >= image.width:
            x2 = image.width - 1
        if y2 >= image.height:
            y2 = image.height - 1

        face_crop = image.crop((x1, y1, x2, y2))
        face_image_ratio = face_crop.width / float(face_crop.height)
        resized_w = int(face_image_ratio * 150)
        resized_h = 150

        face_crop = face_crop.resize((int(resized_w), int(resized_h)))
            
    
        if attribute.liveness == LIVENESS_CODE.L_SPOOF.value:
            liveness = "SPOOF"
        elif attribute.liveness == LIVENESS_CODE.L_REAL.value:
            liveness = "REAL"
        elif attribute.liveness == LIVENESS_CODE.L_TOO_SMALL_FACE.value:    
            liveness = "TOO SMALL FACE"
        elif attribute.liveness == LIVENESS_CODE.L_TOO_LARGE_FACE.value:
            liveness = "TOO LARGE FACE"
        elif attribute.liveness == LIVENESS_CODE.L_NO_FACE.value:
            liveness = "NO FACE"
        elif attribute.liveness == LIVENESS_CODE.L_LIVENESS_CHECK_FAILED.value:
            liveness = "Liveness Check Failed"
        else:    
            liveness = "ERROR"

        if liveness == 'REAL':
            liveness_result = f"""<br/><div class="markdown-success-container"><p style="text-align: center; font-size: 20px; color: green;">Liveness Check:  REAL<br/>Score: {attribute.liveness}</p></div>"""
        else:
            liveness_result = f"""<br/><div class="markdown-fail-container"><p style="text-align: center; font-size: 20px; color: red;">Liveness Check:  {liveness}<br/>Score: {attribute.liveness}</p></div>"""

        gender = "MALE" if attribute.gender == 0 else "FEMALE"
        age = attribute.age
        wear_glass = "NO" if attribute.glass == 0 else "YES"
        mask = "YES" if attribute.mask == 1 else "NO"
        

        attribute = f"""
        <br/>
        <div class="markdown-attribute-container">
        <table>
        <tr>
            <th style="text-align: center;">Attribute</th>
            <th style="text-align: center;">Result</th>
        </tr>
        <tr>
            <td>Gender</td>
            <td>{gender}</td>
        </tr>
        <tr>
            <td>Age</td>
            <td>{int(age)}</td>
        </tr>
        <tr>
            <td>Mask</td>
            <td>{mask}</td>
        </tr>
        <tr>
            <td>Glass</td>
            <td>{wear_glass}</td>
        </tr>
        </table>
        </div>
        """
        one_line_attribute = convert_fun(attribute)
    except:
        pass
    
    return face_crop, liveness_result, one_line_attribute

def compare_face_clicked(frame1, frame2, threshold):
    global g_activation_result
    if g_activation_result != 0:
        gr.Warning("SDK Activation Failed!")
        return None, None, None, None, None, None, None

    try:
        image1 = open(frame1, 'rb')
        image2 = open(frame2, 'rb')
    except:
        raise gr.Error("Please select images files!")

    image_mat1 = cv2.imdecode(np.frombuffer(image1.read(), np.uint8), cv2.IMREAD_COLOR)
    image_mat2 = cv2.imdecode(np.frombuffer(image2.read(), np.uint8), cv2.IMREAD_COLOR)
    start_time = time.time()
    ret1, face_result1 = detect_face(image_mat1, 1, ENGINE_MODE.M_ENROLL.value)
    if ret1 <= 0:
        if ret1 == ENGINE_CODE.E_NO_FILE.value:
            gr.Warning("Can't open file1")
        elif ret1 == ENGINE_CODE.E_NO_FACE.value:
            gr.Warning("NO FACE in image1")
        else:
            gr.Warning("ENGINE ERROR")

        return None, None, None, None, None, None, None

    ret2, face_result2 = detect_face(image_mat2, 1, ENGINE_MODE.M_IDENTIFY.value)
    if ret2 <= 0:
        if ret2 == ENGINE_CODE.E_NO_FILE.value:
            gr.Warning("Can't open file2")
        elif ret2 == ENGINE_CODE.E_NO_FACE.value:
            gr.Warning("NO FACE in image2")
        else:
            gr.Warning("ENGINE ERROR")

        return None, None, None, None, None, None, None


    similarity = get_similarity(face_result1[0].feature, face_result2[0].feature)
    end_time = time.time()
    process_time = (end_time - start_time) * 1000

    try:
        image1 = Image.open(frame1)
        image2 = Image.open(frame2)
        images = [image1, image2]

        face1 = Image.new('RGBA',(150, 150), (80,80,80,0))
        face2 = Image.new('RGBA',(150, 150), (80,80,80,0))
        faces = [face1, face2]
        
        face_results = [face_result1, face_result2]
        face_bboxes_result = []
        for i, face_result in enumerate(face_results):
            x1 = face_result[0].x1
            y1 = face_result[0].y1
            x2 = face_result[0].x2
            y2 = face_result[0].y2
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 >= images[i].width:
                x2 = images[i].width - 1
            if y2 >= images[i].height:
                y2 = images[i].height - 1

            face_bbox_str = f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}"
            face_bboxes_result.append(face_bbox_str)

            faces[i] = images[i].crop((x1, y1, x2, y2))
            face_image_ratio = faces[i].width / float(faces[i].height)
            resized_w = int(face_image_ratio * 150)
            resized_h = 150

            faces[i] = faces[i].resize((int(resized_w), int(resized_h)))
    except:
        pass
    
    matching_result = Image.open(os.path.join(dir_path, "icons/blank.png"))
    similarity_score = ""
    if face1 is not None and face2 is not None:
        if similarity is not None:
            str_score = str("{:.4f}".format(similarity))
        
            if similarity >= float(threshold):
                matching_result = Image.open(os.path.join(dir_path, "icons/same.png"))
                similarity_score = f"""<br/><div class="markdown-success-container"><p style="text-align: center; font-size: 20px; color: green;">Similarity score: {str_score}</p></div>"""
            else:
                matching_result = Image.open(os.path.join(dir_path, "icons/different.png"))
                similarity_score = f"""<br/><div class="markdown-fail-container"><p style="text-align: center; font-size: 20px; color: red;">Similarity score: {str_score}</p></div>"""
    
    return faces[0], faces[1], matching_result, similarity_score, face_bboxes_result[0], face_bboxes_result[1], str(process_time)    

DB_URL = "postgresql://manoj:cpzFOpgniFOUd7qwZ8ZLVUCVHw4jQFW9@dpg-d83vgm3tqb8s73ere75g-a.oregon-postgres.render.com/attendence_j26z"
USERS_CACHE = {}

def init_db():
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                roll_no VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100),
                year VARCHAR(20),
                section VARCHAR(20),
                branch VARCHAR(50),
                feature JSONB
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id SERIAL PRIMARY KEY,
                roll_no VARCHAR(50) REFERENCES users(roll_no),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            ALTER TABLE users ADD COLUMN IF NOT EXISTS image_base64 TEXT;
        ''')
        conn.commit()
        cursor.close()
        conn.close()
        print("Database initialized successfully.")
    except Exception as e:
        print("Database initialization error:", e)

def load_db():
    global USERS_CACHE
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT roll_no, name, year, section, branch, feature FROM users")
        rows = cursor.fetchall()
        USERS_CACHE.clear()
        for row in rows:
            USERS_CACHE[row[0]] = {
                "roll_no": row[0],
                "name": row[1],
                "year": row[2],
                "section": row[3],
                "branch": row[4],
                "feature": row[5]
            }
        cursor.close()
        conn.close()
    except Exception as e:
        print("DB Load Error:", e)
    return USERS_CACHE

def save_user_to_db(user_data):
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (roll_no, name, year, section, branch, feature, image_base64) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (roll_no) DO UPDATE SET name=EXCLUDED.name, year=EXCLUDED.year, section=EXCLUDED.section, branch=EXCLUDED.branch, feature=EXCLUDED.feature, image_base64=COALESCE(EXCLUDED.image_base64, users.image_base64)",
            (user_data['roll_no'], user_data['name'], user_data['year'], user_data['section'], user_data['branch'], json.dumps(user_data['feature']), user_data.get('image_base64', ''))
        )
        conn.commit()
        cursor.close()
        conn.close()
        load_db() # Refresh cache
    except Exception as e:
        print("DB Save Error:", e)
        raise e

def delete_user_from_db(roll_no):
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM attendance_logs WHERE roll_no = %s", (roll_no,))
        cursor.execute("DELETE FROM users WHERE roll_no = %s", (roll_no,))
        conn.commit()
        cursor.close()
        conn.close()
        load_db() 
        return f"User {roll_no} deleted successfully."
    except Exception as e:
        return f"Error: {str(e)}"

def get_user_details(roll_no):
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT name, year, section, branch FROM users WHERE roll_no = %s", (roll_no,))
        res = cursor.fetchone()
        cursor.close()
        conn.close()
        if res:
            return res[0], res[1], res[2], res[3]
        return "", "", "", ""
    except:
        return "", "", "", ""

def log_attendance(roll_no):
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO attendance_logs (roll_no) VALUES (%s)", (roll_no,))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("Attendance Log Error:", e)

def process_attendance(frame, is_enrolling, session_start_time, marked_roll_nos):
    if session_start_time > 0 and time.time() - session_start_time > 120:
        return (
            gr.update(value="<h3 style='text-align:center; padding: 20px;'>Session Timed Out. Please Login Again.</h3>"),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(),
            False,
            gr.update(visible=True),
            gr.update(visible=False),
            []
        )

    if is_enrolling:
        return gr.update(), gr.update(), gr.update(), gr.update(), True, gr.update(), gr.update(), marked_roll_nos

    global g_activation_result
    if g_activation_result != 0:
        return gr.update(visible=True, value="<div class='markdown-fail-container'><p>SDK Activation Failed!</p></div>"), gr.update(visible=False), gr.update(visible=False), gr.update(), False, gr.update(), gr.update(), marked_roll_nos
    
    if frame is None:
        return gr.update(), gr.update(), gr.update(), gr.update(), False, gr.update(), gr.update(), marked_roll_nos

    try:
        if isinstance(frame, str):
            image = open(frame, 'rb')
            image_mat = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        else:
            image_mat = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    except:
        return gr.update(), gr.update(), gr.update(), gr.update(), False, gr.update(), gr.update(), marked_roll_nos
        
    ret, face_results = detect_face(image_mat, 10, ENGINE_MODE.M_IDENTIFY.value)
    
    if ret <= 0 or not face_results:
        return gr.update(), gr.update(), gr.update(), gr.update(), False, gr.update(), gr.update(), marked_roll_nos
        
    db = load_db()
    new_marked = list(marked_roll_nos)
    found_unknown = False
    unknown_face_crop = None
    
    for face in face_results:
        query_feature = face.feature
        if isinstance(query_feature, np.ndarray):
            query_feature = query_feature.tolist()
            
        best_match = None
        best_score = 0
        
        for roll_no, user_data in db.items():
            db_feat = np.array(user_data["feature"], dtype=np.float32)
            q_feat = np.array(query_feature, dtype=np.float32)
            score = get_similarity(db_feat, q_feat)
            if score > best_score:
                best_score = score
                best_match = user_data
        
        if best_match and best_score > 0.85:
            roll = best_match['roll_no']
            if roll not in new_marked:
                log_attendance(roll)
                gr.Info(f"✨ Attendance Marked: {best_match['name']} ({roll})")
                new_marked.append(roll)
                
                # Auto-update image base64 if available
                try:
                    x1, y1, x2, y2 = face.x1, face.y1, face.x2, face.y2
                    h, w = image_mat.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w-1, x2), min(h-1, y2)
                    face_crop = image_mat[y1:y2, x1:x2]
                    import base64
                    _, buffer = cv2.imencode('.jpg', face_crop)
                    img_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    conn = psycopg2.connect(DB_URL)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE users SET image_base64 = %s WHERE roll_no = %s", (img_b64, roll))
                    conn.commit()
                    cursor.close()
                    conn.close()
                except:
                    pass
        else:
            found_unknown = True
            try:
                x1, y1, x2, y2 = face.x1, face.y1, face.x2, face.y2
                h, w = image_mat.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                unknown_face_crop = image_mat[y1:y2, x1:x2]
            except:
                pass
            break # Stop and enroll unknown user

    if found_unknown:
        msg = "<div class='markdown-attribute-container'><h3 style='color:#e74c3c; text-align:center;'>New Face Detected. Please Enroll.</h3></div>"
        return gr.update(visible=True, value=msg), gr.update(visible=True), gr.update(visible=True), unknown_face_crop, True, gr.update(), gr.update(), new_marked
    else:
        return gr.update(value="<h3 style='text-align:center; padding: 20px;'>Live Scanning Active...</h3>"), gr.update(visible=False), gr.update(visible=False), gr.update(), False, gr.update(), gr.update(), new_marked


def enroll_new_user(frame, roll_no, name, year, section, branch):
    if not all([frame is not None, roll_no, name, year, section, branch]):
        return gr.update(visible=True, value="<div class='markdown-fail-container'><p>Please fill all fields and wait for camera!</p></div>"), gr.update(visible=True), gr.update(visible=True), True
        
    try:
        if isinstance(frame, str):
            image = open(frame, 'rb')
            image_mat = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        else:
            image_mat = frame
            
        ret, face_result = detect_face(image_mat, 1, ENGINE_MODE.M_ENROLL.value)
        if ret <= 0 or not face_result:
            return gr.update(visible=True, value="<div class='markdown-fail-container'><p>NO FACE DETECTED FOR ENROLLMENT</p></div>"), gr.update(visible=True), gr.update(visible=True), True
            
        feature = face_result[0].feature
        if isinstance(feature, np.ndarray):
            feature = feature.tolist()
            
        import base64
        x1, y1, x2, y2 = face_result[0].x1, face_result[0].y1, face_result[0].x2, face_result[0].y2
        h, w = image_mat.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        face_crop = image_mat[y1:y2, x1:x2]
        _, buffer = cv2.imencode('.jpg', face_crop)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
            
        user_data = {
            "roll_no": roll_no,
            "name": name,
            "year": year,
            "section": section,
            "branch": branch,
            "feature": feature,
            "image_base64": img_b64
        }
        save_user_to_db(user_data)
        
        msg = f"<div class='markdown-success-container'><p style='font-size: 20px;'>Enrollment Successful for {name}!</p></div>"
        return gr.update(visible=True, value=msg), gr.update(visible=False), gr.update(visible=False), False
        
    except Exception as e:
        return gr.update(visible=True, value=f"<div class='markdown-fail-container'><p>Error: {str(e)}</p></div>"), gr.update(visible=True), gr.update(visible=True), True

def fetch_users_for_admin():
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT roll_no, name, year, section, branch, image_base64 FROM users ORDER BY roll_no ASC")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        html = "<div class='table-wrapper'><table class='custom-admin-table'>"
        html += "<thead><tr><th>Photo</th><th>Roll No</th><th>Name</th><th>Branch</th><th>Year/Sec</th></tr></thead><tbody>"
        for r in rows:
            roll, name, year, sec, branch, img_b64 = r
            img_src = f"data:image/jpeg;base64,{img_b64}" if img_b64 else "https://ui-avatars.com/api/?name=" + name.replace(" ", "+")
            html += f"<tr><td><img src='{img_src}' class='table-avatar'/></td><td>{roll}</td><td>{name}</td><td>{branch}</td><td>{year} / {sec}</td></tr>"
        html += "</tbody></table></div>"
        return html
    except Exception as e:
        print("Error fetching users:", e)
        return "<p>Error loading users</p>"

def fetch_logs_for_admin():
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.id, a.roll_no, u.name, u.branch, TO_CHAR(a.timestamp + interval '5 hours 30 minutes', 'DD Mon YYYY, HH12:MI AM'), u.image_base64
            FROM attendance_logs a
            LEFT JOIN users u ON a.roll_no = u.roll_no
            ORDER BY a.timestamp DESC
            LIMIT 100
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        html = "<div class='table-wrapper'><table class='custom-admin-table'>"
        html += "<thead><tr><th>Log ID</th><th>Photo</th><th>Roll No</th><th>Name</th><th>Branch</th><th>Timestamp</th></tr></thead><tbody>"
        for r in rows:
            log_id, roll, name, branch, ts, img_b64 = r
            img_src = f"data:image/jpeg;base64,{img_b64}" if img_b64 else "https://ui-avatars.com/api/?name=" + str(name).replace(" ", "+")
            html += f"<tr><td>{log_id}</td><td><img src='{img_src}' class='table-avatar'/></td><td>{roll}</td><td>{name}</td><td>{branch}</td><td>{ts}</td></tr>"
        html += "</tbody></table></div>"
        return html
    except Exception as e:
        print("Error fetching logs:", e)
        return "<p>Error fetching logs</p>"

def launch_demo(activate_result):
    theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate",
        spacing_size="md",
        radius_size="lg",
    ).set(
        body_background_fill="*background_fill_primary",
        button_primary_background_fill="linear-gradient(45deg, #00f2fe 0%, #4facfe 100%)",
        button_primary_background_fill_hover="linear-gradient(45deg, #4facfe 0%, #00f2fe 100%)",
    )
    with gr.Blocks(theme=theme) as demo:
        gr.Markdown(
            f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px; background: var(--background-fill-secondary); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); border: 1px solid var(--border-color-primary);">
                <h1 style="font-size: 48px; font-weight: 800; background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; text-align: center; letter-spacing: -1px;">✨ AI Vision Studio ✨</h1>
                <p style="font-size: 20px; color: var(--body-text-color-subdued); margin-top: 15px; text-align: center; font-weight: 400; max-width: 600px;">Next-generation facial verification, liveness detection, and smart attendance powered by deep learning.</p>
            </div>
            """
        )

        if activate_result == 0:
            gr.Markdown("""<div style='text-align: center; padding: 10px; background: rgba(46, 204, 113, 0.1); border-radius: 10px; border: 1px solid rgba(46, 204, 113, 0.3); margin-bottom: 20px;'><p style="margin: 0; font-size: 18px; color: #2ecc71; font-weight: 600;">✅ Face SDK Activated Successfully</p></div>""")         
        else:
            gr.Markdown("""<div style='text-align: center; padding: 10px; background: rgba(231, 76, 60, 0.1); border-radius: 10px; border: 1px solid rgba(231, 76, 60, 0.3); margin-bottom: 20px;'><p style="margin: 0; font-size: 18px; color: #e74c3c; font-weight: 600;">❌ SDK Activation Failed</p></div>""") 
                    
        # --- GATEKEEPER LOGIN ---
        with gr.Column(visible=True) as gatekeeper_group:
            with gr.Column(elem_classes="login-card"):
                gr.Markdown(
                    """
                    <div style='text-align:center; margin-bottom: 30px;'>
                        <div style='font-size: 60px; margin-bottom: 10px;'>🔒</div>
                        <h1>Secure Portal</h1>
                        <p style='color:var(--body-text-color-subdued); font-size: 1.1rem;'>Authorized Access Only</p>
                    </div>
                    """
                )
                gate_user = gr.Textbox(label="Username", placeholder="e.g. NARALA or MANOJ", interactive=True)
                gate_pass = gr.Textbox(label="Password", type="password", placeholder="••••••••", interactive=True)
                gate_status = gr.Markdown("")
                gate_login_btn = gr.Button("🔓 Authenticate & Enter", variant="primary", size="lg")

        # --- SHARED STATE ---
        session_start_time = gr.State(0)
        marked_roll_nos = gr.State([])

        # --- USER PORTAL ---
        with gr.Column(visible=False) as user_portal:
            with gr.Row(elem_classes="block-background", variant="compact"):
                gr.Markdown("<h2 style='margin:0; color:#4facfe; padding: 10px;'>👥 Detection Portal</h2>")
                logout_user_btn = gr.Button("🚪 Exit Portal", size="sm", variant="secondary", scale=0)
            
            with gr.Tabs():
                with gr.Tab("Smart Attendance"):
                    with gr.Group(visible=True) as start_session_group:
                        gr.Markdown("<div style='padding: 40px; text-align:center;'><h2 style='color:#4facfe;'>📸 Attendance Scanner Ready</h2><p>Click below to start a 2-minute automated attendance session.</p></div>")
                        with gr.Row():
                            with gr.Column(scale=1): pass
                            with gr.Column(scale=1):
                                start_session_btn = gr.Button("▶️ Start 2-Min Session", variant="primary", size="lg")
                            with gr.Column(scale=1): pass
                    
                    with gr.Group(visible=False) as attendance_main_group:
                        with gr.Row():
                            with gr.Column(scale=1):
                                attendance_input = gr.Image(label="Live Feed", sources=["webcam"], streaming=True, type='numpy', height=400)
                                is_enrolling = gr.State(False)
                            with gr.Column(scale=1, elem_classes="block-background"):
                                attendance_status = gr.Markdown("<h3 style='text-align:center; padding: 20px;'>Scanning...</h3>")
                                with gr.Group(visible=False, elem_classes="enrollment-group") as enroll_group:
                                    gr.Markdown("<h2 style='text-align:center; color:#4facfe;'>✨ Registration</h2>")
                                    with gr.Row():
                                        reg_roll = gr.Textbox(label="Roll No")
                                        reg_name = gr.Textbox(label="Full Name")
                                    with gr.Row():
                                        reg_year = gr.Textbox(label="Year")
                                        reg_sec = gr.Textbox(label="Section")
                                        reg_branch = gr.Textbox(label="Branch")
                                    enroll_state_image = gr.State()
                                    enroll_btn = gr.Button("🚀 Enroll Face", variant="primary", size="lg")
                                with gr.Group(visible=False) as back_to_scan_group:
                                    back_btn = gr.Button("🔙 Cancel", size="lg")

                with gr.Tab("Face Recognition"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    compare_face_input1 = gr.Image(label="Image1", type='filepath', elem_classes="example-image")
                                    gr.Examples([os.path.join(root_path,'examples/1.jpg'), os.path.join(root_path,'examples/2.jpg')], inputs=compare_face_input1)
                                with gr.Column(scale=1):
                                    compare_face_input2 = gr.Image(label="Image2", type='filepath', elem_classes="example-image")
                                    gr.Examples([os.path.join(root_path,'examples/5.jpg'), os.path.join(root_path,'examples/6.jpg')], inputs=compare_face_input2)
                        with gr.Column(scale=1, min_width=400, elem_classes="block-background"):     
                            txt_threshold = gr.Textbox(f"{MATCH_THRESHOLD}", label="Threshold")
                            compare_face_button = gr.Button("Compare Face", variant="primary", size="lg")
                            with gr.Row(elem_classes="face-row"):
                                face_output1 = gr.Image(label="Face 1", scale=0, elem_classes="face-image")
                                compare_result = gr.Image(min_width=30, scale=0, show_label=False)
                                face_output2 = gr.Image(label="Face 2", scale=0, elem_classes="face-image")
                            similarity_markdown = gr.Markdown("")
                            txt_speed = gr.Textbox(f"", visible=False)
                            with gr.Row():
                                with gr.Column(): txt_bbox1 = gr.Textbox(f"", label="Rect 1")
                                with gr.Column(): txt_bbox2 = gr.Textbox(f"", label="Rect 2")

                with gr.Tab("Face Liveness, Analysis"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            face_input = gr.Image(label="Image", type='filepath', elem_classes="example-image")
                            gr.Examples([os.path.join(root_path,'examples/att_1.jpg')], inputs=face_input)
                        with gr.Column(scale=1, elem_classes="block-background"):     
                            analyze_face_button = gr.Button("Analyze Face", variant="primary", size="lg")
                            with gr.Row(elem_classes="face-row"):
                                face_output = gr.Image(label="Face", scale=0, elem_classes="face-image")
                            liveness_result = gr.Markdown("")
                            attribute_result = gr.Markdown("")

        # --- ADMIN PORTAL ---
        with gr.Column(visible=False) as admin_portal:
            with gr.Row(elem_classes="block-background", variant="compact"):
                gr.Markdown("<h2 style='margin:0; color:#4facfe; padding: 10px;'>🛡️ Administration Portal</h2>")
                logout_admin_btn = gr.Button("🚪 Exit Portal", size="sm", variant="secondary", scale=0)
            
            with gr.Tabs():
                with gr.Tab("👥 Registered Users"):
                    users_table = gr.HTML(elem_classes="block-background")
                with gr.Tab("📜 Attendance Logs"):
                    logs_table = gr.HTML(elem_classes="block-background")
                with gr.Tab("⚙️ Manage Users"):
                    with gr.Row():
                        with gr.Column(scale=1, elem_classes="block-background"):
                            gr.Markdown("### ➕ Add/Update User (Upload)")
                            manual_img = gr.Image(label="User Photo", type="filepath")
                            manual_roll = gr.Textbox(label="Roll No")
                            manual_name = gr.Textbox(label="Full Name")
                            with gr.Row():
                                manual_year = gr.Textbox(label="Year")
                                manual_sec = gr.Textbox(label="Sec")
                                manual_branch = gr.Textbox(label="Branch")
                            manual_btn = gr.Button("🚀 Save User", variant="primary")
                            manual_status = gr.Markdown("")
                        
                        with gr.Column(scale=1, elem_classes="block-background"):
                            gr.Markdown("### 🗑️ Delete Existing User")
                            user_to_manage = gr.Dropdown(label="Select User Roll No", choices=[], interactive=True)
                            refresh_manage_btn = gr.Button("🔄 Refresh List", size="sm")
                            with gr.Group(visible=False) as edit_form_group:
                                edit_name = gr.Textbox(label="Name")
                                delete_btn = gr.Button("❌ Permanent Delete", variant="secondary")
                            manage_status = gr.Markdown("")
            with gr.Row():
                refresh_btn = gr.Button("🔄 Force Refresh Database", variant="primary", size="lg")

            # Logic Handlers
            def gatekeeper_authenticate(user, pwd):
                if user == "MANOJ" and pwd == "manoj@007":
                    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), "", "⌛ Loading Users...", "⌛ Loading Logs..."
                elif user == "NARALA" and pwd == "manoj@007":
                    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), "", gr.update(), gr.update()
                return gr.update(), gr.update(), gr.update(), "<p style='color:red; text-align:center;'>❌ Invalid Credentials</p>", gr.update(), gr.update()

            def logout():
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), ""

            def start_detection_session():
                return gr.update(visible=False), gr.update(visible=True), time.time(), []

            # Global Events
            def load_admin_data_if_needed(user):
                if user == "MANOJ":
                    return fetch_users_for_admin(), fetch_logs_for_admin()
                return gr.update(), gr.update()

            gate_login_btn.click(
                gatekeeper_authenticate, 
                inputs=[gate_user, gate_pass], 
                outputs=[gatekeeper_group, user_portal, admin_portal, gate_status, users_table, logs_table]
            ).then(
                load_admin_data_if_needed,
                inputs=[gate_user],
                outputs=[users_table, logs_table]
            )
            logout_user_btn.click(logout, outputs=[gatekeeper_group, user_portal, admin_portal, gate_status])
            logout_admin_btn.click(logout, outputs=[gatekeeper_group, user_portal, admin_portal, gate_status])
            
            # Smart Attendance Events
            start_session_btn.click(start_detection_session, inputs=[], outputs=[start_session_group, attendance_main_group, session_start_time, marked_roll_nos])
            
            attendance_input.stream(process_attendance, inputs=[attendance_input, is_enrolling, session_start_time, marked_roll_nos], 
                                    outputs=[attendance_status, enroll_group, back_to_scan_group, enroll_state_image, is_enrolling, start_session_group, attendance_main_group, marked_roll_nos])
            
            enroll_btn.click(enroll_new_user, inputs=[enroll_state_image, reg_roll, reg_name, reg_year, reg_sec, reg_branch], outputs=[attendance_status, enroll_group, back_to_scan_group, is_enrolling])
            back_btn.click(lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(value="<h3 style='text-align:center; padding: 20px;'>Scanning...</h3>"), False), 
                           inputs=[], outputs=[enroll_group, back_to_scan_group, attendance_status, is_enrolling])
            
            # Recognition & Analysis Events
            compare_face_button.click(compare_face_clicked, inputs=[compare_face_input1, compare_face_input2, txt_threshold], 
                                      outputs=[face_output1, face_output2, compare_result, similarity_markdown, txt_bbox1, txt_bbox2, txt_speed])
            analyze_face_button.click(analyze_face_clicked, inputs=face_input, outputs=[face_output, liveness_result, attribute_result])
            # Management Events
            def refresh_manage_dropdown():
                choices = []
                try:
                    conn = psycopg2.connect(DB_URL)
                    cursor = conn.cursor()
                    cursor.execute("SELECT roll_no FROM users ORDER BY roll_no ASC")
                    choices = [r[0] for r in cursor.fetchall()]
                    cursor.close()
                    conn.close()
                except: pass
                return gr.update(choices=choices)

            def handle_manual_reg(img, roll, name, year, sec, branch):
                res = enroll_new_user(img, roll, name, year, sec, branch)
                msg = res[0] if isinstance(res[0], str) else res[0].get('value', "Processed")
                return msg, refresh_manage_dropdown()

            def handle_delete(roll):
                if not roll: return "Select a user first", gr.update(visible=False), gr.update()
                msg = delete_user_from_db(roll)
                return msg, gr.update(visible=False), refresh_manage_dropdown()

            manual_btn.click(handle_manual_reg, inputs=[manual_img, manual_roll, manual_name, manual_year, manual_sec, manual_branch], outputs=[manual_status, user_to_manage])
            refresh_manage_btn.click(refresh_manage_dropdown, outputs=[user_to_manage])
            user_to_manage.change(lambda r: (gr.update(visible=True), get_user_details(r)[0]) if r else (gr.update(visible=False), ""), inputs=[user_to_manage], outputs=[edit_form_group, edit_name])
            delete_btn.click(handle_delete, inputs=[user_to_manage], outputs=[manage_status, edit_form_group, user_to_manage])

            refresh_btn.click(lambda: (fetch_users_for_admin(), fetch_logs_for_admin(), refresh_manage_dropdown()), inputs=[], outputs=[users_table, logs_table, user_to_manage])

        import os
        port = int(os.environ.get("PORT", 7860))
        demo.queue().launch(server_name="0.0.0.0", server_port=port, css=css)


if __name__ == '__main__':
    init_db()
    load_db()
    g_activation_result = activate_sdk()
    launch_demo(g_activation_result)
