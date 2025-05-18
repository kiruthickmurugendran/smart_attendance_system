import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

from fastapi import FastAPI, HTTPException, Depends, Form, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import cv2
import numpy as np
from numpy.linalg import norm
from keras.applications.resnet50 import ResNet50, preprocess_input
from datetime import date
import io
import matplotlib.pyplot as plt
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

DATABASE_URL = "postgresql://kiruthick:kiru%409555@localhost:5432/project"

def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

class Student(BaseModel):
    roll_number: str
    name: str
    department: str

class AbsentRequest(BaseModel):
    roll_number: str
    department: str
    reason: str

class AttendanceRecord(BaseModel):
    roll_number: str
    present: bool = True
    attendance_date: date = date.today()

FACE_DETECTION_SCALE = 1.1
FACE_DETECTION_NEIGHBORS = 7
MIN_FACE_SIZE = 150
MATCH_THRESHOLD = 25.0

def get_embedding(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_array = np.expand_dims(np.asarray(face_img, dtype=np.float32), axis=0)
    face_array = preprocess_input(face_array)
    embedding = resnet_model.predict(face_array, verbose=0)[0]
    return embedding / norm(embedding)

def find_best_match(embedding, students):
    best_match = None
    min_distance = float('inf')
    for roll, name, db_embedding in students:
        dist = np.linalg.norm(embedding - db_embedding)
        if dist < min_distance:
            min_distance = dist
            best_match = (roll, name)
    if min_distance < MATCH_THRESHOLD:
        return best_match
    return None, None

@app.get("/api/students", response_model=List[Student])
def get_students(conn = Depends(get_db)):
    with conn.cursor() as cur:
        cur.execute("SELECT roll_number, name, department FROM students")
        return cur.fetchall()

@app.post("/api/students")
def register_student(student: Student, conn = Depends(get_db)):
    with conn.cursor() as cur:
        try:
            cur.execute("INSERT INTO students (roll_number, name, department) VALUES (%s, %s, %s)",
                        (student.roll_number, student.name, student.department))
            conn.commit()
            return {"message": "Student registered successfully"}
        except psycopg2.IntegrityError:
            conn.rollback()
            raise HTTPException(status_code=400, detail="Roll number already exists")

@app.post("/api/attendance/mark")
def mark_attendance(record: AttendanceRecord, conn = Depends(get_db)):
    with conn.cursor() as cur:
        try:
            cur.execute("""INSERT INTO attendance (roll_number, present, date)
                           VALUES (%s, %s, %s)
                           ON CONFLICT (roll_number, date) DO UPDATE
                           SET present = EXCLUDED.present""",
                        (record.roll_number, record.present, record.attendance_date))
            conn.commit()
            return {"message": "Attendance marked successfully"}
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/absent/request")
def submit_absent_request(request: AbsentRequest, conn = Depends(get_db)):
    with conn.cursor() as cur:
        try:
            cur.execute("INSERT INTO absent_requests (roll_number, department, reason, date) VALUES (%s, %s, %s, %s)",
                        (request.roll_number, request.department, request.reason, date.today()))
            conn.commit()
            return {"message": "Absent request submitted successfully"}
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/attendance", response_model=List[dict])
def get_attendance(date: Optional[str] = None, conn = Depends(get_db)):
    with conn.cursor() as cur:
        if date:
            cur.execute("""SELECT a.*, s.name FROM attendance a 
                           JOIN students s ON a.roll_number = s.roll_number 
                           WHERE a.date = %s""", (date,))
        else:
            cur.execute("""SELECT a.*, s.name FROM attendance a 
                           JOIN students s ON a.roll_number = s.roll_number""")
        return cur.fetchall()

@app.get("/api/absent", response_model=List[dict])
def get_absent_requests(conn = Depends(get_db)):
    with conn.cursor() as cur:
        cur.execute("""SELECT ar.*, s.name FROM absent_requests ar 
                       JOIN students s ON ar.roll_number = s.roll_number""")
        return cur.fetchall()

@app.get("/api/attendance/stats")
def get_attendance_stats(conn = Depends(get_db)):
    with conn.cursor() as cur:
        # Get total distinct attendance dates (i.e., max possible attendance days)
        cur.execute("SELECT COUNT(DISTINCT date) as max_days FROM attendance")
        max_days = cur.fetchone()["max_days"] or 0

        cur.execute("""
            SELECT s.roll_number, s.name, 
                   COUNT(a.*) as total_days_recorded,
                   SUM(CASE WHEN a.present THEN 1 ELSE 0 END) as present_days
            FROM students s
            LEFT JOIN attendance a ON s.roll_number = a.roll_number
            GROUP BY s.roll_number, s.name
        """)
        raw_stats = cur.fetchall()

        # Add percentage using max_days instead of total_days_recorded
        for s in raw_stats:
            s["total_days"] = max_days
            s["percentage"] = round((s["present_days"] or 0) * 100.0 / max_days, 2) if max_days > 0 else 0.0

        defaulters = [s for s in raw_stats if s["percentage"] < 75]

        # Plot
        plt.figure(figsize=(10, 5))
        names = [s['name'] for s in raw_stats]
        percentages = [s['percentage'] for s in raw_stats]
        plt.bar(names, percentages)
        plt.xlabel('Students')
        plt.ylabel('Attendance %')
        plt.title('Attendance Statistics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return {"stats": raw_stats, "defaulters": defaulters, "plot": plot_data}


@app.get("/api/face/students")
def get_student_embeddings(conn = Depends(get_db)):
    with conn.cursor() as cur:
        cur.execute("SELECT roll_number, name, embedding FROM students WHERE embedding IS NOT NULL")
        return cur.fetchall()

@app.post("/api/face/register")
async def register_face(
    roll_number: str = Form(...),
    name: str = Form(...),
    department: str = Form(...),
    image: UploadFile = File(...),
    conn = Depends(get_db)
):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=FACE_DETECTION_SCALE,
        minNeighbors=FACE_DETECTION_NEIGHBORS,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )

    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected in image")

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    embedding = get_embedding(face)

    with conn.cursor() as cur:
        cur.execute("""INSERT INTO students (roll_number, name, department, embedding)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (roll_number) DO UPDATE
                       SET name = EXCLUDED.name, department = EXCLUDED.department, embedding = EXCLUDED.embedding""",
                    (roll_number, name, department, embedding.tolist()))
        conn.commit()

    return {"message": "Face registered successfully"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

def generate_frames():
    cap = None
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Using camera at index {camera_index}")
            break
        else:
            print(f"Camera at index {camera_index} not available")

    if not cap or not cap.isOpened():
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Error", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    try:
        with psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT roll_number, name, embedding FROM students WHERE embedding IS NOT NULL")
                students = [(r['roll_number'], r['name'], np.array(r['embedding'])) for r in cur.fetchall()]
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        students = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION_SCALE,
                minNeighbors=FACE_DETECTION_NEIGHBORS,
                minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
            )
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                try:
                    embedding = get_embedding(face)
                    roll, name = find_best_match(embedding, students)

                    if roll:
                        color = (0, 255, 0)
                        text = f"{name} ({roll})"
                        try:
                            with psycopg2.connect(DATABASE_URL) as conn:
                                with conn.cursor() as cur:
                                    cur.execute("""
                                        INSERT INTO attendance (roll_number, present, date)
                                        VALUES (%s, %s, %s)
                                        ON CONFLICT (roll_number, date) DO UPDATE
                                        SET present = EXCLUDED.present
                                    """, (roll, True, date.today()))
                                    conn.commit()
                        except psycopg2.Error as e:
                            print(f"Attendance marking error: {e}")
                    else:
                        color = (0, 0, 255)
                        text = "Unknown"

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                except Exception as e:
                    print(f"Face processing error: {e}")
                    continue

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    finally:
        if cap:
            cap.release()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
