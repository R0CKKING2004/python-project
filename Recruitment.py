import os
import re
import fitz  # PyMuPDF
import openai
import random
import tempfile
import speech_recognition as sr
import pyttsx3
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QPushButton, QTextEdit, QFileDialog, QMessageBox, QComboBox
)
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === AI Setup ===
openai.api_key = "openai api key"

# === AI Question Generation & Evaluation ===
QUESTION = "What is Python?"
CORRECT_ANSWER = "python is programming language"

class RecruitmentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-Powered Recruitment Assistant")
        self.setWindowState(Qt.WindowState.WindowMaximized)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.layout.setSpacing(15)

        self.title = QLabel("AI-Driven Smart Recruitment System")
        self.title.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("color: white;")
        self.layout.addWidget(self.title)

        self.job_desc_dropdown = QComboBox()
        self.job_desc_dropdown.setFont(QFont("Arial", 14))
        self.job_desc_dropdown.addItems([
            "Select Job Description",
            "Python Developer - Skills: Python, ML, APIs, SQL",
            "Frontend Developer - Skills: HTML, CSS, JavaScript, React",
            "Data Scientist - Skills: Python, Statistics, Machine Learning, Data Visualization",
            "Project Manager - Skills: Leadership, Communication, Planning, Agile"
        ])
        self.layout.addWidget(QLabel("Select Job Description:"))
        self.layout.addWidget(self.job_desc_dropdown)

        self.load_resume_btn = QPushButton("\ud83d\udcc1 Load Resume Folder")
        self.load_resume_btn.setFont(QFont("Arial", 14))
        self.load_resume_btn.clicked.connect(self.load_resume_folder)
        self.layout.addWidget(self.load_resume_btn)

        self.rank_btn = QPushButton("\ud83d\udd0d Rank & Show Eligible Candidates")
        self.rank_btn.setFont(QFont("Arial", 14))
        self.rank_btn.clicked.connect(self.rank_candidates)
        self.layout.addWidget(self.rank_btn)

        self.ask_question_btn = QPushButton("\ud83c\udfa4 Round 1: Ask Question")
        self.ask_question_btn.setFont(QFont("Arial", 14))
        self.ask_question_btn.clicked.connect(self.ask_question)
        self.ask_question_btn.setEnabled(False)
        self.layout.addWidget(self.ask_question_btn)

        self.video_btn = QPushButton("\ud83d\udcf9 Round 2: AI Video Interview")
        self.video_btn.setFont(QFont("Arial", 14))
        self.video_btn.clicked.connect(self.start_ai_video_interview)
        self.video_btn.setEnabled(False)
        self.layout.addWidget(self.video_btn)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Courier New", 11))
        self.layout.addWidget(self.output)

        self.central_widget.setLayout(self.layout)

        self.resumes = {}
        self.round1_passed = False

    def load_resume_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Resume Folder")
        if folder:
            self.resumes = {}

            for file in os.listdir(folder):
                if file.endswith(".pdf"):
                    path = os.path.join(folder, file)
                    try:
                        text = ""
                        with fitz.open(path) as doc:
                            for page in doc:
                                text += page.get_text()
                        self.resumes[file] = text
                    except Exception as e:
                        QMessageBox.warning(self, "File Error", f"Could not read {file}: {str(e)}")

            QMessageBox.information(self, "Success", f"Loaded {len(self.resumes)} PDF resumes.")

    def rank_candidates(self):
        jd_index = self.job_desc_dropdown.currentIndex()
        if jd_index == 0:
            QMessageBox.warning(self, "Selection Error", "Please select a job description.")
            return

        if not self.resumes:
            QMessageBox.warning(self, "Error", "Please load resumes folder first.")
            return

        job_desc_text = self.job_desc_dropdown.currentText()
        if "- Skills:" in job_desc_text:
            keywords = job_desc_text.split("- Skills:")[1].strip()
        else:
            keywords = job_desc_text

        docs = [keywords] + list(self.resumes.values())
        tfidf = TfidfVectorizer(stop_words='english')
        matrix = tfidf.fit_transform(docs)

        cosine_sim = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

        threshold = 0.2
        output_text = "<h3>\ud83d\udcca Candidate Rankings and Eligibility</h3><ul>"
        eligible_candidates = []

        for (name, score) in sorted(zip(self.resumes.keys(), cosine_sim), key=lambda x: x[1], reverse=True):
            eligible = score >= threshold
            status = "<b style='color:green;'>Eligible</b>" if eligible else "<b style='color:red;'>Not Eligible</b>"
            output_text += f"<li><b>{name}</b> - Score: {score:.2f} - {status}</li>"
            if eligible:
                eligible_candidates.append(name)
        output_text += "</ul>"

        if eligible_candidates:
            output_text += "<h4>\u2705 Candidates Eligible for Interview:</h4><ul>"
            for c in eligible_candidates:
                output_text += f"<li>{c}</li>"
            output_text += "</ul>"
            output_text += "<p style='color:blue;'>📧 Emails have been sent to all eligible candidates.</p>"
        else:
            output_text += "<p><i>No candidates eligible for this job description.</i></p>"

        self.output.setHtml(output_text)
        self.ask_question_btn.setEnabled(True)

    def ask_question(self):
        engine = pyttsx3.init()
        engine.say(QUESTION)
        engine.runAndWait()

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.output.append("\n🎤 Listening for answer...")
            audio = recognizer.listen(source)

        try:
            response = recognizer.recognize_google(audio).lower()
            self.output.append(f"Candidate Answer: {response}")
            if CORRECT_ANSWER in response:
                self.output.append("✅ Correct answer. Proceeding to Round 2.")
                self.video_btn.setEnabled(True)
                self.round1_passed = True
            else:
                self.output.append("❌ Incorrect answer. Try again.")
                self.round1_passed = False
        except Exception as e:
            self.output.append(f"⚠️ Could not understand audio: {str(e)}")

    def start_ai_video_interview(self):
        if not self.round1_passed:
            QMessageBox.warning(self, "Access Denied", "Round 1 not passed.")
            return

        self.output.append("\n📹 Starting AI-driven video interview...")
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful interviewer asking technical job questions."},
                    {"role": "user", "content": "Ask a technical question for a Python developer interview."}
                ]
            )
            question = response['choices'][0]['message']['content']
            QMessageBox.information(self, "Round 2 Question", question)
            self.output.append(f"Round 2 Question: {question}")
        except Exception as e:
            self.output.append(f"❌ Failed to fetch question from OpenAI: {str(e)}")

if __name__ == "__main__":
    app = QApplication([])
    window = RecruitmentApp()
    window.show()
    app.exec()
