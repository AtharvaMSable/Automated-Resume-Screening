# --- START OF FILE app.py ---

import os
import pandas as pd
import pickle
from pypdf import PdfReader # Using pypdf instead of pdfminer3 directly
from docx import Document
import re
import streamlit as st
import nltk
from nltk.corpus import stopwords
import time
import datetime
import base64
import io
from streamlit_tags import st_tags
import pymysql
try:
    # Ensure Courses.py defines these lists, e.g., ds_course = [('Name', 'Link'), ...]
    from Courses import ds_course, web_course, android_course, ios_course, uiux_course
    courses_available = True
except ImportError:
    st.sidebar.warning("Courses.py not found. Course recommendations will be disabled.")
    courses_available = False
    ds_course, web_course, android_course, ios_course, uiux_course = [], [], [], [], []
import plotly.express as px
import random
import spacy
import numpy as np

# --- Deep Learning Imports ---
transformers_available = False
summarizer_pipeline = None
try:
    from transformers import pipeline
    # Check if torch or tensorflow is installed
    try:
        import torch
    except ImportError:
        try:
            import tensorflow
        except ImportError:
            st.warning("Neither PyTorch nor TensorFlow found. Summarization pipeline might default or fail.")
    transformers_available = True
except ImportError:
    st.sidebar.warning("Hugging Face 'transformers' library not installed. Summarization disabled.")

# --- SET PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    layout="wide",
    page_title="Smart Resume Analyzer (DL)"
)

# --- Download necessary NLTK data ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        # fallback: download manually (won't work on Streamlit Cloud)
        nltk.download('stopwords')

download_nltk_data()
# Update UI after cache call if needed (optional)
# st.sidebar.info("NLTK Check Done.")

# --- Load ML Models ---
@st.cache_resource
def load_ml_models():
    try:
        vector = pickle.load(open("tfidf.pkl", "rb"))
        ml_model = pickle.load(open("model.pkl", "rb"))
        return vector, ml_model, True, None
    except FileNotFoundError:
        return None, None, False, "Error: tfidf.pkl or model.pkl not found."
    except Exception as e:
        return None, None, False, f"Error loading ML models: {e}"

word_vector, model, models_loaded, ml_load_error = load_ml_models()
if models_loaded: st.sidebar.success("ML Categorization Models Loaded.")
elif ml_load_error: st.sidebar.error(ml_load_error); st.error("ML model files not found/failed. Categorization disabled.")

# --- Load spaCy Model ---
@st.cache_resource
def load_spacy_model(model_name="en_core_web_sm"):
    try:
        nlp = spacy.load(model_name)
        return nlp, True, None
    except OSError:
        return None, False, f"spaCy model '{model_name}' not found. Please run: python -m spacy download {model_name}"
    except Exception as e:
         return None, False, f"Error loading spaCy model '{model_name}': {e}"

nlp, spacy_loaded, spacy_load_error = load_spacy_model()
ner_enabled = nlp is not None
if spacy_loaded: st.sidebar.success("spaCy NER model loaded.")
elif spacy_load_error: st.sidebar.error(spacy_load_error); st.error("NER skill extraction disabled.")

# --- Load Summarization Pipeline ---
@st.cache_resource
def load_summarizer():
    if not transformers_available: return None, False, "Transformers library not installed."
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", truncation=True)
        return summarizer, True, None
    except Exception as e:
        return None, False, f"Error loading summarization pipeline: {e}"

summarizer_pipeline, summarizer_enabled, summarizer_load_error = load_summarizer()
if summarizer_enabled: st.sidebar.success("Summarization model loaded.")
elif summarizer_load_error: st.sidebar.error(summarizer_load_error); st.error("Summarization disabled.")

# --- Constants and Mappings ---
SKILLS_DB = [
    'python', 'java', 'c++', 'c#', 'javascript', 'js', 'html', 'css', 'php', 'ruby', 'swift', 'kotlin',
    'sql', 'mysql', 'postgresql', 'sqlite', 'mongodb', 'cassandra', 'redis', 'oracle', 'sql server',
    'aws', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'git',
    'linux', 'unix', 'windows', 'macos',
    'react', 'angular', 'vue', 'nodejs', 'django', 'flask', 'spring', 'ruby on rails', '.net',
    'pandas', 'numpy', 'scipy', 'scikit-learn', 'sklearn', 'tensorflow', 'keras', 'pytorch', 'matplotlib', 'seaborn', 'plotly',
    'machine learning', 'deep learning', 'data science', 'data analysis', 'data visualization', 'nlp', 'natural language processing',
    'computer vision', 'big data', 'hadoop', 'spark', 'kafka', 'hive', 'hbase', 'spacy', 'nltk',
    'agile', 'scrum', 'jira', 'project management', 'product management',
    'communication', 'teamwork', 'leadership', 'problem solving', 'critical thinking',
    'customer service', 'sales', 'marketing', 'seo', 'sem', 'content creation',
    'ui/ux', 'design', 'photoshop', 'illustrator', 'figma',
    'devops', 'automation testing', 'selenium', 'cybersecurity', 'network security',
    'sap', 'etl', 'power bi', 'tableau', 'excel', 'word', 'powerpoint',
    'blockchain', 'solidity', 'ethereum', 'hyperledger',
    'mechanical engineering', 'electrical engineering', 'civil engineering',
    'hr', 'recruitment', 'talent acquisition', 'employee relations',
    'health', 'fitness', 'nutrition',
    'advocate', 'legal', 'law',
    'jquery', 'bootstrap', 'd3.js', 'dc.js', 'logstash', 'kibana', 'r', 'sap hana',
    'rest', 'soap', 'api', 'microservices',
    'pmo', 'operations management', 'business analysis', 'dotnet'
]
SKILLS_DB_LOWER = set([s.lower() for s in SKILLS_DB])

category_mapping = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer",
    20: "Python Developer", 24: "Web Designing", 12: "HR",
    13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
    18: "Operations Manager", 6: "Data Science", 22: "Sales",
    16: "Mechanical Engineer", 1: "Arts", 7: "Database",
    11: "Electrical Engineering", 14: "Health and fitness", 19: "PMO",
    4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
    17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer",
    0: "Advocate",
}

# --- Text Cleaning Function (General Purpose) ---
def clean_text_general(txt):
    txt = str(txt).lower()
    txt = re.sub('http\S+\s', ' ', txt); txt = re.sub('rt|cc', ' ', txt)
    txt = re.sub('#\S+', '', txt); txt = re.sub('@\S+', '  ', txt)
    txt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt); txt = re.sub('\s+', ' ', txt)
    return txt.strip()

# --- Resume Cleaning Function (For ML Model) ---
def cleanResumeForCategorization(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    # Optional stopword removal
    # stop_words = set(stopwords.words('english'))
    # cleanText = ' '.join(word for word in cleanText.split() if word.lower() not in stop_words)
    return cleanText.lower()

# --- Helper Functions ---

# Text Extraction
def pdf_reader(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n" # Add newline between pages
    except Exception as e:
        st.error(f"Error reading PDF {os.path.basename(file_path)} with pypdf: {e}")
        # Fallback or alternative method could be added here if needed
    if not text:
        st.warning(f"Could not extract text from {os.path.basename(file_path)} using pypdf.")
    return text

def extract_text_from_docx(file):
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        if not text: st.warning(f"Could not extract text from DOCX {file.name}.")
        return text
    except Exception as e:
        st.error(f"Error reading DOCX {file.name}: {e}")
        return ""

# Skill Extraction & Analysis
def extract_skills_rule_based(resume_text):
    processed_text = ' '.join(resume_text.lower().split())
    found_skills = set()
    for skill in SKILLS_DB_LOWER:
        skill_pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(skill_pattern, processed_text):
            original_skill = next((s for s in SKILLS_DB if s.lower() == skill), skill)
            found_skills.add(original_skill)
    return list(found_skills)

def extract_skills_ner(resume_text, nlp_model):
    if not nlp_model: return []
    doc = nlp_model(resume_text)
    found_skills_ner = set()
    potential_skill_labels = {"ORG", "PRODUCT", "WORK_OF_ART", "LAW", "NORP", "PERSON", "GPE"} # Broader set
    for ent in doc.ents:
        ent_text_lower = ent.text.lower().strip()
        # Prioritize direct match with SKILLS_DB
        if ent_text_lower in SKILLS_DB_LOWER:
             original_skill = next((s for s in SKILLS_DB if s.lower() == ent_text_lower), ent_text_lower)
             found_skills_ner.add(original_skill)
        # Maybe add entities if they are likely skills based on label AND not just noise
        elif ent.label_ in potential_skill_labels:
            # Basic filtering (avoid single characters, very long strings, pure numbers)
            if 2 <= len(ent_text_lower) < 30 and not ent_text_lower.isdigit():
                 # Check if any part of the entity matches a skill variation (e.g., "machine learning algorithms")
                 # This is optional and adds complexity
                 # for skill_db_item in SKILLS_DB_LOWER:
                 #     if skill_db_item in ent_text_lower:
                 #         original_skill = next((s for s in SKILLS_DB if s.lower() == skill_db_item), skill_db_item)
                 #         found_skills_ner.add(original_skill)
                 #         break # Add only the first matching skill part found

                 # Or just add the entity text if its label is relevant (can be noisy)
                 # found_skills_ner.add(ent.text.strip())
                 pass # Keep it conservative: only add if in SKILLS_DB for now
    return list(found_skills_ner)


def calculate_resume_score(resume_skills, jd_skills):
    if not jd_skills: return 0
    resume_skills_set = set(s.lower() for s in resume_skills)
    jd_skills_set = set(s.lower() for s in jd_skills)
    matching_skills = resume_skills_set.intersection(jd_skills_set)
    score = (len(matching_skills) / len(jd_skills_set)) * 100 if jd_skills_set else 0
    return round(score, 2)

def recommend_skills(resume_skills, jd_skills):
    resume_skills_set = set(s.lower() for s in resume_skills)
    jd_skills_set = set(s.lower() for s in jd_skills)
    missing_skills = jd_skills_set.difference(resume_skills_set)
    return list(missing_skills)

# Utility
def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def show_pdf_streamlit(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except FileNotFoundError: st.error(f"Could not find PDF: {file_path}")
    except Exception as e: st.error(f"Error displaying PDF {os.path.basename(file_path)}: {e}")

def course_recommender(course_list):
    # Needs to be called within the display block of analysis results
    # Assumes course_list contains tuples ('Course Name', 'Link')
    if not courses_available: return [] # Return empty list if Courses.py not found
    if not course_list: return []

    displayed_courses = []
    try:
        # Use a unique key for the slider to avoid conflicts if called elsewhere
        no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, min(10, len(course_list)), 4, key='course_slider_display_main')
        st.markdown("*(Based on Predicted Category)*")
        displayed_count = 0
        # No need to shuffle here if it was done when storing in session state
        for c_name, c_link in course_list:
             if displayed_count < no_of_reco:
                 st.markdown(f"({displayed_count+1}) [{c_name}]({c_link})")
                 displayed_courses.append(c_name) # Log displayed course names
                 displayed_count += 1
             else: break
    except Exception as e:
        st.warning(f"Error displaying course recommendations: {e}") # Handle potential errors
    return displayed_courses


# --- Database Setup ---
connection = None; cursor = None; db_enabled = False; db_setup_error = None
try:
    connection = pymysql.connect(host='localhost', user='root', password='') # Add your password
    cursor = connection.cursor(); cursor.execute("CREATE DATABASE IF NOT EXISTS SRA;")
    connection.select_db("sra"); DB_table_name = 'user_data_dl_final' # New table name
    table_sql = f"""CREATE TABLE IF NOT EXISTS {DB_table_name} (
                     ID INT NOT NULL AUTO_INCREMENT,
                     Name varchar(100) NULL,
                     Email_ID VARCHAR(100) NULL,
                     Timestamp VARCHAR(50) NOT NULL,
                     File_Name VARCHAR(255) NOT NULL,
                     ML_Predicted_Category VARCHAR(50) NOT NULL,
                     Extracted_Skills VARCHAR(1500) NULL,
                     JD_Score VARCHAR(10) NULL,
                     Missing_Skills VARCHAR(1500) NULL,
                     Recommended_Courses VARCHAR(1000) NULL,
                     PRIMARY KEY (ID));
                    """
    cursor.execute(table_sql); db_enabled = True
except pymysql.Error as db_err: db_setup_error = f"DB Connection Error: {db_err}"
except Exception as e: db_setup_error = f"DB setup error: {e}"

if db_enabled: st.sidebar.success("Database Connection Successful.")
elif db_setup_error: st.sidebar.error(db_setup_error); st.warning("Database features disabled.")

def insert_data(name, email, timestamp, file_name, ml_category, skills, jd_score, missing_skills, courses):
    if not db_enabled or cursor is None or connection is None: return
    DB_table_name = 'user_data_dl_final'
    max_len_skills = 1500; max_len_courses = 1000
    skills_str = (str(skills)[:max_len_skills - 3] + '...') if len(str(skills)) > max_len_skills else str(skills)
    missing_str = (str(missing_skills)[:max_len_skills - 3] + '...') if len(str(missing_skills)) > max_len_skills else str(missing_skills)
    courses_str = (str(courses)[:max_len_courses - 3] + '...') if len(str(courses)) > max_len_courses else str(courses)

    insert_sql = f"""INSERT INTO {DB_table_name}
                    (Name, Email_ID, Timestamp, File_Name, ML_Predicted_Category, Extracted_Skills, JD_Score, Missing_Skills, Recommended_Courses)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    rec_values = (name, email, timestamp, file_name, ml_category, skills_str, str(jd_score), missing_str, courses_str)
    try:
        cursor.execute(insert_sql, rec_values)
        connection.commit()
    except pymysql.Error as db_err: st.error(f"DB Insert Error: {db_err}"); connection.rollback()
    except Exception as e: st.error(f"Error during DB insertion: {e}"); connection.rollback()


# --- Session State Initialization ---
if 'categorization_results' not in st.session_state: st.session_state.categorization_results = None
if 'uploaded_file_details' not in st.session_state: st.session_state.uploaded_file_details = {}
if 'analyze_clicked' not in st.session_state: st.session_state.analyze_clicked = False
if 'analysis_output' not in st.session_state: st.session_state.analysis_output = None

# --- UI Mode Selection ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode:", ["🏠 Home", "👤 User Analysis", "🔑 Admin Panel"])

# ================================
# === NORMAL USER MODE =========
# ================================
if app_mode == '👤 User Analysis':
    st.title("👤 User Analysis")
    st.markdown("Upload resumes for categorization, analysis, and AI summarization.")
    st.markdown("---")

    # --- File Upload and Categorization Section ---
    st.header("1. Upload & Categorize Resumes")
    uploaded_files = st.file_uploader(
        "Choose PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True, key="user_file_uploader"
    )
    output_directory = st.text_input("Output Directory for Categorized Files", "categorized_resumes_ml", key="user_output_dir")

    # Clear previous results if new files are uploaded
    if uploaded_files:
        # Check if the list of uploaded file names is different from what might be implicitly stored
        # A simpler approach is just to clear results before the button is processed
        pass # We will clear inside the button press

    if st.button("📁 Categorize Resumes", key="user_categorize_button"):
        # Clear previous results explicitly
        st.session_state.categorization_results = None
        st.session_state.uploaded_file_details = {}
        st.session_state.analyze_clicked = False # Reset analysis state too
        st.session_state.analysis_output = None

        if uploaded_files and output_directory:
            if not os.path.exists(output_directory):
                try: os.makedirs(output_directory)
                except OSError as e: st.error(f"Could not create output dir: {e}"); st.stop()

            categorization_data = [] # Local list for this run

            if not models_loaded:
                 st.error("ML Models not loaded. Cannot perform categorization."); st.stop()

            progress_bar = st.progress(0)
            status_text = st.empty()
            processed_files = 0; failed_files = 0

            with st.spinner("Categorizing resumes..."):
                for i, uploaded_file in enumerate(uploaded_files):
                    file_name = uploaded_file.name
                    file_ext = os.path.splitext(file_name)[1].lower()
                    text = ""; temp_save_path = os.path.join(".", f"temp_{file_name}")
                    status_text.text(f"Processing {file_name}...")

                    try:
                        uploaded_file.seek(0) # Reset pointer
                        with open(temp_save_path, "wb") as f: f.write(uploaded_file.getbuffer())

                        if file_ext == '.pdf': text = pdf_reader(temp_save_path)
                        elif file_ext == '.docx': uploaded_file.seek(0); text = extract_text_from_docx(uploaded_file)
                        else: st.warning(f"Skipping unsupported file: {file_name}"); failed_files += 1; continue

                        category_name = "Unknown"
                        if text:
                            st.session_state.setdefault('uploaded_file_details', {})[file_name] = {'text': text}
                            cleaned_resume_for_model = cleanResumeForCategorization(text)
                            if cleaned_resume_for_model.strip():
                                try:
                                    input_features = word_vector.transform([cleaned_resume_for_model])
                                    prediction_id = model.predict(input_features)[0]
                                    category_name = category_mapping.get(prediction_id, "Unknown")
                                except Exception as model_err: st.error(f"ML prediction error: {model_err}"); category_name = "Prediction Error"
                            else: st.warning(f"Cleaned text empty."); category_name = "Empty Cleaned Text"

                            if file_name in st.session_state.uploaded_file_details:
                                 st.session_state.uploaded_file_details[file_name]['category'] = category_name

                            category_folder = os.path.join(output_directory, str(category_name)); target_path = None
                            if not os.path.exists(category_folder):
                                try: os.makedirs(category_folder); target_path = os.path.join(category_folder, file_name)
                                except OSError as e: st.warning(f"Could not create category folder: {e}")
                            else: target_path = os.path.join(category_folder, file_name)

                            if target_path:
                                try:
                                    uploaded_file.seek(0)
                                    with open(target_path, "wb") as f: f.write(uploaded_file.getbuffer())
                                except Exception as e: st.warning(f"Could not save file: {e}")

                            categorization_data.append({'Filename': file_name, 'Predicted Category': category_name})
                            processed_files += 1
                        else: st.warning(f"Could not extract text from {file_name}."); failed_files += 1

                    except Exception as file_proc_err: st.error(f"Error processing {file_name}: {file_proc_err}"); failed_files += 1
                    finally:
                         if os.path.exists(temp_save_path):
                             try: os.remove(temp_save_path)
                             except OSError as e: st.warning(f"Could not remove temp file: {e}")

                    progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text(f"Categorization complete! Processed: {processed_files}, Failed/Skipped: {failed_files}")
            if categorization_data:
                st.session_state.categorization_results = pd.DataFrame(categorization_data)
                st.rerun() # Rerun to show results below
            else:
                st.warning("No resumes were successfully processed.")
                # Ensure results are cleared if processing failed for all
                st.session_state.categorization_results = None
        else:
            st.error("Please upload files and specify output directory.")

    # --- Display Categorization Results ---
    if st.session_state.get('categorization_results') is not None and not st.session_state['categorization_results'].empty:
        st.markdown("---")
        st.subheader("Categorization Results (ML)")
        st.dataframe(st.session_state['categorization_results'], use_container_width=True)
        results_csv = st.session_state['categorization_results'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV", data=results_csv,
            file_name='categorized_resumes_ml.csv', mime='text/csv', key="user_download_csv_ml"
        )

    # --- Detailed Analysis Section ---
    st.markdown("---")
    st.header("2. Detailed Resume Analysis (vs. Job Description)")

    if st.session_state.get('uploaded_file_details'):
        available_files = list(st.session_state['uploaded_file_details'].keys())
        if not available_files:
             st.info("Upload and categorize resumes first to enable analysis.")
        else:
            # --- Inputs ---
            selected_file = st.selectbox("Select Resume for Detailed Analysis:", available_files, key="user_selected_resume_final")
            job_description = st.text_area("Paste Job Description Here:", height=250, key="user_job_desc_final")

            # --- Analyze Button ---
            if st.button("🔍 Analyze Resume vs. Job Description", key="user_analyze_button_final"):
                st.session_state.analyze_clicked = False # Reset flag before analysis
                st.session_state.analysis_output = None # Clear old output

                if selected_file and job_description:
                    resume_details = st.session_state['uploaded_file_details'].get(selected_file)
                    if resume_details and 'text' in resume_details:
                        # --- Perform Analysis Calculation ---
                        with st.spinner("Analyzing... This may take a moment."):
                            resume_text = resume_details['text']
                            predicted_category = resume_details.get('category', 'N/A')

                            # Skills
                            resume_skills_rule = extract_skills_rule_based(resume_text)
                            resume_skills_ner = extract_skills_ner(resume_text, nlp) if ner_enabled else []
                            combined_skills = sorted(list(set(resume_skills_rule) | set(resume_skills_ner)))

                            # JD Skills & Score
                            jd_skills = extract_skills_rule_based(job_description)
                            score = calculate_resume_score(combined_skills, jd_skills) if jd_skills else 0

                            # Missing/Matching Skills
                            missing_skills = recommend_skills(combined_skills, jd_skills) if jd_skills else []
                            matching_skills_list = []
                            if jd_skills:
                                combined_lower = set(s.lower() for s in combined_skills)
                                jd_lower = set(s.lower() for s in jd_skills)
                                matching_skills_list = sorted(list(combined_lower.intersection(jd_lower)))

                            # Course List Generation
                            full_recommended_courses = []
                            # ... [Generate full course list logic as before] ...
                            if courses_available and predicted_category not in ["Unknown", "Prediction Error", "Empty Cleaned Text"]:
                                category_course_map = {
                                    'Data Science': ds_course, 'Web Designing': web_course,
                                    'Android Development': android_course, 'IOS Development': ios_course,
                                    'UI-UX Development': uiux_course, 'Java Developer': web_course,
                                    'Python Developer': ds_course,
                                }
                                course_list_to_use = category_course_map.get(predicted_category)
                                if course_list_to_use:
                                    full_recommended_courses = course_list_to_use[:]
                                    random.shuffle(full_recommended_courses)

                            # Summarization (Generate summaries here)
                            resume_summary = "Summarization disabled or failed."
                            jd_summary = "Summarization disabled or failed."
                            if summarizer_enabled:
                                try:
                                    max_summary_input = 1024; min_summary = 30; max_summary = 130
                                    clean_res_summary = clean_text_general(resume_text)
                                    resume_summary = summarizer_pipeline(clean_res_summary[:max_summary_input], max_length=max_summary, min_length=min_summary, do_sample=False)[0]['summary_text']
                                except Exception as e: resume_summary = f"Error: {e}"
                                try:
                                    clean_jd_summary = clean_text_general(job_description)
                                    jd_summary = summarizer_pipeline(clean_jd_summary[:max_summary_input], max_length=max_summary, min_length=min_summary, do_sample=False)[0]['summary_text']
                                except Exception as e: jd_summary = f"Error: {e}"


                            # --- Store results ---
                            st.session_state.analysis_output = {
                                "selected_file": selected_file, "job_description": job_description,
                                "predicted_category": predicted_category,
                                "resume_skills_rule": resume_skills_rule, "resume_skills_ner": resume_skills_ner,
                                "combined_skills": combined_skills, "jd_skills": jd_skills, "score": score,
                                "missing_skills": missing_skills, "matching_skills": matching_skills_list,
                                "recommended_courses_full": full_recommended_courses,
                                "resume_summary": resume_summary, # Store summary
                                "jd_summary": jd_summary          # Store summary
                            }
                            st.session_state.analyze_clicked = True

                            # --- DB Insert ---
                            if db_enabled:
                                ts = time.time(); timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
                                name = "N/A"; email = "N/A"
                                insert_data(name, email, timestamp, selected_file, predicted_category,
                                            ", ".join(combined_skills) if combined_skills else "None", score,
                                            ", ".join(missing_skills) if missing_skills else "None",
                                            ", ".join([c[0] for c in full_recommended_courses]) if full_recommended_courses else "None")
                            st.rerun() # Trigger display
                        # --- End Spinner ---
                    else: st.error("Could not retrieve text."); st.session_state.analyze_clicked = False; st.session_state.analysis_output = None
                else: st.warning("Select resume and enter JD."); st.session_state.analyze_clicked = False; st.session_state.analysis_output = None

            # --- Display Analysis Results Block ---
            if st.session_state.get('analyze_clicked', False) and st.session_state.get('analysis_output') is not None:
                # Check if inputs match the analysis stored
                if st.session_state.analysis_output.get("selected_file") == selected_file and \
                   st.session_state.analysis_output.get("job_description") == job_description:

                    results = st.session_state.analysis_output # Use stored results

                    st.markdown("---")
                    st.subheader(f"Analysis Results for: {results['selected_file']}")
                    st.info(f"ML Predicted Category: **{results['predicted_category']}**")

                    # Display Summaries
                    if summarizer_enabled:
                        st.markdown("---")
                        st.markdown("### AI-Generated Summaries (Deep Learning - DistilBART):")
                        st.subheader("Resume Summary:")
                        st.success(results.get("resume_summary", "Not available."))
                        st.subheader("Job Description Summary:")
                        st.info(results.get("jd_summary", "Not available."))
                        st.markdown("---")

                    # Display Skills
                    analysis_col1, analysis_col2 = st.columns(2)
                    with analysis_col1:
                        st.markdown("##### **Extracted Skills:**")
                        # ... [Expanders for Rule/NER skills from 'results'] ...
                        with st.expander("Rule-Based Skills"):
                             if results['resume_skills_rule']: st_tags(label='', text='', value=sorted(results['resume_skills_rule']), key='disp_rule_skills_f')
                             else: st.write("None found via rules.")
                        with st.expander("NER-Based Skills (beta)"):
                             if ner_enabled:
                                 if results['resume_skills_ner']: st_tags(label='', text='', value=sorted(results['resume_skills_ner']), key='disp_ner_skills_f')
                                 else: st.write("NER found no known skills.")
                             else: st.warning("NER model not loaded.")
                        st.markdown("**Combined Skills:**")
                        if results['combined_skills']: st_tags(label='', text='Union:', value=results['combined_skills'], key='disp_combined_skills_f')
                        else: st.write("No skills identified.")


                    # Display Score
                    with analysis_col2:
                        st.markdown("##### **Resume Score (Skill Match vs JD):**")
                        # ... [Display score from 'results'] ...
                        if results['jd_skills']: st.metric(label="Skill Match %", value=f"{results['score']}%"); st.progress(int(results['score']))
                        else: st.write("No skills extracted from JD.")


                    # Display Course Recommendations
                    st.markdown("---")
                    # ** Call course_recommender HERE, using the *stored* full list **
                    course_recommender(results.get("recommended_courses_full", []))


                    # Display Skills Analysis (Matching/Missing)
                    st.markdown("##### **Skills Analysis (vs JD):**")
                    # ... [Display expanders for matching/missing skills from 'results'] ...
                    if results['jd_skills']:
                        expander_match = st.expander("**Matching Skills (Combined)**")
                        with expander_match:
                            if results['matching_skills']: st.success(f"{len(results['matching_skills'])} matching:"); st.write(", ".join(results['matching_skills']))
                            else: st.warning("No matching skills found.")

                        expander_rec = st.expander("**Recommended Skills (Missing from Resume - Combined)**")
                        with expander_rec:
                            if results['missing_skills']: st.warning(f"Consider adding/highlighting:"); st.write(", ".join(sorted(results['missing_skills'])))
                            elif results['matching_skills']: st.success("All skills from JD seem present.")
                            else: st.info("No skills extracted from JD.")
                    else: st.info("No skills extracted from JD for analysis.")
                    # --- End Display ---

    else:
        st.info("Upload and categorize resumes first to enable detailed analysis.")
# ================================
# === ADMIN PANEL MODE ===========
# ================================
elif app_mode == '🔑 Admin Panel':
    st.title("🔑 Admin Panel")
    st.markdown("Login to view user data and analysis results.")
    st.markdown("---")

    # Basic Login Form
    ad_user = st.text_input("Username", key="admin_user_main")
    ad_password = st.text_input("Password", type='password', key="admin_pass_main")

    if st.button('Login', key="admin_login_main"):
        if ad_user == 'admin' and ad_password == 'password': # Replace with secure check
            st.success("Admin Login Successful!")
            st.markdown("---")

            if db_enabled and cursor is not None:
                try:
                    DB_table_name = 'user_data_ml'
                    cursor.execute(f'''SELECT * FROM {DB_table_name}''')
                    data = cursor.fetchall()
                    st.header("**User Upload & Analysis Data**")

                    if data:
                        column_names = [desc[0] for desc in cursor.description]
                        df_admin = pd.DataFrame(data, columns=column_names)
                        st.dataframe(df_admin, use_container_width=True)
                        st.markdown(get_table_download_link(df_admin, 'User_Analysis_Data.csv', 'Download Report'), unsafe_allow_html=True)

                        # Visualizations
                        st.markdown("---")
                        st.header("📊 Data Visualizations")
                        st.subheader("Predicted Category Distribution")
                        if 'ML_Predicted_Category' in df_admin.columns:
                            category_counts = df_admin['ML_Predicted_Category'].value_counts()
                            if not category_counts.empty:
                                fig_cat = px.pie(names=category_counts.index, values=category_counts.values, title='ML Predicted Category Distribution')
                                st.plotly_chart(fig_cat, use_container_width=True)
                            else: st.info("No category data to plot.")
                        else: st.warning("'ML_Predicted_Category' column not found.")
                        # Add more plots here...
                    else:
                        st.info("No data found in the user data table.")
                except pymysql.Error as db_err: st.error(f"DB error fetching admin data: {db_err}")
                except Exception as e: st.error(f"Error displaying admin data: {e}")
            else:
                st.warning("Database not connected. Cannot display admin data.")
        else:
            st.error("Invalid Admin Credentials.")

# ================================
# === HOME PAGE ==================
# ================================
else: # Default to Home
    st.title("📄 Welcome to the Smart Resume Analyzer!")
    st.markdown("---")
    st.header("Navigate using the sidebar:")
    st.markdown("""
    *   **👤 User Analysis:** Upload your resume(s) for ML-based categorization, skill extraction, scoring against job descriptions, and course recommendations.
    *   **🔑 Admin Panel:** View user upload statistics and analysis results (requires login).
    """)
    st.markdown("---")
    st.info("Select a mode from the sidebar on the left.")
    st.caption("Developed for Smart Resume Analysis")


# --- Close DB Connection ---
if db_enabled and connection is not None:
    try:
        connection.close()
    except Exception as e:
        st.sidebar.warning(f"Error closing DB connection: {e}")

# --- END OF FILE app.py ---
