import streamlit as st
import os
import shutil
import whisper
from dotenv import load_dotenv
from langchain_sambanova import ChatSambaNovaCloud
from langchain.prompts import ChatPromptTemplate
import markdown

# Load environment variables
load_dotenv()
os.environ["SAMBANOVA_API_KEY"] = st.secrets["SAMBANOVA_API_KEY"]

# Load Whisper model
whisper_model = whisper.load_model("base")

# Initialize SambaNova LLM
llm = ChatSambaNovaCloud(
    model="Llama-3.1-Tulu-3-405B",
    temperature=0.8,
    max_tokens=2048,
)

# Define medical report prompt
medical_report_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional medical assistant specializing in structuring medical reports.
Given a raw transcription of a clinical consultation, analyze and extract key medical insights.

### **Patient Summary Report**
#### **1. Patient Information:**
- Name (if mentioned):
- Age (if mentioned):
- Gender (if mentioned):
- Chief Complaint:

#### **2. Present Illness:**
- Symptoms:
- Duration:
- Aggravating/Relieving Factors:
- Relevant Medical History:

#### **3. Physical Examination (if discussed):**
- Vitals:
- Physical Findings:

#### **4. Diagnosis (if discussed):**
- Possible Conditions Considered:
- Suggested Diagnostic Tests:

#### **5. Treatment Plan:**
- Medications Prescribed:
- Lifestyle Recommendations:
- Follow-up Instructions:

#### **6. Additional Notes:**
- Other Observations:

---
If any information is missing in the transcription, explicitly state "**Not mentioned**".
Ensure the report is **concise, structured, and medically relevant**.
Maintain **professional medical language** while ensuring readability.

---
Transcription:
"{transcription}"
"""
    )
])

# Streamlit UI
st.set_page_config(page_title="AI Medical Report", layout="wide")
st.title("🩺 AI Medical Report Generator")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file:
    file_path = f"temp_{uploaded_file.name}"
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file, buffer)

    # Transcribe the audio using Whisper
    with st.spinner("Transcribing audio..."):
        transcription = whisper_model.transcribe(file_path)["text"]
    
    # Generate structured report using SambaNova LLM
    with st.spinner("Generating medical report..."):
        prompt = medical_report_prompt.format(transcription=transcription)
        response = llm.invoke(prompt)
        summary_text = response.content

    # Convert to markdown format
    markdown_summary = f"# Medical Report\n\n{summary_text}"
    
    # Store in session state
    st.session_state["medical_summary"] = markdown_summary

    # Display Summary
    st.subheader("📝 Generated Medical Report")
    st.markdown(markdown_summary, unsafe_allow_html=True)

    # Provide Markdown file download
    md_filename = "medical_summary.md"
    with open(md_filename, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_summary)

    st.download_button(
        label="📥 Download Report",
        data=markdown_summary,
        file_name="medical_summary.md",
        mime="text/markdown",
    )

    # Cleanup temp file
    os.remove(file_path)
