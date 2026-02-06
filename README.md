# HR Analytics Chatbot

An intelligent **HR Analytics Chatbot** that allows users to explore and analyze HR data through natural language questions.  
The chatbot supports both **Cloud-based** and **Local** Large Language Models (LLMs) and connects directly to structured HR data using a **Text-to-SQL** approach.

---

## 🎯 Project Objective

The goal of this project is to transform traditional, static HR data analysis into an **interactive conversational experience**.

Users can ask questions in **plain English or Arabic**, and the chatbot dynamically generates accurate answers by querying the underlying HR dataset.

This project demonstrates:

- Integration of LLMs with structured data  
- Practical use of Text-to-SQL  
- Comparison between Cloud-based and Local AI models  
- Professional AI application development standards  

---

## 📂 Dataset

- **Dataset Name:** IBM HR Analytics Employee Attrition & Performance  
- **File:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`  

**Description:**
- Employee demographic information  
- Job roles and departments  
- Attrition status  
- Performance-related attributes  

The dataset is stored locally and queried dynamically using **SQLite**.

---

## 🛠️ Environment & Tools

- **Environment Management:** Conda  
- **Programming Language:** Python  
- **Framework:** Streamlit  
- **Data Processing:** Pandas  
- **Database:** SQLite  

**AI Models:**
- **Cloud Model:** Groq API (High-speed inference)  
- **Local Model:** Qwen 1.5B (Runs locally on CPU)  

**Model Integration:**
- Text-to-SQL  
- No RAG in the final system flow  

---

## 🧠 System Architecture

1. The user submits a question through the chat interface.
2. The selected LLM (Cloud or Local) converts the question into a SQL query.
3. The generated SQL query is executed on the SQLite database.
4. Query results are retrieved and formatted into a conversational response.
5. Chat history is preserved within the session to support follow-up questions.

---

## 🔁 Dual Model Support

### ☁️ Cloud-Based Model (Groq)

- Faster response time  
- Suitable for complex and multi-step analytical questions  
- Requires API key configuration  

### 💻 Local Model (Qwen 1.5B)

- Runs fully offline  
- Privacy-friendly  
- Slightly slower but reliable for core HR analytics  

Users can switch between models directly from the sidebar.

---

## 💬 Supported Question Types

- Employee counts and summaries  
- Attrition rates by department  
- Department comparisons  
- Follow-up analytical questions  
- Arabic and English queries  

**Example Questions:**
- *How many employees are there?*  
- *Attrition rate by department*  
- *Which department has the highest attrition?*  
- *ما هو القسم الأعلى في نسبة الاستقالات؟*  

---

## 🗂️ Project Structure

```text
HR_Analytics_Chatbot/
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── styles.css                 # Custom UI styling
├── logo.png                   # Application logo
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── screenshots/               # Application screenshots
│
└── src/
    ├── llm/
    │   ├── cloud_groq.py      # Cloud LLM integration
    │   ├── local_qwen.py      # Local Qwen model
    │   ├── sql_agent.py       # Text-to-SQL logic
    │   └── prompt.py          # Prompt templates
    │
    ├── data_loader.py         # Data loading utilities
    └── config.py              # Configuration settings

---

## 📌 Professional Practices Followed

- Small, meaningful Git commits  
- Clean project structure  
- Clear separation between Cloud and Local models  
- README documentation with screenshots  
- Reproducible environment via `requirements.txt`  

---

## ☁️ Cloud vs Local Model Comparison

| Aspect             | Cloud (Groq) | Local (Qwen 1.5B) |
|-------------------|--------------|-------------------|
| Speed             | Very Fast    | Moderate          |
| Internet Required | Yes          | No                |
| Privacy           | Lower        | Higher            |
| Resource Usage    | Low          | Higher (CPU)      |

---

## 🖼️ Application Screenshots

### Home Screen – Cloud Mode
![Home Cloud](screenshots/home-cloud.png)

### Home Screen – Local Mode
![Home Local](screenshots/home-local.png)

### Total Employees Query
![Total Employees](screenshots/total_employees.png)

### Core Q&A Flow – Cloud Model
![Core QA Flow Cloud](screenshots/core_qa_flow-cloud.png)

### Cloud Model – Analysis Results
![Cloud Model Results](screenshots/cloud_model_results.png)

### Cloud Model – Advanced Analysis
![Cloud Advanced Analysis](screenshots/cloud_advanced_analysis.png)

### Local Model – Analysis Results
![Local Result 1](screenshots/local_result1.png)  
![Local Result 2](screenshots/local_result2.png)

