# 📰 Fake News Detector

An end-to-end machine learning application for classifying news articles as **Real** or **Fake** using Natural Language Processing (NLP), TF-IDF feature extraction, and supervised machine learning.

The project covers the complete machine learning lifecycle — from raw data processing and exploratory data analysis to model training, leakage-aware evaluation, hyperparameter tuning, error analysis, and deployment through a **FastAPI backend** and **Streamlit frontend**.

---

## 📌 Project Overview

The goal of this project is to develop a machine learning system that can classify news articles as likely **Real** or **Fake** based on their textual content.

Rather than stopping at model training in a notebook, this project follows a complete machine learning engineering workflow:

```text
Raw News Data
      │
      ▼
Data Loading
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Text Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Baseline Model Training
      │
      ▼
Leakage-Free Evaluation
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Error Analysis
      │
      ▼
Model Serialization
      │
      ▼
FastAPI Inference API
      │
      ▼
Streamlit Frontend

✨ Key Features
🧹 Comprehensive text preprocessing
🔤 Natural Language Processing using NLTK
📊 Exploratory Data Analysis
🧮 TF-IDF feature extraction
🤖 Comparison of multiple machine learning algorithms
🔍 Leakage-aware data evaluation
⚙️ Hyperparameter tuning using cross-validation
🧪 Detailed model evaluation
🔎 Error analysis of misclassified articles
💾 Serialized production model and TF-IDF vectorizer
🚀 FastAPI inference API
🖥️ Streamlit frontend
✅ Automated testing with Pytest
📁 Structured ML project architecture
📂 Dataset

The project uses a collection of real and fake news articles.

The raw dataset contains two source files:
data/
└── raw/
    ├── Fake.csv
    └── True.csv
The processed datasets are stored under:
data/
└── processed/
Primary Dataset Fields
| Column    | Description          |
| --------- | -------------------- |
| `title`   | News article title   |
| `text`    | News article content |
| `subject` | News category/topic  |
| `date`    | Publication date     |
| `label`   | Target class         |

Target Variable
| Label | Meaning |
| ----: | ------- |
|   `0` | Real    |
|   `1` | Fake    |

🔍 Exploratory Data Analysis
The initial dataset contained approximately 44,680 records after the relevant loading and preparation stages.

Exploratory analysis investigated several aspects of the dataset, including:

Dataset dimensions
Missing values
Duplicate records
Class distribution
Subject/category distribution
Relationship between subject and class
Publication year
Publication month
Text length
Word count
Title length

The analysis also identified duplicate and overlapping texts.

These overlaps were investigated before the final modeling workflow because duplicate texts appearing in both training and testing datasets can lead to data leakage and overly optimistic model performance.

🧹 Data Preprocessing

A dedicated text preprocessing pipeline was developed to prepare news articles for machine learning.

The cleaning process includes:

Converting text to lowercase
Removing URLs
Removing HTML tags
Removing punctuation
Removing numbers
Tokenizing the text
Removing English stopwords
Lemmatizing words

The same preprocessing logic is used during both model training and production inference.

This ensures that the model receives text in a consistent format during development and deployment.

🧮 Feature Engineering

Several text-related features were investigated during the feature engineering stage, including:

Text length
Word count
Title length
Publication month
Publication year

For the final text classification pipeline, TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert cleaned news articles into numerical feature vectors.

The final TF-IDF representation contained:
100,000 features
The fitted vectorizer was serialized and saved as:
models/tfidf_vectorizer.pkl

🤖 Model Development

Four machine learning algorithms were trained and compared:

Linear Support Vector Machine
Random Forest
Logistic Regression
Naive Bayes

The models were evaluated using:

Accuracy
Precision
Recall
F1 Score
📊 Baseline Model Performance
The baseline evaluation produced the following results:
🤖 Model Development

Four machine learning algorithms were trained and compared:

Linear Support Vector Machine
Random Forest
Logistic Regression
Naive Bayes

The models were evaluated using:

Accuracy
Precision
Recall
F1 Score
📊 Baseline Model Performance

The baseline evaluation produced the following results:
| Model               |     Accuracy |    Precision |       Recall |     F1 Score |
| ------------------- | -----------: | -----------: | -----------: | -----------: |
| **Linear SVM**      | **0.997314** | **0.998320** | **0.995810** | **0.997063** |
| Random Forest       |     0.993222 |     0.996621 |     0.988547 |     0.992568 |
| Logistic Regression |     0.992071 |     0.996332 |     0.986313 |     0.991297 |
| Naive Bayes         |     0.966364 |     0.966264 |     0.960056 |     0.963150 |

Baseline Conclusion

The Linear SVM achieved the strongest overall baseline performance, with an F1 score of approximately 99.71%.

It was therefore selected for further optimization through hyperparameter tuning.

🔐 Leakage-Aware Evaluation

During the project, duplicate and overlapping news texts were specifically investigated.

The analysis initially identified overlapping texts between the training and testing datasets.

To obtain a more reliable evaluation, the dataset was deduplicated before the final modeling workflow.

The resulting dataset contained:
Baseline Conclusion

The Linear SVM achieved the strongest overall baseline performance, with an F1 score of approximately 99.71%.

It was therefore selected for further optimization through hyperparameter tuning.

🔐 Leakage-Aware Evaluation

During the project, duplicate and overlapping news texts were specifically investigated.

The analysis initially identified overlapping texts between the training and testing datasets.

To obtain a more reliable evaluation, the dataset was deduplicated before the final modeling workflow.

The resulting dataset contained:
Original shape:       (44680, 8)
Deduplicated shape:   (39094, 8)
Remaining duplicates: 0

Class Distribution
Real:  21,194
Fake:  17,900

It corresponds approximately:
Real:  54.21%
Fake:  45.79%

Final Train/Test Split
Training samples: 31,275
Testing samples:   7,819

The exact overlap: 0
between the final training and testing datasets.

The resulting TF-IDF matrices were:
X_train: (31275, 100000)
X_test:  (7819, 100000)

This leakage-aware evaluation provides a more realistic assessment of the model's ability to generalize to unseen news articles.

⚙️ Hyperparameter Tuning

After identifying Linear SVM as the strongest baseline model, hyperparameter tuning was performed using cross-validation.

The best configuration was:
c = 100
The best cross-validation F1 score was:
0.996781
The tuned model was then evaluated on the held-out test set.

🏆 Final Tuned Model Performance
The tuned Linear SVM achieved:
| Metric        |        Score |
| ------------- | -----------: |
| **Accuracy**  | **0.997698** |
| **Precision** | **0.998600** |
| **Recall**    | **0.996369** |
| **F1 Score**  | **0.997483** |
The tuned Linear SVM was selected as the final production model.

💾 Final Model Artifacts

The final fine-tuned model from Notebook 07 was serialized using Joblib.

Final Model
models/fake_news_svm.pkl
TF-IDF Vectorizer
models/tfidf_vectorizer.pkl
The production inference pipeline loads both artifacts and applies the same preprocessing and feature transformation used during model development.

🔎 Error Analysis

The final model was evaluated on:
7,819 test samples
it produced:
18 misclassified samples
The resulting error rates was approximately:
0.23%
Misclassification Breakdown
| Error Type  |  Count |
| ----------- | -----: |
| Real → Fake |      5 |
| Fake → Real |     13 |
| **Total**   | **18** |

The misclassified articles were further analyzed using:

Article text
Actual label
Predicted label
Decision score
Word count
Text length

The analysis showed that many of the incorrectly classified articles had decision scores relatively close to the model's classification boundary.

This demonstrates that even a high-performing classifier can encounter ambiguous or difficult examples.

🧠 Production Inference Pipeline

The production inference pipeline follows this process:
User Input
    │
    ▼
Input Validation
    │
    ▼
Text Cleaning
    │
    ▼
TF-IDF Transformation
    │
    ▼
Fine-Tuned Linear SVM
    │
    ▼
Prediction
    │
    ▼
Decision Score
    │
    ▼
JSON Response

The prediction function validates incoming text before making a prediction.

It rejects:

Non-string input
Empty text
Text that contains no usable content after preprocessing
Example Prediction
Input
{
  "text": "The government announced a new policy today."
}
Example Response:
{
  "prediction": "Fake",
  "label": 1,
  "decision_score": 0.2256300535945467
}
Where:

prediction is the human-readable prediction
label is the numerical class
decision_score is the Linear SVM decision function output

Note: The decision score should be interpreted as a model signal rather than proof that an article is factually true or false.

🚀 FastAPI Backend

The project includes a FastAPI backend that exposes the trained model through a REST API.

API Endpoints
Root Endpoint

GET /

Example response:

{
  "message": "Fake News Detector API is running"
}
Health Check
GET /health

Example response:

{
  "status": "healthy"
}
Prediction Endpoint
POST /predict

Request:

{
  "text": "The government announced a new policy today."
}

Response:

{
  "prediction": "Fake",
  "label": 1,
  "decision_score": 0.2256300535945467
}

The API also handles invalid input and returns appropriate HTTP error responses.

🖥️ Streamlit Frontend

A Streamlit frontend provides a user-friendly interface for submitting news articles and receiving predictions.

The frontend communicates with the FastAPI backend rather than directly loading the machine learning model.

This creates a clean separation between:

User interface
API
Inference logic
Machine learning artifacts
Application Architecture

┌─────────────────────┐
│ Streamlit Frontend  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   FastAPI Backend   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Inference Service  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  TF-IDF Vectorizer  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Fine-Tuned Linear  │
│         SVM         │
└─────────────────────┘

🧪 Testing

Automated tests were implemented using Pytest.

The test suite currently covers:

Predictor output validation
Empty predictor input
API prediction endpoint
API empty-input handling

The final test suite successfully passed all four tests:4 passed

📁 Project Structure
Fake-Real-News-Project/
│
├── .vscode/
│   └── settings.json
│
├── app/
│   ├── __init__.py
│   │
│   ├── backend/
│   │   ├── .gitkeep
│   │   └── main.py
│   │
│   └── frontend/
│       ├── .gitkeep
│       ├── app.py
│       ├── assets/
│       └── pages/
│
├── config/
│   ├── config.py
│   └── logging.yaml
│
├── data/
│   ├── README.md
│   │
│   ├── interim/
│   │   └── .gitkeep
│   │
│   ├── processed/
│   │   ├── further_cleaned_news.csv
│   │   ├── news.csv
│   │   ├── news_cleaned.csv
│   │   └── news_deduplicated.csv
│   │
│   └── raw/
│       ├── Fake.csv
│       └── True.csv
│
├── docs/
│   ├── api_design.md
│   ├── architecture.md
│   └── model_card.md
│
├── models/
│   ├── fake_news_svm.pkl
│   ├── linear_svm.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_hyperparameter_tuning.ipynb
│
├── reports/
│   ├── eda_summary.md
│   ├── model_comparison.csv
│   ├── model_comparison_leakage_free.csv
│   ├── model_evaluation.csv
│   ├── model_report.md
│   └── figures/
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   └── __init__.py
│   │
│   ├── features/
│   │   └── __init__.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py
│   │
│   ├── models/
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   └── __init__.py
│   │
│   └── visualization/
│       └── __init__.py
│
├── tests/
│   ├── test_api.py
│   └── test_predictor.py
│
├── .gitignore
├── pyproject.toml
└── README.md

🛠️ Technologies Used
| Category            | Technologies                                                |
| ------------------- | ----------------------------------------------------------- |
| Programming         | Python                                                      |
| Data Science        | Pandas, NumPy, Scikit-learn                                 |
| NLP                 | NLTK, TF-IDF                                                |
| Machine Learning    | Linear SVM, Logistic Regression, Random Forest, Naive Bayes |
| Backend             | FastAPI, Uvicorn, Pydantic                                  |
| Frontend            | Streamlit                                                   |
| Model Serialization | Joblib                                                      |
| Testing             | Pytest                                                      |
| Development         | Jupyter Notebook, VS Code, Git, GitHub                      |


⚠️ Limitations

This project should be treated as a machine learning classification system, not as a definitive fact-checking system.

A prediction of Fake does not prove that an article is false.

Likewise, a prediction of Real does not guarantee that an article is factually correct.

The model learns statistical patterns from its training data and may perform differently on:

New topics
Different news sources
Different writing styles
Different time periods
News formats that differ significantly from the training data
Articles containing information outside the patterns learned during training

The decision score should therefore be interpreted as a model output signal rather than factual certainty.

🔮 Future Improvements

Potential future improvements include:

Testing the model on newer and independently sourced news datasets
Evaluating performance on completely external datasets
Adding model explainability
Improving uncertainty and confidence reporting
Adding prediction logging
Adding model performance monitoring
Containerizing the application with Docker
Adding CI/CD using GitHub Actions
Deploying the API to a cloud platform
Deploying the Streamlit frontend
Adding automated model and data monitoring
Expanding the automated test suite
Improving handling of extremely short or low-information inputs
📈 Project Status
✅ Completed ML Prototype — API and Frontend Integrated

The project currently includes:

 Data loading
 Exploratory data analysis
 Missing-value investigation
 Duplicate detection
 Data preprocessing
 Text cleaning
 Feature engineering
 TF-IDF feature extraction
 Baseline model training
 Model comparison
 Leakage investigation
 Leakage-aware evaluation
 Hyperparameter tuning
 Final model selection
 Error analysis
 Model serialization
 FastAPI inference API
 Streamlit frontend
 Automated testing
 Project documentation
👨‍💻 Author
Aniude Paul Ifeanyi

Statistics | Data Science | Machine Learning | NLP

This project was developed as an end-to-end machine learning engineering project demonstrating the transition from statistical analysis and machine learning experimentation to a functional prediction system.

📄 Disclaimer

This application provides machine learning predictions, not verified fact-checking.

Predictions should be treated as model-generated assessments and should not be considered definitive evidence that a news article is true or false.


