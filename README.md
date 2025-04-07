# Student Summary Evaluation Project
## 🛠️ Tools & Technologies Used

### Programming Language
- <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" height="20"> **Python 3**

### Data Processing
- <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" height="20"> **Pandas** (Data manipulation and analysis)
- <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" height="20"> **NumPy** (Numerical computing)

### Machine Learning
- <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white" height="20"> **Scikit-learn** (Linear Regression, Random Forest)
- <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white" height="20"> **TensorFlow** (Optional for neural network approaches)
- <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" height="20"> **PyTorch** (Optional for advanced modeling)

### NLP Libraries
- <img src="https://img.shields.io/badge/NLTK-008000?logo=python&logoColor=white" height="20"> **NLTK** (Basic text processing)
- <img src="https://img.shields.io/badge/SpaCy-09A3D5?logo=spacy&logoColor=white" height="20"> **SpaCy** (Advanced NLP features)

### Visualization
- <img src="https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white" height="20"> **Matplotlib** (Basic plotting)
- <img src="https://img.shields.io/badge/Seaborn-5B8AC6?logo=python&logoColor=white" height="20"> **Seaborn** (Statistical visualizations)

## Project Description
This project develops a machine learning system to automatically evaluate the quality of student-written summaries by predicting content and wording scores. The system analyzes linguistic features from student summaries and their corresponding prompts to assess writing quality.

## Dataset
Used the Kaggle dataset: [CommonLit Evaluate Student Summaries](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries)
- `summaries_train.csv`: Contains student summaries with content/wording scores
- `prompts_train.csv`: Contains original prompt texts that students summarized

## Workflow

### 1. Data Preparation
- Loaded and merged the summaries and prompts datasets
- Cleaned text data by removing special characters and normalizing whitespace
- Verified data integrity and handled missing values

### 2. Feature Engineering
Created 12 key features from the text data:
1. Basic counts:
   - `word_count`: Number of words in student response
   - `word_count_prompt`: Number of words in prompt text
   - `word_distinct`: Unique words in response
   - `word_distinct2`: Unique words in prompt

2. Overlap features:
   - `word_common`: Words shared between response and prompt text
   - `word_common2`: Words shared with prompt question
   - `word_common3`: Words shared with prompt title

3. Linguistic features:
   - `average word length`: Mean character count per word
   - `stopword_count`: Common words (the, and, etc.)
   - `punctuation_count`: Number of punctuation marks
   - `nouns counted`: Noun frequency
   - `conjunction_count`: Connecting words frequency

### 3. Modeling Approach

#### Model 1: Linear Regression
- Trained separate models for content and wording scores
- Used all 12 engineered features
- Evaluated using Mean Squared Error (MSE)

#### Model 2: Random Forest Regressor
- Implemented ensemble method to capture non-linear relationships
- Used same feature set as linear regression
- Set random_state=23 for reproducibility

### 4. Model Evaluation
- Split data 80/20 for training/testing
- Compared performance using MSE

### 5. Results Analysis
- Random Forest outperformed Linear Regression
- Feature importance analysis showed word overlap features were most significant
- Content scores were slightly easier to predict than wording scores

## Key Findings
1. The engineered features effectively captured summary quality aspects
2. Non-linear models (Random Forest) better handled text complexity
3. Prompt-response word overlaps were strong quality indicators
4. Simple linguistic features provided meaningful signals for scoring

## Future Improvements
- Experiment with neural network approaches
- Add more sophisticated NLP features (sentence structure, coherence)
- Develop an ensemble of specialized models
- Create interpretation tools to explain scores to students
