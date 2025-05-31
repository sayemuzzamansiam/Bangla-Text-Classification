# Bangla MultiClass Text Classification


## Project Overview

This project undertakes a comprehensive exploration of various machine learning and deep learning techniques for **Bangla news text classification**. The primary objective is to accurately categorize news articles written in the Bangla language into one of four predefined classes: `entertainment`, `international`, `national`, and `sports`. 

To achieve this, the project implements, trains, and  evaluates six distinct models:
1.  **TF-IDF + Logistic Regression:** A classical machine learning baseline.
2.  **TF-IDF + Random Forest:** An ensemble learning baseline.
3.  **Keras LSTM with Embedding:** A recurrent neural network approach for sequence modeling.
4.  **Keras CNN with Embedding:** A convolutional neural network approach for local feature extraction in text.
5.  **BERT (bert-base-multilingual-cased) with TensorFlow/Keras:** A powerful pre-trained transformer model fine-tuned for the task.
6.  **BERT (sagorsarker/bangla-bert-base) with TensorFlow/Keras:** A Bangla-specific pre-trained transformer model, fine-tuned for enhanced language understanding.

The project emphasizes a standardized evaluation methodology on a held-out test set and includes feature interpretability analysis using LIME for the best-performing BERT model.

## 1. Dataset

### 1.1. Description

The dataset utilized for this project is a collection of Bangla news articles sourced from various categories, provided in the `Bangla_news.csv` file by SM Technology. This dataset is specifically curated for NLP tasks, particularly text classification in the Bangla language. Each entry consists of a news article's components and its assigned category.

### 1.2. File: `Bangla_news.csv`

The primary data source contains the following columns:
*   `title`: The headline or title of the news article.
*   `published_date`: The date and time when the article was published.
*   `reporter`: The name of the reporter or news agency (contains null values).
*   `category`: The predefined category of the news article (e.g., "sports", "international", "entertainment", "national"). This serves as the target label for classification.
*   `url`: The URL source of the article.
*   `content`: The main body or text content of the news article.

## Ydata Profiling - Interactive dataset report
[**Interactly Report View**](https://htmlpreview.github.io/?https://sayemuzzamansiam.github.io/Bangla-Text-Classification/)

## 2. Implementation Details

The entire project is developed within a Jupyter Notebook environment, utilizing Python and prominent libraries for data science, machine learning, and deep learning.

### 2.1. Data Loading and Exploratory Data Analysis (EDA):

*   The `Bangla_news.csv` dataset is loaded into a pandas DataFrame.
*   **Initial Inspection:** DataFrame shape, `info()`, and `head()` are used to understand the dataset's structure and content.
*   **Missing Value Analysis:** `isnull().sum()` is employed to identify columns with missing data. The "reporter" column is noted for a significant number of NaNs. Visualizations are generated to show the percentage of missing values.
*   **Target Variable Distribution:** The distribution of the `category` column is analyzed and visualized using a count plot to understand class balance. The dataset appears to be well-balanced across the four categories.
*   **Text Length Analysis:** The number of words in `title` and `content` columns are calculated and their distributions are visualized using histograms. This helps in determining appropriate sequence lengths for models like LSTM, CNN, and BERT. `content_len` is observed to have a wide distribution, with most articles having content up to 600 words.
*   **Duplicate Check:** Exact duplicates across all columns are identified and removed to ensure data integrity. Duplicates based on just `title` or `content` are also checked.
*   **Word Cloud Generation:** For the top 2 most frequent categories, word clouds are generated from the `content` field (after basic Bangla text cleaning) to visually represent prominent terms within each category. The `NotoSansBengali-Regular.ttf` font is utilized for correct Bangla character rendering.

### 2.2. Detailed Data Preprocessing Steps:

*   **Column Dropping:** Irrelevant columns for the classification task (`reporter`, `url`, `published_date`, and EDA-generated length columns) are dropped from the DataFrame.
*   **Handling Missing Values (Critical Data):** Rows where `content`, `title`, or `category` are missing are dropped to ensure model input quality.
*   **Feature Combination:**
    *   `raw_text`: Created by concatenating `title` and `content` with a `[SEP]` token (e.g., `title + " [SEP] " + content`). This version retains original punctuation and casing, intended for BERT models.
    *   `text_for_others`: Created by concatenating `title` and `content` with a simple space (e.g., `title + " " + content`). This version is then further cleaned for non-BERT models.
*   **Text Cleaning (for non-BERT models):**
    *   A function `clean_text_non_bert` is applied to the `text_for_others` column to produce `clean_text`. This function:
        *   Removes all characters except Bangla script (Unicode range `\u0980-\u09FF`) and whitespace.
        *   Replaces multiple whitespace characters with a single space.
        *   Strips leading/trailing whitespace.
*   **Label Encoding:** The textual `category` labels are converted into numerical labels (0, 1, 2, 3) using `sklearn.preprocessing.LabelEncoder`. A mapping from numerical labels back to original category names is stored.
*   **Train-Validation-Test Split:**
    *   The dataset (using `raw_text` for X_raw, `clean_text` for X_clean, and `label` for y) is split into:
        *   Training set: 70% of the data.
        *   Validation set: 15% of the data (derived from a temporary 30% split).
        *   Test set: 15% of the data (derived from a temporary 30% split).
    *   Stratified sampling based on the `label` is used in `train_test_split` to maintain class proportions across all sets.
    *   Target labels (`y_train`, `y_val`, `y_test`) are converted to NumPy arrays (`_np`) and further to one-hot encoded categorical format (`_cat`) for Keras models requiring it.

### 2.3. Feature Extraction:

*   **TF-IDF Vectorization (for Logistic Regression, Random Forest):**
    *   Applied to the `X_train_clean`, `X_val_clean` (for potential hyperparameter tuning, though not explicitly shown for these baselines), and `X_test_clean` sets.
    *   `sklearn.feature_extraction.text.TfidfVectorizer` is used.
    *   Parameters: `max_features=5000` (limits vocabulary size), `ngram_range=(1,2)` (considers unigrams and bigrams).
    *   The vectorizer is `fit_transform`ed on `X_train_clean` and then `transform`ed on `X_test_clean`.
*   **Keras Tokenization & Embedding (for LSTM, CNN):**
    *   Applied to `X_train_clean`, `X_val_clean`, and `X_test_clean`.
    *   `tensorflow.keras.preprocessing.text.Tokenizer` (aliased as `KerasTokenizer`) is used with `num_words=10000` (max vocabulary size) and an OOV token.
    *   Texts are converted to sequences of integers, and then padded to `maxlen=200` using `pad_sequences`.
    *   A Keras `Embedding` layer (input_dim=10000, output_dim=128, input_length=200) is the first layer in these models.
*   **Transformer Tokenization (for BERT models):**
    *   Applied to `X_train_raw`, `X_val_raw`, and `X_test_raw` (as BERT models benefit from raw text with punctuation and casing).
    *   `transformers.AutoTokenizer` (aliased as `HFTokenizer`) is used.
    *   Specific pre-trained tokenizers:
        *   `bert-base-multilingual-cased` for Model 5.
        *   `sagorsarker/bangla-bert-base` for Model 6.
    *   Parameters: `truncation=True`, `padding='max_length'`, `max_length=128` (or `MAX_LEN_BERT`), `return_tensors='tf'`.
    *   The output is a dictionary containing `input_ids` and `attention_mask` as TensorFlow tensors.

## 3. Model Selection and Training

Six models are trained using the prepared training and validation sets.

### 3.1. Traditional Models (Baselines)

Trained on TF-IDF features from `X_train_clean`.

*   **TF-IDF + Logistic Regression:**
    *   Model: `sklearn.linear_model.LogisticRegression` (`solver='liblinear'`, `C=1.0`, `max_iter=200`, `random_state=SEED`).
    *   Training: Fitted on `X_train_tfidf_lr` and `y_train`.
*   **TF-IDF + Random Forest:**
    *   Model: `sklearn.ensemble.RandomForestClassifier` (`n_estimators=100`, `random_state=SEED`).
    *   Training: Fitted on `X_train_tfidf_rf` and `y_train`.

### 3.2. Deep Learning Models (RNN/CNN)

Trained on Keras tokenized sequences from `X_train_clean` and one-hot encoded labels `y_train_cat`.

*   **LSTM (Long Short-Term Memory):**
    *   Architecture (`tensorflow.keras.Sequential`):
        1.  `Embedding(input_dim=10000, output_dim=128, input_length=200)`
        2.  `LSTM(128, dropout=0.2, recurrent_dropout=0.2)`
        3.  `Dense(64, activation='relu')`
        4.  `Dropout(0.3)`
        5.  `Dense(num_classes, activation='softmax')`
    *   Compilation: `optimizer='adam'`, `loss='categorical_crossentropy'`, `metrics=['accuracy']`.
    *   Training: `fit()` method with `epochs=5`, `batch_size=64`, using `(X_val_seq, y_val_cat)` for validation, and `EarlyStopping` (patience=2, restore_best_weights=True).
*   **CNN (Convolutional Neural Network):**
    *   Architecture (`tensorflow.keras.Sequential`):
        1.  `Embedding(input_dim=10000, output_dim=128, input_length=200)`
        2.  `Conv1D(filters=128, kernel_size=5, activation='relu')`
        3.  `GlobalMaxPooling1D()`
        4.  `Dense(64, activation='relu')`
        5.  `Dropout(0.3)`
        6.  `Dense(num_classes, activation='softmax')`
    *   Compilation: `optimizer='adam'`, `loss='categorical_crossentropy'`, `metrics=['accuracy']`.
    *   Training: `fit()` method with `epochs=10`, `batch_size=64`, using `(X_val_seq, y_val_cat)` for validation, and `EarlyStopping` (patience=2, restore_best_weights=True). A `ModelCheckpoint` is also used to save the best model as `cnn_best_model.h5`.

### 3.3. Transformer-Based Models

Fine-tuned using TensorFlow/Keras on tokenized sequences from `X_train_raw` and integer labels `y_train_np`.

*   **BERT (bert-base-multilingual-cased):**
    *   Model: `transformers.TFAutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_classes)`.
    *   Tokenizer: As specified. `max_length=128`.
    *   Optimizer: AdamW created using `transformers.create_optimizer` (`init_lr=3e-5`, warmup steps calculated based on training steps).
    *   Loss: `tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)`.
    *   Training: `fit()` method with `epochs=3` (or 5, user's notebook varies), `batch_size=16` (or `BATCH_SIZE_BERT`), using `(val_bert_multi_inputs, y_val_np)` for validation. `EarlyStopping` (patience=1 or 2) is used.
*   **BERT (sagorsarker/bangla-bert-base):**
    *   Model: `transformers.TFAutoModelForSequenceClassification.from_pretrained("sagorsarker/bangla-bert-base", num_labels=num_classes)`.
    *   Tokenizer: As specified. `max_length=128`.
    *   Optimizer: AdamW (similar setup to multilingual BERT).
    *   Loss: `tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)`.
    *   Training: `fit()` method with `epochs=3` (or 5), `batch_size=16`, using `(val_bert_bangla_inputs, y_val_np)` for validation. `EarlyStopping` (patience=1 or 2) is used.

## 4. Model Evaluation

### 4.1. Evaluation Metrics & Function
A standardized Python function, `evaluate_model_performance`, is defined and used for all models to ensure consistent evaluation on the **test set**. This function calculates and prints:
*   **Accuracy:** Overall correct predictions.
*   **Classification Report (String & Dictionary):** Precision, recall, F1-score, and support for each class, along with macro and weighted averages. `zero_division=0` is set.
It also generates and displays:
*   **Confusion Matrix:** A heatmap visualization of true vs. predicted labels.
*   **ROC AUC Score & Curves (One-vs-Rest):** For multi-class problems, if probabilities (`y_proba`) are provided. It binarizes labels, calculates per-class ROC AUC, plots individual ROC curves, and plots a macro-average ROC curve with its AUC.

The metrics (Accuracy, Macro F1, Weighted F1, Macro AUC, and the full report dictionary) for each model are stored in a `model_performance` dictionary.

### 4.2. Test Set Evaluation
Each of the six trained models is used to predict on its respective `X_test` (or `X_test_processed` / `X_test_enc`) and `y_test_np`. The `evaluate_model_performance` function is then called for each model.

## 5. Feature Interpretability

*   **LIME (Local Interpretable Model-agnostic Explanations):**
    *   Applied to the **best performing BERT model**, determined by comparing test set accuracy of "BERT (Multilingual)" and "BERT (Bangla-Specific)".
    *   A `predictor_lime_best_bert` function is defined that takes raw text, tokenizes it using the best BERT's tokenizer, and returns prediction probabilities from the best BERT model.
    *   `lime.lime_text.LimeTextExplainer` is initialized with the class names.
    *   LIME explains predictions for 2 random samples from the **raw test set (`X_test_raw`)**.
    *   For each sample, it prints the text, true label, model's predicted label (with probability), and then lists the top 10 words/tokens contributing to the predicted class with their LIME weights.
    *   A plot visualizing the LIME explanation is also generated for each sample, with attempts to use the Bangla font if available.

## 6. Results

### 6.1. Model Performance Comparison
The `model_performance` dictionary, containing test set metrics for all six models, is converted into a pandas DataFrame (`summary_performance_df`). This DataFrame includes:
*   Model Name
*   Accuracy
*   Macro Precision
*   Macro Recall
*   Macro F1-Score
*   Weighted F1-Score
*   Macro AUC

The DataFrame is sorted by Accuracy in descending order and printed.

### 6.2. Performance Visualizations
Bar plots are generated using `seaborn.barplot` to visually compare:
*   Model Accuracy on Test Set
*   Model Macro F1-Score on Test Set
*   Model Macro AUC on Test Set (for models where AUC is applicable and calculated)
These plots help in quickly identifying the relative performance of the different models.

### 6.3. Performance Table


| Model                                  | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :------------------------------------- | :------: | :-------------: | :----------: | :------------: | :---------------: |
| BERT (Bangla-Specific)                 |  0.9703  |     0.9703      |    0.9703    |     0.9703     |      0.9703       |
| BERT (Multilingual)                    |  0.9574  |     0.9576      |    0.9575    |     0.9574     |      0.9574       |
| Keras CNN with Embedding               |  0.9563  |     0.9564      |    0.9563    |     0.9563     |      0.9563       |
| Keras LSTM with Embedding              |  0.9446  |     0.9452      |    0.9446    |     0.9447     |      0.9447       |
| TF-IDF + Logistic Regression           |  0.9177  |     0.9184      |    0.9177    |     0.9177     |      0.9177       |
| TF-IDF + Random Forest                 |  0.9037  |     0.9046      |    0.9037    |     0.9035     |      0.9035       |

## 7. Usage

### 7.1. Key Libraries Used:
*   **Data Handling & Numerics:** `pandas`, `numpy`
*   **Visualization:** `matplotlib`, `seaborn`, `wordcloud`
*   **Text Processing:** `re` (regular expressions)
*   **Traditional ML:** `scikit-learn` (for `train_test_split`, `LabelEncoder`, all evaluation metrics, `TfidfVectorizer`, `LogisticRegression`, `RandomForestClassifier`, `Pipeline`)
*   **Deep Learning (RNN/CNN):** `tensorflow.keras` (for `Sequential` model, `Embedding`, `LSTM`, `Conv1D`, `GlobalMaxPooling1D`, `Dense`, `Dropout` layers, `Tokenizer`, `pad_sequences`, `to_categorical`, `EarlyStopping`, `ModelCheckpoint`)
*   **Transformers (BERT):** `transformers` (Hugging Face library for `AutoTokenizer`, `TFAutoModelForSequenceClassification`, `create_optimizer`)
*   **Interpretability:** `lime`

### 7.2. Environment
The project is developed and intended to be run in a Kaggle Notebook environment with GPU acceleration (e.g., Tesla T4) enabled for efficient training of deep learning and transformer models.

### 7.3. Steps to Run the Code
1.  **Data Setup:**
    *   Upload `Bangla_news.csv` to Kaggle and make it available as an input dataset (e.g., at `/kaggle/input/bangladata/Bangla_news.csv`).
    *   Upload the Bangla font file `NotoSansBengali-Regular.ttf` (e.g., at `/kaggle/input/banglafront/NotoSansBengali-Regular.ttf`). Update paths in the notebook if different.
2.  **Notebook Execution:**
    *   Open the `bangla-multi-class-tex.ipynb` notebook in the Kaggle environment.
    *   Ensure the kernel is set to use Python 3 and a GPU accelerator.
    *   Execute all cells sequentially from top to bottom using "Run All".
    *   The notebook will perform:
        *   Library imports and setup.
        *   Data loading and comprehensive EDA.
        *   Data preprocessing, including text cleaning and feature engineering.
        *   Train-validation-test splitting.
        *   Training and evaluation (on the test set) of all six specified models. Output for each model's evaluation (reports, CM, ROC curves) will be displayed.
        *   LIME feature interpretability for the best performing BERT model.
        *   Generation of a final summary table and comparative plots for all models.
3.  **Review Results:**
    *   Observe the printed metrics and generated plots for each model.
    *   Analyze the final comparison table and plots to understand relative model performances.
    *   Examine LIME explanations for insights into BERT model decisions.

## 8. Conclusion

A comprehensive analysis of six different models was conducted for Bangla news text classification. Based on the evaluation metrics from the held-out test set:

*   The **[BERT (Bangla-Specific)]** demonstrated the highest performance with an Accuracy of **[0.9703 ]**, Macro F1-Score of **[0.970305 ]**, and Macro AUC of **[0.9]**.
*   Transformer-based models (BERT Multilingual and Bangla-Specific) generally outperformed traditional machine learning and RNN/CNN approaches, showcasing their strength in understanding nuanced linguistic features.
*   The [CNN/LSTM] model also achieved notable results with an Accuracy of **[0.9563]**, indicating the utility of [convolutional/recurrent] architectures for this task.
*   The TF-IDF based baselines, Logistic Regression and Random Forest, provided a solid performance floor, with accuracies of **[0.9177]** and **[0.9037]** respectively.
*   LIME analysis on the **[BERT (Bangla-Specific]** provided valuable insights into its decision-making process by highlighting salient words for each predicted category.


