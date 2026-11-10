# Twitter Sentiment Analysis Using GBM and AdaBoost

## Project Overview

This project focuses on classifying Twitter posts into positive and negative sentiments using two ensemble machine learning algorithms:

- Gradient Boosting Machine (GBM)
- AdaBoost Classifier

The objective is to preprocess tweet text, convert it into numerical representations using TF-IDF vectorization, train both classifiers, and compare their performance.

The project also explores hyperparameter tuning using GridSearchCV to improve model performance.

## Dataset

The project uses a Twitter sentiment analysis dataset containing tweets labeled with sentiment values:

- **0:** Negative sentiment
- **4:** Positive sentiment

The original dataset contains approximately 1.6 million records.

Due to the computational cost of processing the complete dataset, a smaller sample of 2,000 records was selected for this experiment:

- 1,000 records from the beginning of the dataset
- 1,000 records from the end of the dataset

**Dataset Access:** Add the original dataset source link here if available.

## Technologies and Libraries

The project uses the following tools and libraries:

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- Google Colab

## Methodology

### 1. Data Sampling

The original dataset was relatively large, containing approximately 1.6 million tweets. To reduce preprocessing time and computational requirements, 2,000 records were selected for the experiment.

### 2. Text Preprocessing

Basic text-cleaning techniques were applied to prepare the tweets for classification:

- Converting text to lowercase
- Removing URLs
- Removing special characters and numbers
- Tokenizing text
- Removing English stopwords

The original sentiment labels were also converted from 0 and 4 into binary labels:

- 0: Negative
- 1: Positive

### 3. Train-Test Split

The processed dataset was divided into training and testing subsets using an 80:20 split with a fixed random state of 42.

### 4. TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) was used to transform the cleaned tweet text into numerical feature vectors.

The vectorizer was configured with a maximum of 5,000 features.

### 5. Model Training

Two ensemble learning classifiers were trained:

**Gradient Boosting Machine (GBM)**

GBM builds an ensemble of decision trees sequentially, with each new tree attempting to improve upon the errors of the previous model.

**AdaBoost Classifier**

AdaBoost combines multiple weak learners by assigning greater importance to incorrectly classified training examples during successive learning rounds.

### 6. Model Evaluation

Both classifiers were evaluated using:

- Accuracy
- F1-score

### 7. Hyperparameter Tuning

GridSearchCV with 3-fold cross-validation was used to identify better hyperparameter configurations.

The following parameters were explored:

**GBM:**
- Number of estimators: 50, 100, 200
- Learning rate: 0.05, 0.1, 0.2
- Maximum depth: 3, 4, 5

**AdaBoost:**
- Number of estimators: 50, 100, 200
- Learning rate: 0.05, 0.1, 0.2

## Results and Model Comparison

The initial model evaluation produced the following results:

| Model | Initial Accuracy | Initial F1-score |
|---|---:|---:|
| GBM | 66.50% | 0.7048 |
| AdaBoost | 60.25% | 0.6851 |

After hyperparameter tuning using GridSearchCV, the results improved:

| Model | Tuned Accuracy | Tuned F1-score |
|---|---:|---:|
| GBM | 68.00% | 0.7051 |
| AdaBoost | 69.25% | 0.7248 |

### Best Hyperparameter Configurations

**GBM:**
- Maximum depth: 5
- Number of estimators: 200
- Learning rate: Default value (0.1)

**AdaBoost:**
- Learning rate: 0.2
- Number of estimators: 200

### Results Discussion

Initially, GBM performed better than AdaBoost in terms of both accuracy and F1-score.

However, after hyperparameter tuning, AdaBoost achieved better results:

- GBM accuracy increased from 66.50% to 68.00%.
- AdaBoost accuracy increased from 60.25% to 69.25%.
- AdaBoost also achieved the highest tuned F1-score of 0.7248.

These results indicate that hyperparameter tuning improved both models, with AdaBoost showing a greater improvement in this experiment.

## Challenges and Limitations

### 1. Large Dataset Size

The original dataset contained approximately 1.6 million records. Processing the entire dataset was time-consuming, so a smaller sample of 2,000 records was used.

### 2. Limited Training Data

Using only 2,000 tweets limits the amount of language variation and sentiment patterns available to the models. Results obtained from this sample may not represent performance on the complete dataset.

### 3. Text Representation

TF-IDF represents text through word-based numerical features. However, it may not fully capture context, sarcasm, or the relationships between words.

### 4. Model Performance

Both classifiers achieved moderate accuracy. Further experiments with larger samples, alternative text representations, and additional model configurations could help improve performance.

## Conclusion and Future Improvements

This project demonstrates a sentiment classification workflow using text preprocessing, TF-IDF vectorization, ensemble learning, and hyperparameter tuning.

Both GBM and AdaBoost showed improved performance after GridSearchCV. Although GBM initially performed better, AdaBoost achieved higher accuracy and F1-score after tuning.

Possible future improvements include:

- Experimenting with larger and more representative dataset samples.
- Exploring alternative text vectorization techniques.
- Testing additional machine learning classifiers.
- Applying more advanced text preprocessing methods.
- Exploring deep learning approaches for sentiment classification.
- Evaluating models using precision, recall, and confusion matrices.

## Repository Contents

```text
Twitter-Sentiment-Analysis-GBM-AdaBoost/
│
├── Twitter_Sentiment_Analysis.ipynb
├── README.md
├── DATASET.md
└── requirements.txt
```

The Jupyter Notebook contains the complete implementation, preprocessing steps, model training, hyperparameter tuning, and experimental outputs.
