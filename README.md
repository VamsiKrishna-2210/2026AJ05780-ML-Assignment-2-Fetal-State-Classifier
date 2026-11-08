# Fetal state classifier

Classifying fetal state from cardiotocography recordings.

GitHub repository: https://github.com/VamsiKrishna-2210/Fetal-State-Classifier-ML-Assignment-2

## a. Problem statement

A cardiotocogram is the tracing taken during pregnancy of the fetal heart rate and the mother's contractions. Each row in this dataset is one tracing, reduced to 21 numbers: the baseline heart rate, how often the rate accelerates or decelerates, the short-term and long-term variability, and the shape of the heart-rate histogram. From those numbers I predict one of three states, Normal, Suspect or Pathologic, so a Suspect or Pathologic tracing can be looked at first. I trained logistic regression, a decision tree, k-nearest neighbours, Gaussian naive Bayes and a support vector machine on the same stratified split, and compared them with accuracy, AUC, precision, recall, F1 and MCC.

## b. Dataset description

| Item | Value |
| --- | --- |
| Source | UCI Machine Learning Repository, Cardiotocography |
| Instances | 2,114. The UCI spreadsheet has 2,126 recorded tracings; 3 rows were empty and 12 were duplicates, and I removed both |
| Features | 21 numeric. Examples: `LB` (baseline heart rate), `AC` (accelerations), `DP` (prolonged decelerations), `ASTV` and `ALTV` (percent of time with abnormal short-term and long-term variability), `Variance` and `Tendency` from the heart-rate histogram |
| Target | `FetalState`, one of Normal, Suspect, Pathologic. These are the original NSP codes 1, 2 and 3 |
| Class counts | Normal 1,647, Suspect 292, Pathologic 175 |

No missing values remain. I split the rows 80/20 with stratification (`random_state=42`), which gives 1,691 training rows and 423 test rows. The test rows are `test_data.csv`.

Logistic regression, k-nearest neighbours and the SVM are pipelines with `StandardScaler`, so the scaling is saved inside the model file. The tree and Gaussian naive Bayes take the raw measurements. Precision, recall and F1 are macro averages, because Suspect and Pathologic should count as much as Normal. AUC is the one-vs-rest macro score from the predicted probabilities, and MCC is the multi-class coefficient. About 78% of the tracings are Normal, so accuracy on its own would flatter a model that never raises an alarm.

## c. GitHub repository

https://github.com/VamsiKrishna-2210/Fetal-State-Classifier-ML-Assignment-2

This repository holds the training script, the Streamlit app, the requirements file, this README, the test split, and the five saved models.

- `app.py` scores an uploaded test CSV from the saved models
- `train_models.py` fits the five models, writes `test_data.csv`, and prints the metric table
- `common.py` is the feature list, the model settings, and the metric code shared by both scripts
- `requirements.txt`
- `data/ctg.csv`, the cleaned dataset
- `test_data.csv`, the 423 held-out rows
- `model/`, the five `.joblib` files
- `results/metrics.csv`, the table printed by training
- `.streamlit/config.toml`, the app colours

To run it from this folder: `pip install -r requirements.txt`, then `python train_models.py`, then `streamlit run app.py`.

## d. Models used

Results on the 423 test rows. Fitting all five models takes a couple of seconds.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8842 | 0.9749 | 0.7605 | 0.8776 | 0.8067 | 0.7372 |
| Decision Tree | 0.9031 | 0.9632 | 0.8080 | 0.9169 | 0.8534 | 0.7720 |
| KNN | 0.9243 | 0.9717 | 0.8990 | 0.8323 | 0.8618 | 0.7861 |
| Naive Bayes | 0.8038 | 0.8981 | 0.6606 | 0.7543 | 0.6858 | 0.5845 |
| Support Vector Machine | 0.8960 | 0.9828 | 0.8287 | 0.9054 | 0.8565 | 0.7600 |

Settings: logistic regression with `max_iter=2000` and `class_weight=balanced`. Decision tree with `max_depth=6`, `min_samples_leaf=8` and `class_weight=balanced`. KNN with `k=11` after scaling. Gaussian naive Bayes at its defaults. SVM with an RBF kernel, `C=1` and `class_weight=balanced`.

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Accuracy 0.8842 and AUC 0.9749. It finds 32 of the 35 Pathologic tracings, and the other 3 are labelled Suspect rather than Normal. Suspect precision is 0.59: 30 of the 330 Normal tracings are labelled Suspect, and 6 Normal tracings are labelled Pathologic. The linear boundary separates the extremes and still mixes the middle class into both sides. |
| Decision Tree | Accuracy 0.9031 and the best recall of the five, 0.9169. It finds 34 of 35 Pathologic tracings. The depth cap keeps the leaves coarse, so AUC is 0.9632, behind logistic regression and the SVM. 24 Normal tracings are called Suspect and 9 are called Pathologic, which is why precision (0.8080) stays under KNN. |
| KNN | Best accuracy (0.9243), precision (0.8990), F1 (0.8618) and MCC (0.7861). On this split every Pathologic call is correct: all 28 predicted Pathologic rows really are Pathologic. It still misses 7 Pathologic tracings (2 called Normal, 5 called Suspect) and calls 16 Suspect tracings Normal, so recall (0.8323) is the low one among the stronger models. After scaling, the vote is 11 neighbours, and the saved file keeps the training rows. |
| Naive Bayes | Lowest on every metric: accuracy 0.8038, F1 0.6858, MCC 0.5845. Width, min, max, mode, mean, median and variance all describe one histogram, while naive Bayes treats those columns as independent. 46 Normal tracings are called Suspect and 14 are called Pathologic. Only 21 of 35 Pathologic tracings are recovered; 12 of the rest are called Suspect. |
| Support Vector Machine | Best AUC, 0.9828. It finds 33 of 35 Pathologic tracings and labels only 1 Normal tracing as Pathologic. Accuracy 0.8960, F1 0.8565 and MCC 0.7600 sit just behind KNN. The mix-up that remains is Suspect: 34 Normal tracings are pulled in, and 5 Suspect tracings are called Normal. The RBF kernel follows that overlap more closely than the linear model, which is what the AUC is showing. |
| Overall, winner for this dataset | KNN. It leads on accuracy, precision, F1 and MCC. The SVM is the one I would keep beside it, because it misses 2 Pathologic tracings where KNN misses 7, and it has the best AUC. Naive Bayes is a poor match for this set of repeated histogram measurements. |

## e. Streamlit app

The page is headed Fetal state classifier. The model dropdown sits under the title. Down the left, the five models are stacked in F1 order, and clicking a card opens that model. The open model is the dark card. A line separates the cards from that model's confusion matrix and classification report. Each card shows accuracy, AUC, precision, recall, F1 and MCC, and the best figure in a metric is marked.

The test CSV, the switch for the included `test_data.csv`, and the column list are at the bottom of the page.

On the right is that model only: a confusion matrix as a count grid, then the classification report as three blocks (Normal, Suspect, Pathologic) with precision, recall and F1, then the tracings it misread. Predictions download from that side. The page loads the files in `model/` and only scores them.

The CSV needs the 21 numeric features and a `FetalState` column with the values Normal, Suspect or Pathologic. Extra columns are ignored, and the column order does not matter. Anything else shows an error. When the file has a single class, AUC is shown as n/a.
