# Internship-Project
Internship project focused on building a prediction model to detect anomalies in angular data. The work involves the use of clustering algorithms and neural networks.

# Table of Contents

## Part I : List of All Models Reviewed
1. Introduction
2. Clustering Algorithm
3. Neural Networks
4. Autoencoder Neural Networks

## Part II : Results + Analysis
1. Results
2. Manual Labeling Interfaces
3. Analysis of the Results

## Introduction (named 0.Tools)

The objective of this project is to develop a method (or several methods) for detecting anomalies in time-series data streams.

Let us first take a look at the data. Each file is a time series stored in `.parquet` format, containing 76 columns and potentially up to 9 hours of continuous data. For this project, we focus exclusively on the `alpha_angle` and `beta_angle` columns. The sensor records 25 angle measurements per second, which corresponds to one measurement every 40 milliseconds (so 90 000 per hour).

Using the notebook **"0.Tools"**, I was able to organize and categorize the filenames into one JSON file, as shown below :

- all_files.json
  - files[0] → Total files (10 430)  
    All files that were detected or processed
  - files[1] → Degree files (4 350)  
    Valid files with alpha and beta angles in degrees
  - files[2] → Radian files (1 937)  
    Valid files with alpha and beta angles in radians

Throughout this project, I mainly use the files listed in the Degree files (files[1]).  
It is important to keep this JSON file regularly updated in order to continue analyzing the most recent sensor data.

Then, in the second part of the notebook, you can find the steps and explanations on how to save and load a model/scaler using joblib.

## Part I : List of All Models Reviewed

### 1. Clustering Algorithm (named 1.Clustering)

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm particularly well-suited for discovering clusters of arbitrary shape and for identifying outliers as noise.
Unlike algorithms like K-Means, DBSCAN does not require specifying the number of clusters in advance. It works by grouping together points that are closely packed (points with many nearby neighbors), and marking points that lie alone in low-density regions as outliers.
It is useful in this context for identifying natural groupings in the sensor data without supervision, such as separating normal patterns from abnormal behavior based on density.

#### ➤ On One File : Testing and Parameter Tuning

The **"On one file"** section of the code is designed to help users understand how the DBSCAN algorithm works and how to apply it effectively.

You can select either the index or the name of the file you want to test DBSCAN on, which allows for flexible experimentation with different datasets. However, be cautious about the number of rows extracted :

> ⚠️ If there are more than **100,000 data points**, DBSCAN may fail or become extremely slow due to its high computational cost.

To prevent this, the script uses a downsampling technique to reduce the number of data points, which is highly recommended. Ideally, the dataset should be reduced to around 50,000 to 80,000 points for stable and efficient clustering.

The code includes a function that plots the k-distance graph, a common technique for estimating a good value for the DBSCAN `epsilon` parameter. It also computes the 95th percentile of the distances to the k-th nearest neighbor as a statistical estimate for `epsilon`.

We can observe the following on the three output plots :

- **First plot** : the clusters created by DBSCAN, with each cluster shown in a different color, and the isolated points (noise) displayed in black.
- **Second plot** : the same clusters but without the noise points, offering a clearer view of the grouped data.
- **Third plot** : the centroids of each cluster, computed by averaging the X and Y coordinates (alpha and beta angles) of all the points within each cluster.

These centroids will be our main focus for the next steps.


#### ➤ DBSCAN centroids, Hausdorff distance :

We can now take a look at the first classification method. We will use the dataset labeled.csv, as shown below :

| # | filepath                                              | start_index | end_index | label |
|---|--------------------------------------------------------|-------------|-----------|-------|
| 1 | gs://featurestore-spinewise/083-SPINE-2023-11-23-6-57-3-1700825489-v3.1.1.parquet      | 22464       | 44928     | White |
| 2 | gs://featurestore-spinewise/083-SPINE-2023-11-23-6-57-3-1700825489-v3.1.1.parquet         | 44928       | 67392     | Black |
| 3 | gs://featurestore-spinewise/083-SPINE-2023-11-23-6-57-3-1700825489-v3.1.1.parquet         | 67392       | 89856     | Black |
| 4 | gs://featurestore-spinewise/083-SPINE-2023-11-23-6-57-3-1700825489-v3.1.1.parquet         | 89856       | 112320    | White |
| 5 | gs://featurestore-spinewise/083-SPINE-2023-11-23-6-57-3-1700825489-v3.1.1.parquet         | 112320      | 134784    | Grey |
| 6 | gs://featurestore-spinewise/083-SPINE-2023-11-23-6-57-3-1700825489-v3.1.1.parquet         | 134784      | 157248    | Black |


As shown, each file was split into frames of 22,464 samples (a multiple of 64, close to 22,500), because we aim to classify 15-minute segments.  
Each 15-minute segment was labeled as one of the following :
- **White** (valid files)
- **Grey** (potential anomalies)
- **Black** (anomalous files)
- **Exclude** (no signal or constant signal)

I selected the centroids of the main clusters from Pieter’s *gold_list*. This list was built from 35 files stored in the following folder :  
*gs://featurestore-spinewise/sensor_data_goldlist*  

Each file contains one hour of data (approximately 90 000 points per file) and these files are known to represent normal behavior and have been manually validated as clean, high-quality segments.  
I applied DBSCAN to each of these 35 files and extracted the centroids of the resulting clusters. This process produced a set of 83 centroids, which I refer to as the `gold_centroids`.

Then, for each file in the labeled CSV, I extracted the centroids of the clusters identified by DBSCAN (i.e., the center of each cluster), and computed the Hausdorff distance between the `gold_centroids` and each segment’s set of centroids.

The higher the Hausdorff distance, the more likely the group of centroids is considered anomalous, and thus the corresponding 15-minute sequence is labeled as such.

- I obtained:
  - a vector `X` of shape *(n, 1)* with Hausdorff distances,
  - a vector `Y` of shape *(n, 1)* with the corresponding labels (0 = White, 1 = Black).

Then, I applied **logistic regression** to see whether there's a correlation between the Hausdorff distance and the assigned label.  
The average accuracy across 1000 models was **69%**, indicating **no clear correlation**.

This result led me to consider two possible issues :
1. The centroids from the `gold_centroids` may not be optimal.
2. The input vector `X` may lack additional relevant features beyond the Hausdorff distance.

To improve performance, I replaced the `gold_centroids` with a random selection of centroids from 10 labeled segments in `labeled_s.csv` : 5 Black and 5 White segments.  
The idea is that if a segment has low Hausdorff distances to the White references and high distances to the Black ones, the Random Forest classifier can learn this pattern and infer that the segment is likely normal, and vice versa for anomalous segments.

The resulting feature vector `X` now has shape `(n, 10)`, corresponding to the 10 Hausdorff distances (one for each reference segment).  
With an average accuracy of **91.9%** across 1000 models, this method remains **really promising** but it could still be slightly improved.

#### ➤ Centroids Hausdorff distance + Statistical Parameters :

Starting from the previous idea and the issue related to parameters, I added the following statistical features at the beginning of each `X` vector : **min**, **max**, **median**, **mean**, **standard deviation** for each angle and the **number of centroids**.

I then tested two clustering methods :

- **DBSCAN**, as discussed earlier. After experimenting with various configurations, I found that the best results were obtained when setting `epsilon = 0.5` and `min_samples = 5`.

- **Downsampling** : This method computes the average of all points within a sliding window. Here, I used a window size of 100, which corresponds to an average over 4 seconds for the alpha and beta angles. Over a 15-minute segment, this yields approximately **225 centroids**.

After computing the centroids for each segment, I appended the **10 Hausdorff distances** between these centroids and those of the 10 reference segments selected from the labeled file (as described earlier).  
This brings the total number of features per segment to **21**, hence `X.shape = (n, 21)`.

It is worth noting that applying a **StandardScaler** to `X` improves the classification accuracy by approximately **5%**.  
Additionally, for the classification of **good** vs **anomalous** segments, I used an **SVC model** instead of a **Random Forest classifier**. Although both models yield similar accuracy, **SVC is significantly faster** in terms of computation time.

Here is a brief definition of SVC :  
A supervised machine learning model that aims to separate classes by finding the **optimal hyperplane** that maximizes the margin between points of different classes.

As for the results, the **average accuracy over 500 tests** is **97.3%** when using DBSCAN-based clusters, and **99.0%** when using window-based clusters.

#### ➤ How to update the clustering algorithm ?

Simply run the import cells, then go directly to the **“Centroids Hausdorff dist + parameters”** section. Next, run the cells in **“Data preparation”** followed by those in **“Windows centroids + parameters”**.  
Once the results from the new model are generated, you can save both the updated model and the new scaler by following the instructions in the **“0.Tools”** notebook.  
If you wish to go further, you can also change the reference segments — but be very careful to use the correct segment indices and verify which CSV file they come from.

### 2. Neural Networks (named 2.LSTM classifier)

**LSTM** (Long Short-Term Memory) is a type of recurrent neural network (RNN) particularly well-suited for processing sequential or temporal data, such as signals, sensor readings, or time-stamped events.

Unlike traditional feedforward neural networks, LSTMs are designed to retain memory over long periods of time, allowing them to capture temporal dependencies between elements in a sequence.

In this context, the LSTM model is used to analyze 15-minute segments of sensor data (split into windows of window_size = 1024) and determine whether a segment corresponds to normal (White) or anomalous (Black) behavior.

#### ➤ Load and Prepare Train data :

For this second classification method, we also use the `labeled_s.csv` file, as in the previous approach. At the time of building the model, I had **162 Black files** and **156 White files**. For training, we selected **75 files from each class** to ensure a balanced dataset.

This time, we apply `resample` to reduce each 15-minute segment from **22,464 to 8,192 points**. In short, `resample` adjusts the number of points in a signal by interpolating its values, allowing it to be compressed while preserving its overall shape.

Each resampled segment is then split into **8 windows of 1,024 points**, corresponding to approximately **112 seconds** per window.

As a result, the arrays have the following shapes:
- `X`: `(1200, 1024, 2)`
- `Y`: `(1200, 1)`

I chose this method because it consistently produced the **best results** in my experiments.

#### ➤ Model :

To classify time series segments, I use the following LSTM-based architecture.  
The goal is to train a neural network that takes **2 minutes of sensor data** (i.e., a window of `1024` time steps with 2 features : alpha and beta) and predicts whether this portion comes from a **normal 15-minute segment** (`label = 0`) or an **anomalous one** (`label = 1`).

```python
model = Sequential([
    Input(shape=(window_size, 2)),
    LSTM(128, return_sequences = True, activation = 'tanh'),
    LSTM(64, return_sequences = True, activation = 'tanh'),
    LSTM(4, activation = 'tanh'),
    Dense(1, activation = 'sigmoid')
])
```
#### ➤ Load and Prepare Test data

To build the test set, we use the remaining data that was **not included in the training set**. This ensures that the model is evaluated on **unseen segments**, providing a reliable measure of its generalization performance.

We apply the **exact same preprocessing steps** as for the training set:
- Each 15-minute segment is resampled to **8,192 points**,
- Then split into **8 windows of 1,024 points** (i.e., 112 seconds each).

As a result, we obtain the test set :
- `X_test` has a shape of **(2080, 1024, 2)**, corresponding to 2,080 windows of 2 minutes each.

We can then use the trained model to **predict** on these windows:

```python
prediction = model.predict(X_test)
```

#### ➤ Files results

For each 15-minute segment, the model produces predictions for its 8 sub-windows (each corresponding to 112 seconds of data).  
We compute the average of these 8 predicted values to obtain a single prediction score per segment, this score reflects the model’s overall confidence that the segment is anomalous.

This average is then compared to the segment's true label from `labeled_s.csv` using a fixed decision threshold (typically 0.5):

- If the average prediction is **greater than or equal to 0.5**, the segment is classified as **Black** (anomalous),
- If it is **below 0.5**, the segment is classified as **White** (normal).

The predicted labels are then compared with the ground truth to compute accuracy and analyze the model’s overall performance.

After repeating the training and evaluation process multiple times, the model consistently achieves an accuracy **above 96%**.  
One of the best-performing models was saved, with an accuracy of **98.2%**.

#### ➤ Visualisation

Finally, the visualization section allows us to display the False Positive (FP) and False Negative (FN) segments, along with their actual indices.  
This makes it easier to manually review and correct mislabeled segments in the CSV file if necessary (*we will see how to do this in Part 2*).

#### ➤ How to update the LSTM classifier algorithm ?

To update the **LSTM classifier** algorithm, you need to run almost all the cells. To ensure the model’s accuracy, you can retrain it several times and keep only the best-performing one.  
Even though there is a chart showing accuracy for different threshold values, we always use a fixed threshold of **0.5**. 
To save the new neural network model, use the **“Save model”** cell located right after the last confusion matrix.  

### 3. Autoencoder Neural Networks (named 3.LSTM AE)

An **LSTM Autoencoder** is a neural network that learns to **reconstruct time series data**, making it well-suited for anomaly detection.  
The idea is simple : the model is trained to reproduce normal sequences; when it encounters anomalous data, reconstruction fails and the error increases.

An autoencoder is composed of two parts:

- **Encoder**: Compresses the input sequence into a compact latent representation.
- **Decoder**: Reconstructs the original sequence from this representation.

Unlike basic autoencoders built with dense layers, the **LSTM Autoencoder** is designed for temporal data, such as:
- Sensor streams,
- Time-stamped events,
- Motion sequences.

Thanks to LSTMs, the model can capture time-based dependencies and learn patterns over time.

#### ➤ Load and Prepare Train Data

In this case, the data preparation is slightly different because we want the autoencoder to learn as many **normal patterns** as possible.  
To achieve this, we randomly select **200 files** (excluding scenario files) that contain between 1 and 8 hours of recordings.

#### ➤ Model

For the model, I tested two types of recurrent layers : **LSTM** and **GRU**, while keeping the same overall architecture.  
This allowed me to compare their performance on the same task and evaluate which one handled the reconstruction of normal patterns more effectively.


```python
RNNLayer=tf.keras.layers.LSTM
        
self.encoder = tf.keras.Sequential([
  RNNLayer(128, return_sequences=True, activation='tanh'),
  RNNLayer(64, return_sequences=True, activation='tanh'),
  RNNLayer(32, return_sequences=True, activation='tanh'),
  RNNLayer(16, return_sequences=False, activation='tanh')
])
        
self.decoder = tf.keras.Sequential([
  RepeatVector(input_shape[0]),
  RNNLayer(32, return_sequences=True, activation='tanh'),
  RNNLayer(64, return_sequences=True, activation='tanh'),
  RNNLayer(128, return_sequences=True, activation='tanh'),
  TimeDistributed(Dense(input_shape[1]))
])
```

#### ➤ Test data

Now that both **GRU** and **LSTM** autoencoder models are trained, we can use them to process the 15-minute segments from `labeled_s.csv`.  
Each segment is reconstructed by the model, and we compute the **reconstruction loss**.

For every segment, we extract the following statistics from the reconstruction errors of the alpha and beta angles :
- **min**, **max**, **mean**, **median**, and **standard deviation**.

This gives us `X_train` with shape `(n, 10)` and `Y_train` with shape `(n, 1)`,  where each label is either **0** (normal) or **1** (anomalous).

We then use a **Random Forest Classifier** to check whether there's a correlation between reconstruction error patterns and segment labels.

As a first step, we evaluate the performance of the GRU-based model.  
As in the previous method, we also visualize **False Positives** and **False Negatives**, along with their indices in the CSV file, to manually review or correct them if needed.

#### ➤ Test Best Model Random Forest

To determine which model performs best, we compared the two by running **1,000 tests** on random data.  
The results show that the **LSTM model slightly outperforms GRU**, with:
- **LSTM** : average accuracy of **0.990**
- **GRU** : average accuracy of **0.986**

Therefore, the LSTM autoencoder appears to be slightly more efficient for this task.

#### ➤ How to update the LSTM ae algorithm ?

To update the **LSTM AE** algorithm, you don’t need to retrain the autoencoder to reconstruct the segments.  
You can directly run the cells in **“Test Data”** and check the Random Forest results.  
You may also use the **“Test best model random forest”** section in case the current LSTM autoencoder model is no longer the one achieving the highest accuracy.  
After that, you can save the new Random Forest model by following the instructions in the **“0.Tools”** notebook.

## Part II: Analysis

### 1. Results

| Methods                                                                                         | Accuracy |
|--------------------------------------------------------------------------------------------------|----------|
| *1. Clustering*                                                                                |          |
| Hausdorff distance DBSCAN centroids to DBSCAN gold list centroids                               | 69.0%      |
| Hausdorff distance DBSCAN centroids to 10 DBSCAN centroids group                                | 91.90%   |
| Hausdorff distance DBSCAN centroids to 10 DBSCAN centroids group + statistical parameters        | 97.30%   |
| **Hausdorff distance windows centroids to 10 windows centroids group + statistical parameters** | **99.0%**  |
| *2. LSTM classifier*                                                                           |          |
| **LSTM classifier**                                                                                  | **98.20%**   |
| *3. LSTM AE*                                                                            |          |
| GRU reconstruction loss                                                              | 98.60%      |
| **LSTM reconstruction loss**                                                          | **99.0%** |

We will use the **highlighted models** in the final steps, as they achieved the best accuracy scores.


### 2. Manual Labeling Interfaces (named 4.Analyze)

The purpose of the **Analyze** module is to **manually classify 15-minute segments** by assigning them a label.

#### ➤ File Classification Interface

We start by loading the list of usable files from `all_file.json`, then display all folders containing at least one valid file, allowing the user to select a specific folder. The files to be classified are taken exclusively from this chosen folder.

Two interfaces are available to **label new segments** and save the results in `segments_label.csv`: the first processes files directly from folders, while the second uses the outputs from previous predictions.

You can assign the following labels :
- `white` → normal segment
- `grey` → potentially anomalous
- `black` → clearly anomalous
- `exclude` → no signal or constant signal

Additional options are also available:
- `skip` → skip the segment without saving anything to the CSV
- `back and delete` → remove the last saved line from the CSV and return to the previous segment (in case of a labeling mistake)

For each 15-minute segment, the interface displays:
- The **original signal**,
- Its **reconstructed version** (via a pre-trained LSTM Autoencoder),
- The **reconstruction loss**,
- And **predictions from previously trained models** (e.g., LSTM classifier and Autoencoder).

This allows for quick comparison and helps guide manual labeling.

#### ➤ Classification verification interface

The second interface is used to **review and correct the labels** of previously processed segments.

At the beginning of the script, you can select a **starting index** if you already know which segment you want to review or modify.

A **dropdown menu** allows you to filter the displayed segments by label, with the following options :
- `All`
- `black`
- `grey`
- `white`
- `exclude`

If a label was assigned incorrectly, you can easily correct it using the following commands :
- `set black`
- `set grey`
- `set white`
- `set exclude`

#### ➤ Prediction on all the no-scenario files

**Reminder :** The database contains **4,300 files** with alpha and beta angles in degrees, among which **890** are labeled as *no-scenario* files. 
These 890 files were divided into 15-minute segments, resulting in a total of **9,087 segments**.

For each segment, the prediction is represented by a **three-letter code**:

- The **first letter** comes from the **clustering algorithm** :  
  `"B"` if the segment is classified as abnormal (*black*),  
  `"W"` if it is considered normal (*white*).

- The **second letter** is the prediction from the **LSTM classifier**.

- The **third letter** is the prediction from the **LSTM autoencoder**.

Segment Distribution by Prediction : 

| Prediction | Count | Percentage |
|------------|-------|------------|
| BBB        | 198   | 2 %         |
| BBW        | 2570  | ~28 %       |
| BWB        | 45    | 0.5 %       |
| BWW        | 640   | 7 %         |
| WBB        | 18    | 0.1 %       |
| WBW        | 2261  | 25 %        |
| WWB        | 9     | 0.05 %      |
| WWW        | 3346  | ~37 %       |

#### ➤ Visualizing the predictions from each algorithm

This app is used to visualize the predictions of the three classification algorithms. It enables manual inspection by allowing users to :

- Visually compare alpha and beta signals with predicted classifications.
- Apply filters to focus on specific threshold patterns.
- Identify discrepancies between models, such as cases where the clustering algorithm detects a specific type of anomaly while the LSTM classifier identifies a different one.

#### ➤ Construction of the ensemble prediction

In this section, we use a Random Forest model that takes as input (`X`) the three predictions from the different algorithms for a given segment. It is trained on 1,000 labeled segments from `labeled_s2.csv` to build a prediction model. The output is a global prediction for each segment, represented as a number between 0 and 1, where 0 indicates a normal segment and 1 indicates an abnormal segment. The final prediction for each segment is then saved in the `predictions.csv` file.

#### ➤ Visualisation of the ensemble prediction

Finally, the last interface displays, for each segment, the final prediction probability along with the probabilities from the three previous algorithms.


### 3. Analysis of the results

After classifying each segment into a distinct category, we can interpret the labels as follows :

- **BBB** : Segments with **both abnormal shape and abnormal values**. These are the most clearly anomalous cases, flagged by all three models.

- **BBW** : Segments with a **normal shape but abnormal values**. This often corresponds to a sudden shift or gap in the signal, while the overall structure remains consistent.

- **BWB** : Segments containing **wrapped angle discontinuities** (explained below), combined with an **anomalous shape**. These segments are flagged both for **sudden angle jumps** and **irregular temporal structure**.

- **BWW** : Segments with a **normal shape but abnormal values**, often due to small gaps or amplitude shifts. Despite the values being off, the signal remains structurally consistent.

- **WBB** : Segments that appear **mostly normal**, but include **one or more small angular discontinuities**, caused by **angle wrapping**. These are usually **subtle anomalies**, borderline between normal and abnormal.

- **WBW** : Segments that are **neither clearly normal nor clearly abnormal**, similar to those found in *"grey"* files. These may also include cases where the subject is sitting still, producing ambiguous or flat signal patterns.

- **WWB** : Segments that are **mostly normal but include a few clipped values**, not enough to strongly influence the shape or values globally.

- **WWW** : Segments with **both normal shape and normal values**, considered as fully normal by all models.

#### What are wrapped angle discontinuities ?

Wrapped angle discontinuities occur when an angular measurement **drops below –90°** and is automatically **re-expressed as +270°**, due to how angular data is encoded. This behavior is typical in systems where angles are confined within a fixed range, such as –90° to +270°. Instead of continuing smoothly, the signal suddenly "jumps" from a low negative value to a high positive one.


#### ➤ Interpretation of Model Behavior Based on the Three-Letter Code

- **Clustering**  
  This model is mainly influenced by the **raw values** of the segment.  
  If neither alpha nor beta exceed typical thresholds, the clustering model tends to classify the segment as normal (`W`).  
  Conversely, large deviations or **clipped values** often result in a `B`.

- **LSTM Classifier**  
  This model is more sensitive to the **shape of the signal**, regardless of the absolute values.  
  It tends to flag segments with **irregular patterns, or sudden structural changes**, even when the values remain within a normal range.

- **LSTM Autoencoder**  
  This model is based on the **statistical characteristics of the reconstruction loss**.  
  It is particularly effective at detecting **clipped signals**, which produce a **higher reconstruction error** compared to typical segments.
