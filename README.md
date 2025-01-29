## SVM Classifier for Cats vs. Dogs Image Classification
### 1 -Project Overview :
This project aims to classify images of cats and dogs using a Support Vector Machine (SVM) model. The data for this project is sourced from the Kaggle competition "Dogs vs. Cats," where images are labeled either as a "cat" or a "dog." The task involves downloading, preprocessing the images, extracting features, training the model, and evaluating its performance through various metrics.

### 2- Libraries and Tools:
Several key libraries were utilized throughout the project, each serving specific roles in handling data, images, and modeling:

- **NumPy:** Utilized for numerical operations, especially handling arrays that store image data and labels.
- **NumPy:**  Used to organize image paths into dataframes and manage the dataset for easier manipulation.
- **OpenCV(cv2):** This library is crucial for image processing tasks such as loading, resizing, and converting images from one format to another (RGB conversion).
- **Matplotlib:** Employed for visualizing both the dataset (sample images) and performance metrics (ROC curves, confusion matrix).
**scikit-learn: Provides tools for:**
- **Data splitting:** Separating the dataset into training and testing sets.
- **Feature scaling:** Ensuring all image features are scaled consistently.
- **Principal Component Analysis (PCA):** Reducing the dimensionality of image data for efficiency.
- **SVM:** The primary classification algorithm used in this project.
- **Model evaluation:** Includes accuracy, classification report, confusion matrix, and ROC-AUC metrics.

### 3-Dataset:
The dataset used in this project is sourced from Kaggle's "Dogs vs. Cats" competition. It contains thousands of images labeled either as cats or dogs. The dataset is divided into training and testing subsets, and each image is associated with a filename that identifies whether it is a cat or dog.

After downloading the dataset, the images are stored in a structured format, with separate directories for training and testing images.


### 4-Image Preprocessing
Since the images in the dataset come in varying sizes and formats, preprocessing is essential to ensure they are ready for model training. The following preprocessing steps were applied:

- **Resizing:** All images were resized to a uniform size of 64x64 pixels. This not only reduces computational complexity but also standardizes the input to the model.
- **RGB Conversion:** Each image was converted to an RGB format for consistent handling of color information.
- **Flattening:** After resizing, each image (originally a 3D matrix) was flattened into a 1D vector. This transforms the image into a format suitable for use in machine learning algorithms like SVM.

### 5-Feature Extraction
The flattened image vectors contain thousands of features, making it difficult for traditional machine learning models to handle. To tackle this, Principal Component Analysis (PCA) was used to reduce the dimensionality of the data. PCA is a widely used technique for dimensionality reduction, which extracts the most important features from the data while minimizing the loss of information.

In this project, PCA reduced the feature space to 100 components, effectively balancing computational efficiency and the retention of significant information.


### 6- Model: Support Vector Machine (SVM)
The model used for classification is an SVM with a Radial Basis Function (RBF) kernel. SVM is well-suited for binary classification tasks, making it ideal for distinguishing between cats and dogs in images.

- **RBF Kernel:** The RBF kernel allows the SVM to model non-linear relationships between the input features and output labels. This is particularly useful in image classification, where pixel values exhibit complex patterns that may not be linearly separable.
- **Hyperparameters:** The model's C parameter, which controls the trade-off between maximizing the margin and minimizing classification errors, and the gamma parameter, which defines the influence of a single training example, were optimized for better performance.

The SVM model was integrated into a pipeline along with feature scaling and PCA to ensure a seamless workflow from preprocessing to model training.


### 7-Model Training
The dataset was split into training and testing sets, with 80% of the data used for training and 20% reserved for testing. A pipeline was constructed that included:

StandardScaler: This normalized the image features, ensuring that all pixel values were on the same scale, which is important for SVM.
- **PCA:** Reduced the dimensionality of the input data to 100 components.
- **SVM:** The final classifier in the pipeline, which learned to distinguish between cats and dogs based on the reduced feature set.
The SVM model was trained using the training data, and then the test data was used to evaluate its performance.

### 8-Evaluation Metrics
8.1-Accuracy and Classification Report The accuracy of the model was calculated to assess its overall performance in classifying the images correctly. In addition, a classification report was generated, which provided precision, recall, F1-score, and support for each class (cat and dog).

- **Precision:** Measures the accuracy of the model's positive predictions (e.g., the proportion of predicted dogs that are actually dogs).
- **Recall(Sensitivity):** Indicates how well the model captures all positive instances (e.g., the proportion of actual dogs that were correctly identified).
- **F1-Score:** The harmonic mean of precision and recall, which provides a balanced evaluation of the model’s performanc.

8.2-Receiver Operating Characteristic (ROC) and AUC A ROC curve was plotted to visualize the trade-off between the true positive rate (recall) and the false positive rate for different thresholds of the classifier. The Area Under the Curve (AUC) score was calculated to summarize the classifier's performance. A high AUC score (closer to 1) indicates strong classification performance.

- **True Positive Rate (TPR):** Measures how effectively the model identifies true positives (e.g., correctly identified dogs).
- **False Positive Rate (FPR):** Measures the proportion of actual negatives (cats) that are incorrectly classified as positives (dogs).

8.3 -Confusion Matrix A confusion matrix was generated to give a detailed breakdown of the model's predictions. The matrix includes:

- **True Positives (TP):** Correctly identified dogs.
- **True Negatives (TN):** Correctly identified cats.
- **False Positives (FP):** Cats incorrectly classified as dogs.
- **False Negatives (FN):** Dogs incorrectly classified as cats.
This matrix helps identify specific areas where the model may struggle.


### 9-Visualizations
Two key visualizations were created to help understand the model’s performance:

- **Sample Images:** A grid of cat and dog images was displayed to provide a visual understanding of the input data.
- **ROC Curve:** The ROC curve provided insights into the model's performance across different thresholds, and the AUC score summarized its effectiveness.



#### Conclusion
The SVM classifier with an RBF kernel, coupled with PCA for dimensionality reduction, performed well on the task of classifying images of cats and dogs. The model achieved high accuracy, with balanced precision and recall for both classes. Additionally, the ROC-AUC score and confusion matrix provided a comprehensive view of the model's strengths and weaknesses.

Further improvements could be made by:

- **Hyperparameter Tuning:** Experimenting with different C and gamma values for the SVM.
- **Data Augmentation:** Increasing the dataset’s diversity by applying transformations to the training images, such as rotations, flips, or zooms.
- **Transfer Learning:** Using pre-trained Convolutional Neural Networks (CNNs) could improve accuracy by leveraging models that have already been trained on similar image data.
