# %% [markdown]
# # **Lab Assignment 2**

# %% [markdown]
# ## 1. Import required libraries and load the load_Wine Dataset from scikit-learn [0 Marks]

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from sklearn.datasets import load_wine

wine = load_wine()
x = wine.data       # feature variables
y = wine.target     # target variable
df = pd.DataFrame(x, columns=wine.feature_names)
df['Class'] = y
print("Shape of the data frame:", df.shape)
df.head()

# %% [markdown]
# ## 2. Exploratory Data Analysis [2 Marks]
# 
# Perform the following analyses.
# 
# (a) Check
# - Missing values
# - Data types
# 
# (b) Visualize Class distribution
# 
# (c) Draw Correlation Heatmap
# 
# (d) Plot Histograms
# 
# (e) Generate Pair Plot
# 
# (f) Write your observations for each visualization.

# %%
print('COUNT OF MISSING VALUES')
df.isnull().sum()

# %%
print('DATA TYPE OF EACH COLUMN')
print(df.dtypes)

# %%
print('CLASS DISTRIBUTION')
class_labels = df['Class'].value_counts()
print(class_labels)

class_labels.plot(kind='bar', color=['purple', 'orange', 'green'])
plt.title('Class Distribution', fontsize=20)
plt.xlabel('Wine Class')
plt.ylabel('Count')
plt.savefig('plots/class_distribution.png')

# %%
# Correlation Heatmap
plt.figure(figsize=(15, 10))
plt.title("Correlation Heatmap", fontsize=20)
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='Spectral')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png')

# %%
# Histograms
df.hist(figsize=(15, 12), bins=20, edgecolor='Black')
plt.suptitle('Distribution of All Features', fontsize=20)
plt.tight_layout()
plt.savefig('plots/histogram_all_features.png')

# %%
# Pair plot of selected features
selected_features = ['alcohol', 'flavanoids', 'color_intensity', 'proline']
sns.pairplot(df, vars=selected_features, hue='Class')
plt.savefig('plots/pair_plot_selected_features.png')

# %% [markdown]
# ***EDA Observations***
# 
# | **Analysis** | **Observation** |
# |---|---|
# | Missing Values | No missing values are present in the dataset. |
# | Data Types | All features are numerical; target is an integer representing 3 classes (0 to 2). |
# | Class Distribution | Class 1 has the most samples (71), followed by Class 0 (59) and Class 2 (48). |
# | Correlation | Highest +ve: flavanoids and total_phenols (0.86), Highest -ve: flavanoids and Class (-0.85). |
# | Histograms | Features have different scales and varying distributions. Many distributions are skewed. |
# | Pair Plot | Several feature combinations involving flavanoids show less overlap between the classes. |

# %% [markdown]
# ## 3. Perform Feature Scaling and One-Hot Encoding [1 Marks]
# 
# - Use StandardScaler
# 
# - Compare Original features with Scaled features
# 
# - Display Mean , Standard deviation
# 
# - Convert the target labels into one-hot vectors.
# 
# - Display the encoded labels.

# %%
# Feature scaling using Standard Scaler
from sklearn.preprocessing import StandardScaler

x = df.drop('Class', axis=1)    # drop the Class column
y = df['Class']                 # store the target Class

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)  # returns numpy array

# convert back into a data frame
x_scaled = pd.DataFrame(x_scaled, columns=x.columns)    
x_scaled.head()

# %%
# Original vs scaled features (Box Plot)
fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# original features
sns.boxplot(data=x, ax=axes[0])
axes[0].set_title('Original features', fontsize=20)
axes[0].tick_params(axis='x', rotation=30)

# scaled features
sns.boxplot(data=x_scaled, ax=axes[1])
axes[1].set_title('Scaled features', fontsize=20)
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('plots/original_vs_scaled.png')

# %%
# Mean & Standard Deviation
stats = pd.DataFrame({
    'Original mean': x.mean(),
    'Original std. deviation': x.std(),
    'Scaled mean': x_scaled.mean(),
    'Scaled std. deviation': x_scaled.std()
})
display(stats)

# %%
# One-hot encoding of Target Class
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)                # don't return sparse matrix
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))  # reshape into single-column matrix

# convert back into a data frame
y_encoded = pd.DataFrame(y_encoded, columns=encoder.get_feature_names_out(['Class']))

# display the encoded labels
display(y_encoded)

# %% [markdown]
# ## 5. Split the data into training, validation, and test sets (70%-15%-15%). [Marks 1]

# %%
from sklearn.model_selection import train_test_split

# Train : Val : Test = 70 : 15 : 15
x_train, x_temp, y_train, y_temp = train_test_split(x_scaled, y_encoded, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

print(f'Shape of Training data: X = {x_train.shape} Y = {y_train.shape}')
print(f'Shape of Validation data: X = {x_val.shape} Y = {y_val.shape}')
print(f'Shape of Testing data: X = {x_test.shape} Y = {y_test.shape}')

# %% [markdown]
# ## 6. Implement Elastic-Net Softmax Regression from Scratch [6 Marks]
# 
# In this step, you will implement an **Elastic-Net Regularized Softmax Regression** model from scratch using **Gradient Descent**. **Do not use any machine learning library** (e.g., Scikit-learn, TensorFlow, or PyTorch) for model training.
# 
# Your implementation must include the following components.
# 
# ---
# 
# ### (a) Softmax Function
# 
# Implement the **Softmax activation function** to convert the raw logits into class probabilities.
# 
# The Softmax function should satisfy the following properties:
# 
# - The probability of each class lies between **0 and 1**.
# - The sum of probabilities for each sample should be **equal to 1**.
# 
# ---
# 
# ### (b) Cross-Entropy Loss
# 
# Implement the **Multiclass Cross-Entropy Loss** to measure the difference between the predicted probabilities and the true class labels.
# 
# Your implementation should compute the average loss over all training samples.
# 
# ---
# 
# ### (c) L1 Regularization (Lasso)
# 
# Extend the loss function by adding an **L1 Regularization** term.
# 
# The L1 penalty is defined as
# 
# $$
# L_{L1}=\lambda_1\sum_{i,j}|W_{ij}|
# $$
# 
# where
# 
# - $W$ is the weight matrix.
# - $\lambda_1$ is the L1 regularization parameter.
# 
# ---
# 
# ### (d) L2 Regularization (Ridge)
# 
# Further extend the loss function by adding an **L2 Regularization** term.
# 
# The L2 penalty is defined as
# 
# $$
# L_{L2}=\frac{\lambda_2}{2}\sum_{i,j}W_{ij}^{2}
# $$
# 
# where
# 
# - $W$ is the weight matrix.
# - $\lambda_2$ is the L2 regularization parameter.
# 
# ---
# 
# ### (e) Gradient of L1 Regularization
# 
# Implement the gradient of the L1 regularization term.
# 
# The gradient is given by
# 
# $$
# \frac{\partial |W|}{\partial W}=\operatorname{sign}(W)
# $$
# 
# ---
# 
# ### (f) Gradient of L2 Regularization
# 
# Implement the gradient of the L2 regularization term.
# 
# The gradient is given by
# 
# $$
# \frac{\partial}{\partial W}\left(\frac{\lambda_2}{2}\|W\|_2^2\right)=\lambda_2W
# $$
# 
# ---
# 
# ### (g) Gradient Descent Optimization
# 
# Implement the complete **Gradient Descent** algorithm for training the model.
# 
# During each training epoch, your implementation must perform the following steps:
# 
# 1. Compute the logits.
# 2. Apply the Softmax function.
# 3. Compute the Cross-Entropy Loss.
# 4. Add the L1 and L2 regularization terms.
# 5. Compute the gradients of the weights and bias.
# 6. Update the model parameters using Gradient Descent.
# 7. Store the **Training Loss**.
# 8. Compute and store the **Validation Loss**.
# 
# ---
# 
# ### Expected Outputs
# 
# After completing this step, your implementation should:
# 
# - Successfully train an Elastic-Net Softmax Regression model.
# - Store the **training loss** after every epoch.
# - Store the **validation loss** after every epoch.
# - Learn the optimal weight matrix and bias vector.
# 
# > **Note**
# >
# > - You must implement the complete algorithm **from scratch**.
# > - Do **not** use any built-in machine learning library for model training.
# > - Only **NumPy** may be used for numerical computations.

# %%
class ElasticNetSoftmaxRegression:
    def __init__(self, learning_rate, n_epochs, l1, l2):
        """
        Hyperparameters
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs    # no. of iterations
        self.lambda1 = l1           # L1 loss
        self.lambda2 = l2           # L2 loss
    
    def softmax(self, z):
        """
        Softmax function
        """
        z = z - np.max(z, axis=1, keepdims=True)        # prevent overflow   
        exp_z = np.exp(z)                               # exponentiation
        exp_sum = np.sum(exp_z, axis=1, keepdims=True)  # sum of exponentiation
        return exp_z / exp_sum
    
    def compute_loss(self, y_true, y_pred):
        """
        Loss Function = mean Cross-Entropy loss + lambda1 * L1 loss + lambda2 / 2 * L2 loss
        """
        epsilon = 1e-15
        sample_errors = -np.sum(
            y_true * np.log(y_pred + epsilon), axis = 1)    # add epsilon to prevent log(0) computation
        mean_ce_error = np.mean(sample_errors)              
        
        l1_loss = self.lambda1 * np.sum(np.abs(self.w))     # sum of absolute weights
        l2_loss = self.lambda2 / 2 * np.sum(self.w ** 2)    # sum of squared weights
        
        total_loss = mean_ce_error + l1_loss + l2_loss
        return total_loss
    
    def fit(self, x_train, y_train, x_val, y_val):
        """
        Train the model
        """
        n_samples, n_features = x_train.shape
        n_classes = y_train.shape[1]
        
        # initialize the model weights & bias
        self.w = np.zeros((n_features, n_classes))
        self.b = np.zeros((1, n_classes))
        
        # loss storage
        self.train_loss = []
        self.val_loss = []
        
        # Gradient descent optimization
        for epoch in range(self.n_epochs):
            # forward pass
            logits = x_train @ self.w + self.b              # compute the logits 
            y_pred = self.softmax(logits)                   # apply softmax 
            
            # compute the loss function
            train_loss = self.compute_loss(y_train, y_pred) 
            
            # compute gradients
            dz = (y_pred - y_train) / n_samples             
            dw = x_train.T @ dz                             # weight gradient of cross entropy loss
            dw += self.lambda1 * np.sign(self.w)            # add gradient of L1
            dw += self.lambda2 * self.w                     # add gradient of L2
            db = np.sum(dz, axis=0, keepdims=True)          # bias gradient
            
            # update model parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # store training loss
            self.train_loss.append(train_loss)
            
            # store validation loss
            val_pred = self.softmax(x_val @ self.w + self.b)
            self.val_loss.append(self.compute_loss(y_val, val_pred))

    def predict_probabilities(self, x):
        """
        Compute the probability vectors
        """
        logits = x @ self.w + self.b
        return self.softmax(logits)
    
    def predict_class(self, x):
        """
        Return the predicted class labels
        """
        return np.argmax(self.predict_probabilities(x), axis=1)
    

# %% [markdown]
# ## 7. Train the Model [1 Mark]
# 
# Train the **Elastic-Net Softmax Regression** model using the **Gradient Descent** algorithm.
# 
# ---
# 
# ## Training Configuration
# 
# Train the model for **1000 epochs**.
# 
# Select appropriate values for the following hyperparameters:
# 
# - **Learning Rate ($\alpha$)**
# - **L1 Regularization Parameter ($\lambda_1$)**
# - **L2 Regularization Parameter ($\lambda_2$)**
# 
# You may experiment with different values of these hyperparameters to achieve better model performance.

# %%
# convert data frames into numpy arrays
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_val = x_val.to_numpy()
y_val = y_val.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

# train the model
ens_regressor = ElasticNetSoftmaxRegression(learning_rate=0.01, n_epochs=1000, l1=0.01, l2=0.01)
ens_regressor.fit(x_train, y_train, x_val, y_val)

# %% [markdown]
# ## 8. Plot Learning Curve [1 Mark]
# 
# - Plot Training Loss and Validation Loss
# 
# - Discuss Underfitting and Overfitting

# %%
# Learning curve
plt.figure(figsize=(10, 5))
sns.lineplot(ens_regressor.train_loss, label='Training loss', linestyle='-', color='green')
sns.lineplot(ens_regressor.val_loss, label='Validation loss', linestyle='--', color='red')

plt.title('Training vs Validation loss')
plt.xlabel('Epoch (Iterations)')
plt.ylabel('Total Loss (Cross Entropy + L1 + L2)')
plt.grid(True)
plt.savefig('plots/training_vs_validation.png')

# %% [markdown]
# **Remark on Overfitting and Underfitting**
# - Both training and validation losses decrease steadily, indicating ***no significant underfitting***.
# - The losses remain close, and validation loss continues to decrease, indicating ***no significant overfitting***.

# %% [markdown]
# ## 9. Predict on test set [1 Mark]
#  - display actual and predicted class

# %%
# Predict classes for test data
y_pred = ens_regressor.predict_class(x_test)
y_true = np.argmax(y_test, axis=1)

# Display results
results = pd.DataFrame({
    'Actual class': y_true,
    'Predicted class': y_pred
})
display(results)

# %% [markdown]
# ## 10. Evaluate the Model [1 Mark]
# 
# - compute Accuracy, precision, recall, f1_score

# %%
# Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_true, y_pred):
    """
    Calculate the scores to evaluate model predictions
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return accuracy, precision, recall, f1


# %%
accuracy, precision, recall, f1 = evaluate_model(y_true, y_pred)

metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score': [accuracy, precision, recall, f1]
})
display(metrics)

# %% [markdown]
# ## 11. Plot the Confusion matrix, Classification report and interpret the results. [1 Marks]

# %%
# Confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, cmap='PuRd')

plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png')

# %%
# Classification report
from sklearn.metrics import classification_report

cr = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=['Class 0', 'Class 1', 'Class 2'])
print(cr)

# %% [markdown]
# **Interpretation of the Results**
# - All the test samples were correctly classified, with ***no misclassifications*** between the three classes.
# - Precision, recall, and F1-score are ***1.0000*** for all three classes, indicating perfect classification.
# 
# Overall, the `from-scratch implementation of Elastic-Net Softmax Regression` model performs very well on this test set, with no classification errors.

# %% [markdown]
# ## 12. Perform the Sklearn Implementation and Compare the both model. [1 Marks]

# %%
from sklearn.linear_model import LogisticRegression

sk_ens_regressor = LogisticRegression(
    # penalty='elasticnet',   # support both L1 and L1 penalties (DEPRECATED)
    solver='saga',          # only 'saga' solver supports 'elasticnet'
    C=500,                  # C = 1 / lambda
    l1_ratio=0.5,           # ratio of lambda 1 in lambda
    max_iter=1000,
    random_state=42
)
sk_ens_regressor.fit(x_train, np.argmax(y_train, axis=1))   # train
sk_y_pred = sk_ens_regressor.predict(x_test)                # predict

# %%
# compare prediction results
result_comparison = pd.DataFrame({
    'Actual values': y_true,
    'Predicted values (From-scratch)': y_pred,
    'Predicted values (Sklearn)': sk_y_pred
})
display(result_comparison)

# %%
# compare performance metrics
sk_accuracy, sk_precision, sk_recall, sk_f1 = evaluate_model(y_true, sk_y_pred)

metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score (From-scratch)': [accuracy, precision, recall, f1],
    'Score (Sklearn)': [sk_accuracy, sk_precision, sk_recall, sk_f1]
})
display(metrics)

# %% [markdown]
# ***Interpretation of the Comparison***
# 
# Both the from-scratch and Scikit-learn models achieved similar performance across accuracy, precision, recall, and F1-score. This indicates that the `from-scratch implementation` **correctly reproduces the behaviour of Elastic-Net Softmax Regression**.

# %% [markdown]
# ## 13: Effect of Regularization [3 Marks]
# 
# In this section, analyze the effect of different regularization techniques on the performance of the **Softmax Regression** model.
# 
# Train the model using the following three configurations and compare the results.
# 
# ---
# 
# ### Case 1: No Regularization
# 
# Train the model without any regularization.
# 
# **Hyperparameters**
# 
# - L1 Regularization ($\lambda_1$) = **0**
# - L2 Regularization ($\lambda_2$) = **0**
# 
# This serves as the baseline model.
# 
# ---
# 
# ### Case 2: L2 Regularization (Ridge)
# 
# Train the model using **only L2 Regularization**.
# 
# **Hyperparameters**
# 
# - L1 Regularization ($\lambda_1$) = **0**
# - L2 Regularization ($\lambda_2$) = **0.01**
# 
# Observe how L2 regularization affects the model performance and the learning curves.
# 
# ---
# 
# ### Case 3: Elastic-Net Regularization
# 
# Train the model using **both L1 and L2 Regularization**.
# 
# **Hyperparameters**
# 
# - L1 Regularization ($\lambda_1$) = **0.01**
# - L2 Regularization ($\lambda_2$) = **0.01**
# 
# Observe the combined effect of L1 and L2 regularization on the model.
# 
# ---
# 
# ### Compare the Following Performance Metrics
# 
# For each configuration, compute and compare:
# 
# - Accuracy
# - Precision
# - Recall
# - F1-Score
# - Training Loss
# - Validation Loss
# 
# Present your results in the following table.
# 
# | Model | Accuracy | Precision | Recall | F1-Score | Training Loss | Validation Loss |
# |--------|---------:|----------:|--------:|---------:|--------------:|----------------:|
# | No Regularization | | | | | | |
# | L2 Regularization | | | | | | |
# | Elastic-Net Regularization | | | | | | |
# 
# ---
# 
# 

# %%
# Case-1: No regularization
no_reg = ElasticNetSoftmaxRegression(learning_rate=0.01, n_epochs=1000, l1=0, l2=0)
no_reg.fit(x_train, y_train, x_val, y_val)
no_y_pred = no_reg.predict_class(x_test)
 
# Case-2: L2 (Ridge) regularization
l2_reg = ElasticNetSoftmaxRegression(learning_rate=0.01, n_epochs=1000, l1=0, l2=0.01)
l2_reg.fit(x_train, y_train, x_val, y_val)
l2_y_pred = l2_reg.predict_class(x_test)

# Case-3: Elastic-Net regularization
en_reg = ElasticNetSoftmaxRegression(learning_rate=0.01, n_epochs=1000, l1=0.01, l2=0.01)
en_reg.fit(x_train, y_train, x_val, y_val)
en_y_pred = en_reg.predict_class(x_test)

# %%
# Compare the performance metrics
no_accuracy, no_precision, no_recall, no_f1 = evaluate_model(y_true, no_y_pred)
l2_accuracy, l2_precision, l2_recall, l2_f1 = evaluate_model(y_true, l2_y_pred)
en_accuracy, en_precision, en_recall, en_f1 = evaluate_model(y_true, en_y_pred)

metrics_comparison = pd.DataFrame({
    'Model': ['No Regularization', 'L2 Regularization', 'Elastic-Net Regularization'],
    'Accuracy': [no_accuracy, l2_accuracy, en_accuracy],
    'Precision': [no_precision, l2_precision, en_precision],
    'Recall': [no_recall, l2_recall, en_recall],
    'F1-Score': [no_f1, l2_f1, en_f1],
    'Training Loss': [no_reg.train_loss[-1], l2_reg.train_loss[-1], en_reg.train_loss[-1]],
    'Validation Loss': [no_reg.val_loss[-1], l2_reg.val_loss[-1], en_reg.val_loss[-1]]
})
display(metrics_comparison)

# %% [markdown]
# ### 14. Analysis [1 Marks]
# 
# Based on the experimental results, answer the following questions.
# 
# 1. Which regularization technique achieved the highest classification accuracy?
# 
# 2. How did L2 regularization affect the training and validation losses compared to the model without regularization?
# 
# 3. What impact did adding L1 regularization have on the overall model performance?
# 
# 4. Which model showed the best generalization performance on the validation and test datasets?
# 
# 5. Which regularization technique would you recommend for this dataset? Justify your answer using the obtained results.

# %% [markdown]
# ***Answers***
# 
# 1. All three models achieved the same **100% accuracy**. Therefore, no regularization technique had an accuracy advantage.
# 2. `L2 regularization` increased the training loss from **0.0949 to 0.1238** and the validation loss from **0.1284 to 0.1564**.
# 3. `Adding L1 regularization` further increased the losses. The Elastic-Net model had a training loss of **0.2288** and validation loss of **0.2540**, while the classification metrics remained at **100%**.
# 4.  The `No Regularization` model showed the best generalization performance because it had the lowest validation loss (**0.1284**) while achieving **100% accuracy, precision, recall, and F1-score** on the test dataset.
# 5. `No Regularization` is recommended for this dataset. All three models achieved identical test performance, while the unregularized model had the lowest training and validation losses. Therefore, **regularization did not provide any observable performance benefit** for this dataset.

# %% [markdown]
# ---
# End of the Assignment


