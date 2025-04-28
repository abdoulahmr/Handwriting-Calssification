import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set your dataset paths
train_folder = 'dataset/train'
test_folder = 'dataset/test'

def extract_hog_features(img):
    """Extract HOG features from an image."""
    fd, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return fd

def load_images_from_folder(folder):
    """Load images from a given folder, extracting HOG features and labels."""
    X = []
    y = []
    paths = []  # Store image paths for later use
    for label in sorted(os.listdir(folder)):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize image
                    img_resized = cv2.resize(img, (64, 64))
                    # Extract HOG features
                    features = extract_hog_features(img_resized)
                    X.append(features)
                    y.append(int(label))
                    paths.append(img_path)  # Store path for each image
    return np.array(X), np.array(y), paths

def euclidean_distance(x1, x2):
    """Compute the Euclidean distance between two feature vectors."""
    return np.linalg.norm(x1 - x2)

def classify_with_euclidean_distance(X_train, y_train, X_test):
    """Classify test data based on the Euclidean distance to the training data."""
    y_pred = []
    
    for test_sample in X_test:
        # Compute the distance from the test sample to all the training samples
        distances = [euclidean_distance(test_sample, train_sample) for train_sample in X_train]
        
        # Find the index of the closest training sample
        closest_index = np.argmin(distances)
        
        # Assign the label of the closest training sample
        y_pred.append(y_train[closest_index])
    
    return np.array(y_pred)

# Load training and testing data
X_train, y_train, train_paths = load_images_from_folder(train_folder)
X_test, y_test, test_paths = load_images_from_folder(test_folder)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Perform classification using Euclidean distance
y_pred = classify_with_euclidean_distance(X_train_pca, y_train, X_test_pca)

# Evaluate the model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Identify misclassified images
misclassified_indices = np.where(y_pred != y_test)[0]

# Display the misclassified images
for index in misclassified_indices:
    img_path = test_paths[index]  # Get the path for the misclassified image
    
    # Load and resize the image for better visualization
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (64, 64))

    # Plot the misclassified image
    plt.imshow(img_resized, cmap='gray')
    plt.title(f"True label: {y_test[index]}, Predicted: {y_pred[index]}")
    plt.axis('off')  # Hide axis
    plt.show()
