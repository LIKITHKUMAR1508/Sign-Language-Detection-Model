import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# DATA PATH
data_path = "AtoZ_3.1"

# LABEL MAPPING (same as your project)
letter_groups = {
    'A': 0, 'E': 0, 'M': 0, 'N': 0, 'S': 0, 'T': 0,
    'B': 1, 'D': 1, 'F': 1, 'I': 1, 'K': 1, 'R': 1, 'U': 1, 'V': 1, 'W': 1,
    'C': 2, 'O': 2,
    'G': 3, 'H': 3,
    'L': 4,
    'P': 5, 'Q': 5, 'Z': 5,
    'X': 6,
    'Y': 7, 'J': 7
}

# LOAD DATA
images = []
labels = []

for letter in os.listdir(data_path):
    folder = os.path.join(data_path, letter)
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)
            labels.append(letter_groups[letter])

X = np.array(images) / 255.0
y = np.array(labels)

print("Total images:", len(X))

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# DATA AUGMENTATION
datagen = ImageDataGenerator(
    rotation_range=23,
    zoom_range=0.25,
    width_shift_range=0.12,
    height_shift_range=0.12,
    horizontal_flip=True
)

datagen.fit(X_train)

# BUILD MODEL
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(8, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# TRAIN MODEL
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(X_test, y_test)
)

# SAVE MODEL
model.save("final_model.h5")

# EVALUATE
loss, acc = model.evaluate(X_test, y_test)
print(f"\nFinal Accuracy: {acc*100:.2f}%")


# import tensorflow as tf
# from tensorflow.keras import layers, models
# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split

# # 🔥 DATA PATH
# data_path = r"C:\Users\LIKITH KUMAR\OneDrive\Desktop\sign\AtoZ_3.1"

# # 🔥 LABEL MAPPING (same as your project)
# letter_groups = {
#     'A': 0, 'E': 0, 'M': 0, 'N': 0, 'S': 0, 'T': 0,
#     'B': 1, 'D': 1, 'F': 1, 'I': 1, 'K': 1, 'R': 1, 'U': 1, 'V': 1, 'W': 1,
#     'C': 2, 'O': 2,
#     'G': 3, 'H': 3,
#     'L': 4,
#     'P': 5, 'Q': 5, 'Z': 5,
#     'X': 6,
#     'Y': 7, 'J': 7
# }

# # 🔥 LOAD DATA
# images = []
# labels = []

# for letter in os.listdir(data_path):
#     folder = os.path.join(data_path, letter)
#     if os.path.isdir(folder):
#         for file in os.listdir(folder):
#             img_path = os.path.join(folder, file)
#             img = cv2.imread(img_path)

#             if img is None:
#                 continue

#             img = cv2.resize(img, (64, 64))
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#             images.append(img)
#             labels.append(letter_groups[letter])

# X = np.array(images) / 255.0
# y = np.array(labels)

# # 🔥 SPLIT DATA
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # 🔥 BUILD MODEL
# model = models.Sequential([
#     layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
#     layers.MaxPooling2D(2,2),

#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),

#     layers.Conv2D(128, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),

#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(8, activation='softmax')  # 8 groups
# ])

# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# # 🔥 TRAIN
# model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# # 🔥 SAVE MODEL
# model.save("new_model.h5")

# # 🔥 TEST ACCURACY
# loss, acc = model.evaluate(X_test, y_test)
# print(f"Final Accuracy: {acc*100:.2f}%")





        
# """
# Simple Model Accuracy Test
# Shows how 8-group classification achieves better accuracy than direct 26-class approach
# """

# import cv2
# import numpy as np
# import os
# from keras.models import load_model
# from sklearn.metrics import accuracy_score, classification_report
# import time

# def load_test_data(data_path='AtoZ_3.1', samples_per_letter=10):
#     """Load test samples from dataset"""
#     print("Loading test data...")
    
#     # Letter to group mapping based on your model's approach
#     letter_groups = {
#         'A': 0, 'E': 0, 'M': 0, 'N': 0, 'S': 0, 'T': 0,  # Group 0: AEMNST
#         'B': 1, 'D': 1, 'F': 1, 'I': 1, 'K': 1, 'R': 1, 'U': 1, 'V': 1, 'W': 1,  # Group 1
#         'C': 2, 'O': 2,  # Group 2: CO
#         'G': 3, 'H': 3,  # Group 3: GH
#         'L': 4,  # Group 4: L
#         'P': 5, 'Q': 5, 'Z': 5,  # Group 5: PQZ
#         'X': 6,  # Group 6: X
#         'Y': 7, 'J': 7   # Group 7: YJ
#     }
    
#     images = []
#     labels = []
    
#     for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
#         folder_path = os.path.join(data_path, letter)
#         if os.path.exists(folder_path):
#             files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
#             sample_files = files[:samples_per_letter]
            
#             for filename in sample_files:
#                 img_path = os.path.join(folder_path, filename)
#                 img = cv2.imread(img_path)
#                 if img is not None:
#                     img = cv2.resize(img, (400, 400))
#                     images.append(img)
#                     labels.append(letter_groups[letter])
    
#     print(f"Loaded {len(images)} test images")
#     return np.array(images), np.array(labels)

# def test_model_accuracy():
#     """Test the 8-group classification model"""
#     print("Testing Sign Language Model Accuracy")
#     print("=" * 50)
    
#     # Load the trained model
#     try:
#         model = load_model('cnn8grps_rad1_model.h5')
#         print("Model loaded successfully")
#     except:
#         print("Error: Could not load model file 'cnn8grps_rad1_model.h5'")
#         return
    
#     # Load test data
#     try:
#         X_test, y_test = load_test_data()
#         if len(X_test) == 0:
#             print("Error: No test data found. Check AtoZ_3.1 folder exists")
#             return
#     except:
#         print("Error: Could not load test data from AtoZ_3.1 folder")
#         return
    
#     # Normalize and predict
#     X_test = X_test.astype('float32') / 255.0
#     predictions = model.predict(X_test, verbose=0)
#     y_pred = np.argmax(predictions, axis=1)
    
#     # Calculate accuracy
#     accuracy = accuracy_score(y_test, y_pred)
    
#     print(f"\nModel Performance Results:")
#     print("-" * 30)
#     print(f"Total test samples: {len(X_test)}")
#     print(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
#     # Group-wise analysis
#     group_names = {
#         0: 'AEMNST (6 letters)',
#         1: 'BDFIKRUVW (9 letters)', 
#         2: 'CO (2 letters)',
#         3: 'GH (2 letters)',
#         4: 'L (1 letter)',
#         5: 'PQZ (3 letters)',
#         6: 'X (1 letter)',
#         7: 'YJ (2 letters)'
#     }
    
#     print(f"\nGroup-wise accuracy:")
#     print("-" * 40)
    
#     for group_id in range(8):
#         mask = y_test == group_id
#         if np.sum(mask) > 0:
#             group_acc = accuracy_score(y_test[mask], y_pred[mask])
#             correct = np.sum(y_test[mask] == y_pred[mask])
#             total = np.sum(mask)
#             print(f"Group {group_id} ({group_names[group_id]}): {correct}/{total} = {group_acc:.3f} ({group_acc*100:.1f}%)")
    
#     return accuracy

# def explain_approach():
#     """Explain the 8-group classification approach"""
#     print("\n" + "=" * 60)
#     print("Classification Approach Explanation")
#     print("=" * 60)
    
#     print("\nProblem with direct 26-class classification:")
#     print("- Training a CNN to directly classify 26 letters (A-Z) typically achieves 70-80% accuracy")
#     print("- Many letters have similar hand shapes, causing confusion")
#     print("- High complexity leads to overfitting on training data")
    
#     print("\nOur solution - 8-group classification:")
#     print("- Group similar letters together based on hand shape characteristics")
#     print("- Train CNN to classify into 8 groups instead of 26 individual letters")
#     print("- Use mathematical rules on hand landmarks to identify exact letter within each group")
    
#     print("\nThe 8 groups we defined:")
#     groups = {
#         "Group 0": ["A", "E", "M", "N", "S", "T"],
#         "Group 1": ["B", "D", "F", "I", "K", "R", "U", "V", "W"], 
#         "Group 2": ["C", "O"],
#         "Group 3": ["G", "H"],
#         "Group 4": ["L"],
#         "Group 5": ["P", "Q", "Z"],
#         "Group 6": ["X"],
#         "Group 7": ["Y", "J"]
#     }
    
#     for group_name, letters in groups.items():
#         print(f"{group_name}: {', '.join(letters)}")
    
#     print("\nTwo-stage classification process:")
#     print("1. CNN predicts which of the 8 groups the hand gesture belongs to")
#     print("2. Mathematical analysis of hand landmark positions determines the specific letter within that group")
    
#     print("\nAdvantages of this approach:")
#     print("- Reduced complexity: 8 classes vs 26 classes")
#     print("- Better generalization and less overfitting") 
#     print("- Higher accuracy due to more focused learning")
#     print("- Maintains real-time performance")

# def save_results(accuracy):
#     """Save results to a simple text file"""
#     with open('accuracy_results.txt', 'w') as f:
#         f.write("Sign Language Recognition Model - Accuracy Test Results\n")
#         f.write("=" * 55 + "\n\n")
#         f.write("Approach: 8-Group Classification\n")
#         f.write(f"Achieved Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
#         f.write("Method:\n")
#         f.write("- Grouped 26 letters into 8 similar classes\n")
#         f.write("- Used CNN for group prediction + mathematical rules for final classification\n")
#         f.write("- This approach achieves significantly higher accuracy than direct 26-class classification\n")
    
#     print(f"Results saved to accuracy_results.txt")

# if __name__ == "__main__":
#     start_time = time.time()
    
#     # Test the model
#     accuracy = test_model_accuracy()
    
#     if accuracy:
#         # Explain the approach
#         explain_approach()
        
#         # Save results
#         save_results(accuracy)
        
#         end_time = time.time()
        
#         print(f"\n" + "=" * 50)
#         print("Summary:")
#         print(f"8-group classification accuracy: {accuracy*100:.2f}%")
#         print(f"Test completed in {end_time - start_time:.1f} seconds")
#         print("=" * 50)
        
#         if accuracy >= 0.97:
#             print("Result: Target accuracy of 97%+ achieved")
#         else:
#             print("Result: Accuracy below 97% target")
#     else:
#         print("Test failed - check model and data files")







# import cv2 as cv
# import numpy as np
# import os
# from keras.models import load_model
# from sklearn.metrics import accuracy_score

# def load_test_data(data_path='AtoZ_3.1', samples_per_letter=10):
#     print("Loading test data...")

#     letter_groups = {
#         'A': 0, 'E': 0, 'M': 0, 'N': 0, 'S': 0, 'T': 0,
#         'B': 1, 'D': 1, 'F': 1, 'I': 1, 'K': 1, 'R': 1, 'U': 1, 'V': 1, 'W': 1,
#         'C': 2, 'O': 2,
#         'G': 3, 'H': 3,
#         'L': 4,
#         'P': 5, 'Q': 5, 'Z': 5,
#         'X': 6,
#         'Y': 7, 'J': 7
#     }

#     images = []
#     labels = []

#     for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
#         folder_path = os.path.join(data_path, letter)

#         if os.path.exists(folder_path):
#             files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
#             sample_files = files[:samples_per_letter]

#             for filename in sample_files:
#                 img_path = os.path.join(folder_path, filename)

#                 img = cv.imread(img_path)
#                 if img is None:
#                     continue

#                 # 🔥 IMPORTANT FIXES
#                 img = cv.cvtColor(img, cv.COLOR_BGR2RGB)   # FIX 1: BGR → RGB
#                 img = cv.resize(img, (400, 400))           # FIX 2: match training size
#                 img = img.astype('float32') / 255.0        # FIX 3: normalization

#                 images.append(img)
#                 labels.append(letter_groups[letter])

#     print(f"Loaded {len(images)} test images")
#     return np.array(images), np.array(labels)


# def test_model():
#     print("Testing Model...")
#     print("=" * 40)

#     # Load model
#     model = load_model('cnn8grps_rad1_model.h5')
#     print("Model loaded successfully")

#     # Load test data
#     X_test, y_test = load_test_data()

#     if len(X_test) == 0:
#         print("No test data found!")
#         return

#     # Predictions
#     predictions = model.predict(X_test, verbose=0)
#     y_pred = np.argmax(predictions, axis=1)

#     # 🔍 DEBUG (VERY IMPORTANT)
#     print("\nDebug Info:")
#     print("Unique predictions:", np.unique(y_pred))
#     print("Sample prediction probs:", predictions[0])

#     # Accuracy
#     accuracy = accuracy_score(y_test, y_pred)

#     print("\nResults:")
#     print("-" * 30)
#     print(f"Total samples: {len(X_test)}")
#     print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

#     # Group-wise accuracy
#     print("\nGroup-wise accuracy:")
#     print("-" * 30)

#     for group in range(8):
#         mask = y_test == group
#         if np.sum(mask) > 0:
#             acc = accuracy_score(y_test[mask], y_pred[mask])
#             correct = np.sum(y_test[mask] == y_pred[mask])
#             total = np.sum(mask)
#             print(f"Group {group}: {correct}/{total} = {acc:.2f}")

#     print("\nDone.")


# if __name__ == "__main__":
#     test_model()