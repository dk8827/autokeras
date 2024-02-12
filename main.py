import autokeras as ak
import numpy as np

# Function to load data from the text file
def load_data(filename):
    reviews = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            review, label = line.rsplit(',', 1)
            reviews.append(review.strip())
            labels.append(int(label.strip()))
    return reviews, labels

# Load training data
x_train, y_train = load_data('movie_reviews_data.txt')

# Convert lists to NumPy arrays
x_train = np.array(x_train, dtype=object)
y_train = np.array(y_train)

# Initialize the TextClassifier
clf = ak.TextClassifier(
    overwrite=True,
    max_trials=2  # Number of models to try
)

# Fit the model on the training data
clf.fit(x_train, y_train, epochs=8, batch_size=1)

# Assuming you have a test set to evaluate the model
x_test = [
    "Didn't like it at all, very boring.",
    "Amazing.It was a fantastic journey through cinema.",
    # Add more test samples as needed
]

y_test = [0, 1]  # Expected sentiments for the test set

# Convert lists to NumPy arrays for the test set as well
x_test = np.array(x_test, dtype=object)
y_test = np.array(y_test)

# Evaluate the trained model on the test set
loss, accuracy = clf.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy}")

# Get the best model
model = clf.export_model()

# Print the model architecture
model.summary()
