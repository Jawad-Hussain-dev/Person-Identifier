import joblib
from decision_tree import train_decision_tree
from explain_prediction import explain_prediction

# ✅ Load precomputed dataset (fast!)
X, y, label_map = joblib.load("hog_dataset.pkl")
print("✅ Dataset loaded from hog_dataset.pkl")

# ✅ Train decision tree as usual
model = joblib.load("tree_model.pkl")
print("✅ Model loaded from tree_model.pkl")

# 🔍 Test with a single image
test_image = r"C:\Users\jawad\Downloads\Processed_Dataset_2\03\frame_00045.jpg"
explain_prediction(model, test_image, label_map)
