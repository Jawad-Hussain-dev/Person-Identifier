from dataset_builder import build_dataset

dataset_path = r"C:\Users\jawad\Downloads\Processed_Dataset_2"
X, y, label_map = build_dataset(dataset_path)

print(f"Total samples: {len(X)}")
print(f"Label mapping: {label_map}")
