import cv2
import os
import matplotlib.pyplot as plt

folder = r"C:\Users\jawad\Downloads\Processed_Dataset_2\01"
images = os.listdir(folder)[:5]  

for i, img_file in enumerate(images):
    img_path = os.path.join(folder, img_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.subplot(1, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

plt.show()
