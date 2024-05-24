import cv2
import numpy as np
import time
import os
import psutil
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd

start_memory = psutil.virtual_memory().used
start_time = time.time()

def Extreme(img, thresh):
    row, col, chanel = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    points = []
    newimage = np.zeros((row, col), np.uint8)
    for i in range(col):
        # Pmax = np.max(gray[:,i])
        Prow = np.argmax(gray[:, i])
        if Prow > thresh:
            points.append([Prow, i])
    for p in points:
        newimage[p[0], p[1]] = 255
        img[p[0], p[1], :] = [0, 0, 255]

    return img, newimage

if __name__ == "__main__":
    image_path = "path"
    save_path = "path"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    thresh_value = 150 

    image_files = os.listdir(image_path)

    processing_times = [] 
    memory_usage = []  

    for filename in sorted(os.listdir(image_path)):  
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join(image_path, filename))
            start_processing = time.time()
            processed_image, _ = Extreme(image, thresh_value)
            end_processing = time.time()
            processing_time = end_processing - start_processing
            processing_times.append(processing_time)

            cv2.imwrite(os.path.join(save_path, filename), processed_image)

            current_memory = psutil.virtual_memory().used
            memory_usage.append(current_memory)


    end_memory = psutil.virtual_memory().used
    end_time = time.time()


    memory_used = end_memory - start_memory
    memory_used_mb = memory_used / 1024 / 1024  


    avg_memory_usage = memory_used_mb / (end_time - start_time)


    smoothed_memory_usage = gaussian_filter1d(memory_usage, sigma=3)

    print(f"Average Memory Usage: {avg_memory_usage:.2f} MB/s")



    total_time = sum(processing_times)


    if len(processing_times) > 0:
        average_time = total_time / len(processing_times)
    else:
        average_time = 0

    print("Total processing time: ", total_time)
    print("Average processing time per image: ", average_time)


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(processing_times) + 1), processing_times, marker='o', linestyle='-')
    plt.title('Image Processing Time')
    plt.xlabel('Image Number')
    plt.ylabel('Processing Time (seconds)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'processing_time_plot.png'))
    plt.show()
