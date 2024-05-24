import os
import cv2
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

start_memory = psutil.virtual_memory()
start_time = time.time()

def GravityCen(img_path, output_folder):

    img = cv2.imread(img_path)
    row, col, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    points = []
    newimage = np.zeros((row, col), np.uint8)
    threshold = 8 
    
    for i in range(col):
        pos = np.argmax(gray[:, i])
        Pmax = gray[pos, i]
        Pmin = np.min(gray[:, i])
        
        if Pmax == Pmin:
            continue
        
        sum = 0.0
        down = 0.0
        

        for j in range(-threshold, threshold+1):
            colp = pos + j

            if colp < 0 or colp >= row:
                continue
            sum += colp * gray[colp, i]
            down += gray[colp, i]
        
        Prow = sum / down
        points.append([Prow, i])
        
    for p in points:
        pr, pc = map(int, p)
        newimage[pr, pc] = 255
        img[pr, pc, :] = [0, 255, 0]
    

    img_name = os.path.basename(img_path)
    processed_img_path = os.path.join(output_folder, img_name)
    cv2.imwrite(processed_img_path, newimage)
    
    return points

if __name__ == "__main__":
    start_time = time.time()
    input_folder = "path"
    output_folder = "path"

    processing_times = []  
    memory_usage = []  
    

    image_files = os.listdir(input_folder)


    for img_file in os.listdir(input_folder):

        img_path = os.path.join(input_folder, img_file)
        

        processing_start_time = time.time()
        

        points = GravityCen(img_path, output_folder)


        processing_end_time = time.time()
        

        processing_time = processing_end_time - processing_start_time
        processing_times.append(processing_time)     
        
        print("Processed image:", img_file)
        print("Processing time for this image:", processing_time)

        current_memory = psutil.virtual_memory()
        memory_usage.append(current_memory.used)


    end_memory = psutil.virtual_memory()
    end_time = time.time()


    memory_used_start = start_memory.used
    memory_used_end = end_memory.used
    memory_used_diff = memory_used_end - memory_used_start
    memory_used_mb = memory_used_diff / 1024 / 1024  


    avg_memory_usage = memory_used_mb / (end_time - start_time)

    print(f"Average Memory Usage: {avg_memory_usage:.2f} MB/s")   
    

    total_time = time.time() - start_time
    

    if len(processing_times) > 0:
        average_time_per_image = sum(processing_times) / len(processing_times)
    else:
        average_time_per_image = 0
    
    print("Total time: ", total_time)
    print("Average time per image: ", average_time_per_image)


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(processing_times) + 1), processing_times, marker='o', linestyle='-')
    plt.title('Image Processing Time')
    plt.xlabel('Image Number')
    plt.ylabel('Processing Time (seconds)')
    plt.tight_layout()
    plt.show()
