import os
import time
import queue
import threading
from PIL import Image
import numpy as np
from tqdm import tqdm
from hrnet_semantic_segmentation_tensorrt import HRNetSemanticSegmentationTensorRT, get_custom_hrnet_args
import mapillary_visualization as mapillary_visl

# Get hrnet arguments
args = get_custom_hrnet_args()

# Override it to our engine path
args.engine_file_path = "./hrnet.engine"
args.image_width = 960
args.image_height = 720
args.folder_name = "./image_series"
args.out_folder_name = "./image_series_out"
args.dummy_image_path = "./image_series/1713298962.137851000.jpg"

model = HRNetSemanticSegmentationTensorRT(args)
seg_color_fn = mapillary_visl.apply_color_map

folder_name = args.folder_name
out_folder_name = args.out_folder_name + "_multi_threaded_right"
img_queue = queue.Queue(maxsize=20)

def target_funct():
    total_time = 0
    if not os.path.exists(out_folder_name):
        os.makedirs(out_folder_name)
    num_files = len(os.listdir(folder_name))
    for file in tqdm(os.listdir(folder_name)):
        input_file = os.path.join(folder_name, file)
        input_image, _, _ = model.load_image(input_file)
        img_queue.put((input_image, file))

def process_frame():
    if img_queue.empty():
        return
    input_image, file = img_queue.get()
    image_out_resized = model.segmentation(input_image)
    image_out_resized = np.reshape(image_out_resized, (1, 1440 // 2, 1920 // 2))
    colored_output = model.postprocess_map(image_out_resized)
    output_file = os.path.join(out_folder_name, file)
    colored_output.save(output_file)

new_thread = threading.Thread(target=target_funct)
new_thread.start() 

while True:
    process_frame()
