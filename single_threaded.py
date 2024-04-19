import os
import time
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

total_time = 0
folder_name = args.folder_name
out_folder_name = args.out_folder_name + "_single_threaded"
if not os.path.exists(out_folder_name):
    os.makedirs(out_folder_name)
num_files = len(os.listdir(folder_name))
for file in tqdm(os.listdir(folder_name)):
    input_file = os.path.join(folder_name, file)
    input_image, _, _ = model.load_image(input_file)
    start = time.time()
    image_out_resized = model.segmentation(input_image)
    image_out_resized = np.reshape(image_out_resized, (1, 1440 // 2, 1920 // 2))
    print(image_out_resized.shape)
    colored_output = model.postprocess_map(image_out_resized)
    end = time.time()
    output_file = os.path.join(out_folder_name, file)
    colored_output.save(output_file)
    total_time += end - start
print(f'Total Time taken : {(total_time)*1000}ms')
print(f'Avg Time taken : {((total_time)*1000)/ num_files}ms')