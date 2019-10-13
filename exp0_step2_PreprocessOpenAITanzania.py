from maskrcnn.preprocess.preprocess_openaitanzania import process_file

# define paths
IN_IMAGE_DIR = 'data/OpenAITanzania/GeoTIFF/'
IN_ANN_DIR = 'data/OpenAITanzania/GeoJSON/'
OUT_IMAGE_DIR = 'data/OpenAITanzania/Image/'
OUT_ANN_DIR = 'data/OpenAITanzania/Mask/'
# construct list of training file ids
file_ids = [os.path.basename(f).split('.')[0]
            for f in glob(os.path.join(IN_ANN_DIR, '*.geojson'))]
# parameters
CHIP_SIZE = 1600  # pixels
DOWN_RESOLUTION_FACTOR = 1.2  # resolution = this x 7.7cm
WINDOW_SIZE = int(CHIP_SIZE * DOWN_RESOLUTION_FACTOR)
SAMPLE_RATIO = 1.2
# process every file
for file_id in tqdm(file_ids):
    process_file(file_id)
