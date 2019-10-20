import os
import json
import skimage.draw
import time

ROOT_DIR = os.path.abspath('../')
DATASET_DIR = os.path.join(ROOT_DIR, 'datasets/big_deepfashion2')
#DATASET_DIR = os.path.join(ROOT_DIR, 'datasets/tiny_deepfashion2')
subsets = ['val','train']

for subset in subsets:

    # For tracking progress
    start = time.time()
    progress = 0

    print("Preprocessing of {} started. Please wait...".format(subset))
    dataset_dir = os.path.join(DATASET_DIR, subset)
    dataset_dir_image = os.path.join(dataset_dir, 'image')
    dataset_dir_annos = os.path.join(dataset_dir, 'annos')
    dataset_dir_annos_preprocessed = os.path.join(dataset_dir, 'annos_preprocessed')

    #Create folder for preprocessed.
    if(not os.path.exists(dataset_dir_annos_preprocessed)):
        os.makedirs(dataset_dir_annos_preprocessed)

    annotations = [(pos_json.split('.')[0], json.load(open(os.path.join(
        dataset_dir_annos, pos_json)))) for pos_json in os.listdir(dataset_dir_annos)]

    preprocessed_annotations = []

    print("Working on {} annotations".format(len(annotations)))

    # Add images
    for image_id, a in annotations:
        # load_mask() needs the image size to convert polygons to masks.
        # Unfortunately, VIA doesn't include it in JSON, so we must read
        # the image. This is only managable since the dataset is tiny.
        image_path = os.path.join(dataset_dir_image, image_id + '.jpg')
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]

        image_annotations = {
            'id': image_id,
            'width': width,
            'height': height
        }

        clothes_masks = []  # [[[x1, y1, ..., xn yn]]]
        categories = []

        for key in a:
            if key.startswith('item'):
                clothes_masks.append(a[key]['segmentation'])
                categories.append(a[key]['category_id'])

        clothing_segmentations = []

        for clothing_i, segmentations in enumerate(clothes_masks):
            preprocessed_segmentations = []
            for i, segmentation in enumerate(segmentations):
                all_x_points = [x for idx, x in enumerate(
                    segmentation) if idx % 2 == 0]
                all_y_points = [y for idx, y in enumerate(
                    segmentation) if idx % 2 == 1]
                preprocessed_segmentations.append({
                    'all_x_points': all_x_points,
                    'all_y_points': all_y_points
                })
            clothing_segmentations.append({
                'category_id': categories[clothing_i],
                'segmentations': preprocessed_segmentations
            })

        image_annotations['clothes'] = clothing_segmentations
        preprocessed_annotations.append(image_annotations)

                #To keep user updated on progress
        progress += 1
        if(progress%100 == 0 or progress == len(annotations)):
            completed = 100*progress/len(annotations)
            remaining = 100 - completed
            end = time.time()
            elapsed = end - start
            ETA= (elapsed / completed)*(remaining)
            print("{0:.2f}% done".format(completed))
            print("\t{0:.2f} minutes left (estimate)".format(ETA/60))

    print("Dumping annotations to file...")
    with open(os.path.join(dataset_dir_annos_preprocessed, 'annotations.json'), 'w') as f:
        json.dump(preprocessed_annotations, f)

print("Done, after {} hours".format((end-start)/3600))
