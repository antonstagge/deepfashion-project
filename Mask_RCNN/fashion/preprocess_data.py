import os
import json
import skimage.draw

ROOT_DIR = os.path.abspath('../')
DATASET_DIR = os.path.join(ROOT_DIR, 'datasets/tiny_deepfashion2')
subsets = ['train', 'val']

for subset in subsets:
    dataset_dir = os.path.join(DATASET_DIR, subset)
    dataset_dir_image = os.path.join(dataset_dir, 'image')
    dataset_dir_annos = os.path.join(dataset_dir, 'annos')
    dataset_dir_annos_preprocessed = os.path.join(
        dataset_dir, 'annos_preprocessed')

    annotations = [(pos_json.split('.')[0], json.load(open(os.path.join(
        dataset_dir_annos, pos_json)))) for pos_json in os.listdir(dataset_dir_annos)]

    preprocessed_annotations = []

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
        data_landmarks = []

        for key in a:
            if key.startswith('item'):
                clothes_masks.append(a[key]['segmentation'])
                categories.append(a[key]['category_id'])
                data_landmarks.append(a[key]['landmarks'])

        clothing_segmentations = []

        for clothing_i, segmentations in enumerate(clothes_masks):
            # Mask
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

            # Landmark
            data_landmark = data_landmarks[clothing_i]
            all_x_points = [x for idx, x in enumerate(
                data_landmarks[clothing_i]) if idx % 3 == 0]
            all_y_points = [y for idx, y in enumerate(
                data_landmarks[clothing_i]) if idx % 3 == 1]
            all_v_points = [v for idx, v in enumerate(
                data_landmarks[clothing_i]) if idx % 3 == 2]

            landmark = {
                'all_x_points': all_x_points,
                'all_y_points': all_y_points,
                'all_v_points': all_v_points
            }

            clothing_segmentations.append({
                'category_id': categories[clothing_i],
                'segmentations': preprocessed_segmentations,
                'landmark': landmark
            })

        image_annotations['clothes'] = clothing_segmentations
        preprocessed_annotations.append(image_annotations)

        with open(os.path.join(dataset_dir_annos_preprocessed, 'annotations.json'), 'w') as f:
            json.dump(preprocessed_annotations, f)
