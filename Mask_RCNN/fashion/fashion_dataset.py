import os
import json
import numpy as np
import skimage.draw
from mrcnn import utils

class FashionDataset(utils.Dataset):

    def load_fashion(self, dataset_dir, subset):
        """Load a subset of the deepfashion2 dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("fashion", 1, "short sleeve top")
        self.add_class("fashion", 2, "long sleeve top")
        self.add_class("fashion", 3, "short sleeve outwear")
        self.add_class("fashion", 4, "long sleeve outwear")
        self.add_class("fashion", 5, "vest")
        self.add_class("fashion", 6, "sling")
        self.add_class("fashion", 7, "shorts")
        self.add_class("fashion", 8, "trousers")
        self.add_class("fashion", 9, "skirt")
        self.add_class("fashion", 10, "short sleeve dress")
        self.add_class("fashion", 11, "long sleeve dress")
        self.add_class("fashion", 12, "vest dress")
        self.add_class("fashion", 13, "sling dress")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        dataset_dir_image = os.path.join(dataset_dir, 'image')
        dataset_dir_annos = os.path.join(dataset_dir, 'annos_preprocessed')
        
        """
        [
            {
                id,
                width,
                height,
                clothes: [
                    {
                        category_id,
                        segmentations: [
                            {
                                all_x_points: [],
                                all_y_points: [],
                            }
                        ],
                        landmark: {
                            'all_x_points': [],
                            'all_y_points': [],
                            'all_v_points': [],
                        }
                    }
                ]
            }
        ]
        """
        
        
        annotations = json.load(open(os.path.join(dataset_dir_annos, 'annotations.json')))

        # Add images
        for annotation in annotations:
            image_id = annotation['id']
            image_path = os.path.join(dataset_dir_image, image_id + '.jpg')
            
            self.add_image(
                "fashion",
                image_id=image_id,  # use file name as a unique image id
                path=image_path,
                width=annotation['width'],
                height=annotation['height'],
                clothes=annotation['clothes']
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        landmarks: A int array of shape [height, width, instance count] with
            one landmark per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a fashion dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "fashion":
            return super(self.__class__, self).load_mask(image_id)

        # Convert segmentations to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["clothes"])],
                        dtype=np.uint8)
        landmark = np.zeros([info["height"], info["width"], len(info["clothes"])],
                dtype=np.uint8)
        class_ids = np.zeros(len(info['clothes']), dtype=np.uint8)
        for clothing_idx, cloth in enumerate(info["clothes"]):
            class_ids[clothing_idx] = cloth['category_id']
            for segmentation_idx, segmentation in enumerate(cloth['segmentations']):
                # Get indexes of pixels inside the polygon and set them to 1
                rr, cc = skimage.draw.polygon(
                    segmentation['all_y_points'],
                    segmentation['all_x_points'])
                mask[rr, cc, clothing_idx] = 1
            landmark[
                cloth['landmark']['all_y_points'],
                cloth['landmark']['all_x_points'],
                clothing_idx] = cloth['landmark']['all_v_points']

        # Return mask, and array of class IDs of each instance.
        return mask, class_ids, landmark

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "fashion":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
    def load_image(self, image_id):
        info = self.image_info[image_id]
        return skimage.io.imread(info['path'])
