import os
import sys
import time
import random
import numpy as np
import imgaug
import json
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import zipfile
import urllib.request
import shutil
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.model import log
from sampling import gmm_em
from sampling import sample

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class PulmonaryConfig(Config):
    # Give the configuration a recognizable name
    NAME = "pulmonary"
    # MODEL_TYPE has 2 choices, if you set it as MASK_RCNN, the code will run as the original code on Mask RCNN
    # if you set it as P_MASK_RCNN, it will run with P_mask_RCNN mode
    MODEL_TYPE = "P_MASK_RCNN"
    # DOWN_SAMPLE_STRIDE = [4, 8, 16, 32]
    # the sample stride for each feature map.
    # Notice: the stride should not be set too small, or it will cause memory overflow
    DOWN_SAMPLE_STRIDE = [1, 4, 4, 8]
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    UP_SAMPLING_STRATEGY = "bilinear"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 2000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 20

    MEAN_PIXEL = np.array([54.18, 54.18, 54.18])


class PulmonaryEmbolismDataset(utils.Dataset):
    def load_pulmonary_embolism(self, dataset_dir, subset, class_ids=None, return_pul=False):
        """
        :param dataset_dir:  the directory of the dataset
        :param subset: the subset of the dataset, Options: train, test, val
        :param class_ids: the indexes of the class, default None
        :param return_pul: whether to return the generated data set. default False
        """
        pul_emb = COCO("{}/annotations/instances_{}.json".format(dataset_dir, subset))

        image_dir = "{}/{}".format(dataset_dir, subset)
        print(dataset_dir, subset, image_dir)
        if not class_ids:
            # All classes
            class_ids = sorted(pul_emb.getCatIds())
        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(pul_emb.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(pul_emb.imgs.keys())
        for i in class_ids:
            self.add_class("pul_emb", i, pul_emb.loadCats(i)[0]["name"])
        for i in image_ids:
            # print(pul_emb.imgs[i])
            # print(os.path.join(image_dir, pul_emb.imgs[i]['file_name']))
            # ann_ids = pul_emb.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None)
            # print(ann_ids)
            # annotations = pul_emb.loadAnns(pul_emb.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None))
            # print(annotations)
            self.add_image(
                "pul_emb", image_id=i,
                path=os.path.join(image_dir, pul_emb.imgs[i]['file_name']),
                width=pul_emb.imgs[i]["width"],
                height=pul_emb.imgs[i]["height"],
                annotations=pul_emb.loadAnns(pul_emb.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        # generate the pixels coordinates that will be used to generate anchor
        self.generate_sample_points(dataset_dir)
        if return_pul:
            return pul_emb

    def generate_sample_points(self, dataset_dir, subset="train", save_path="locations/sample_points.txt",
                               sample_No=1000000):
        """
        :param dataset_dir: the directory of the dataset
        :param subset: the subset of the dataset, Options: train, test, val
        :param save_path: the path to save the sample points
        :param sample_No: the number of sample points
        """
        points = []
        # if the sample points file has exist, load it directly
        if os.path.exists(save_path):
            print("load sample points file")
            points = np.loadtxt(save_path)
        else:
            file_name = "{}/annotations/instances_{}.json".format(dataset_dir, subset)
            print("build GMM and sample points through " + file_name)
            f = open(file_name)
            data = json.load(f)
            annotations = data["annotations"]
            centers = []
            # get the center coordinate of each object
            for ann in annotations:
                bbox = ann["bbox"]
                y = bbox[0] + bbox[2] / 2
                x = bbox[1] + bbox[3] / 2
                centers.append([x, y])
            centers = np.asarray(centers)
            # build GMM
            gmm = gmm_em.GMM(centers)
            gmm.em_algorithm(100, 0.0001)
            print("mu: ", gmm.mu)
            print("sigma: ", gmm.sigma)
            print("alpha: ", gmm.alpha)
            alpha = np.array(gmm.alpha)
            mu = np.array(gmm.mu)
            sigma = np.array(gmm.sigma)
            print("sample from GMM")
            points = sample.sample_from_mixture_gaussian(alpha, mu, sigma, sample_No=sample_No)
            np.savetxt(save_path, points, fmt="%d")
        # print(points.shape)
        self.sample_points = points
        return points

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        # print(len(self.image_info))
        # print(self.image_info[0])
        if image_info["source"] != "pul_emb":
            return super(PulmonaryEmbolismDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "pul_emb.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            arr = mask.copy().astype(np.int8)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(PulmonaryEmbolismDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "pul_emb":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(PulmonaryEmbolismDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        s = segm[0]
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


# class Argument:
#     def __init__(self):
#         self.command = "training"
#         self.model = "coco"
#         self.dataset = "E:/dataset/pulmonary_embolism"
#         self.logs = os.path.join(ROOT_DIR, "logs")
#         self.download = False


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train P_Mask R-CNN on PE dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'inference'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/pe/",
                        help='Directory of the PE dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)
    # args = Argument()
    # print("Command: ", args.command)
    # print("Model: ", args.model)
    # print("Dataset: ", args.dataset)
    # print("Logs: ", args.logs)
    # print("Auto Download: ", args.download)

    dataset_dir = args.dataset
    subset = "train"
    # sample_points_save_path = "locations/sample_points.txt"
    train_pulmonary = PulmonaryEmbolismDataset()
    train_pulmonary.load_pulmonary_embolism(dataset_dir, subset)
    # train_pulmonary.generate_sample_points(dataset_dir)
    train_pulmonary.prepare()
    # print(train_pulmonary.sample_points)


    subset = "val"
    val_pulmonary = PulmonaryEmbolismDataset()
    val_pulmonary.load_pulmonary_embolism(dataset_dir, subset)
    val_pulmonary.prepare()

    subset = "test"
    test_pulmonary = PulmonaryEmbolismDataset()
    test_pulmonary.load_pulmonary_embolism(dataset_dir, subset)
    test_pulmonary.prepare()

    augmentation = imgaug.augmenters.Fliplr(0.5)

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    if args.command == "training":
        config = PulmonaryConfig()

        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs, sample_points=train_pulmonary.sample_points)
        # COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # print("Loading weights ", COCO_MODEL_PATH)
        # model.load_weights(COCO_MODEL_PATH, by_name=True,
        #                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
        #                             "mrcnn_bbox", "mrcnn_mask"])
        # print("finish")
        MODEL_PATH = args.model
        print("Loading weights from ", MODEL_PATH)

        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
        print("finish")

        print("Training network heads")
        model.train(train_pulmonary, val_pulmonary,
                    learning_rate=config.LEARNING_RATE,
                    epochs=20,
                    layers='heads',
                    augmentation=augmentation)
        print("finish")

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(train_pulmonary, val_pulmonary,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='4+',
                    augmentation=augmentation)

        print("Fine tune all layers")
        model.train(train_pulmonary, val_pulmonary,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=80,
                    layers='all',
                    augmentation=augmentation)
    elif args.command == "inference":
        class InferenceConfig(PulmonaryConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        inference_config = InferenceConfig()
        sample_points = np.loadtxt("locations/sample_points.txt")
        model = modellib.MaskRCNN(mode="inference",
                                  config=inference_config,
                                  model_dir=MODEL_DIR, sample_points=sample_points)
        print(model.model_dir)
        # model_path = "../../logs\pulmonary20190903T1624_standard\mask_rcnn_pulmonary_0080.h5"
        # model_path = model.find_last()
        # MODEL_PATH = "G:\longkun\pythonWorkspace\Mask_RCNN\logs\\a_pulmonary20190710T1738\mask_rcnn_pulmonary_0080.h5"
        # MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        MODEL_PATH = args.model
        print("Loading weights from ", MODEL_PATH)

        model.load_weights(MODEL_PATH, by_name=True)

        test_subset = test_pulmonary

        APs = []
        AP50s = []
        AP75s = []
        import time

        localtime1 = time.time()
        for image_id in test_subset.image_ids:
            # print(test_subset.image_info[image_id])
            original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(test_subset, inference_config,
                                       image_id, use_mini_mask=False)
            results = model.detect([original_image], verbose=1)
            r = results[0]
            AP_50, _, _, _ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                              r["class_ids"], r["scores"], r['masks'], iou_threshold=0.5)
            AP_75, _, _, _ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                              r["class_ids"], r["scores"], r['masks'], iou_threshold=0.75)
            AP = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"],
                                        r['masks'], verbose=0)
            APs.append(AP)
            AP50s.append(AP_50)
            AP75s.append(AP_75)
            print(AP, AP_50, AP_75)
            # ax = get_ax(1, 2)
            #             # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
            #             #                             train_pulmonary.class_names, figsize=(8, 8), ax=ax[0], title="real")
            #             # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
            #             #                             test_subset.class_names, r['scores'], figsize=(8, 8), ax=ax[1], title="result")
            #             # plt.show()
        localtime2 = time.time()
        t = localtime2 - localtime1
        print(t, len(test_subset.image_ids), t / len(test_subset.image_ids))
        print("AP:%f,   AP50:%f,    AP75:%f" % (np.mean(APs), np.mean(AP50s), np.mean(AP75s)))
        print(APs)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
