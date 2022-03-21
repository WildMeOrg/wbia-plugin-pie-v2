# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import os.path as osp
import json
import imageio
import numpy as np
from skimage import img_as_ubyte
from skimage import transform as skimage_transform

from .dataset import ImageDataset
from collections import defaultdict

from .data_manipulation import increase_bbox, to_origin, resize_coords


class COCODataset(ImageDataset):
    """Create dataset from COCO annotations:
    Preprocess data for faster loading during training.
    Preprocessed data is stored alongside original data.

    Input:
        name (str): name of the dataset to reference dataset in config
        dataset_dir (str): name of the directory for dataset files
        dataset_url (str): url to donwload the dataset from
        split (string): name of the annotated subset, e.g. train2020, test2020
        root (str): root directory for dataset folders
        crop (bool): if True crop images by bounding boxes from annotations
        resize (bool): if True resize images to a unified size
        imsize (int): min side of the image to resize during preprocessing
        train_min_samples (int): min number of images per name for train set
        test_min_samples (int): min number of images per name for test set
        split_test (bool): if True, use separate data for testing
            Default False
        id_attr (list of str): list of annotation attributes to use as
            a unique name. Default ['name']
        viewpoint_list (str): path to a csv file with viewpoint annotations
        debut (bool): if True only first 300 images are preprocessed
            Default False
        excluded_names (list of str): list of names to exclude from data
        flip_test (bool): if True, flip test images horizintally
            Default False
    """

    def __init__(
        self,
        name,
        dataset_dir,
        dataset_url,
        split,
        root,
        crop,
        resize,
        imsize,
        train_min_samples,
        test_min_samples=None,
        split_test=None,
        id_attr=['name'],
        viewpoint_list=None,
        debug=False,
        viewpoint_csv=None,
        excluded_names=None,
        flip_test=False,
        **kwargs
    ):
        # Prepare directories
        self.name = name
        self.split = split
        self.split_test = split_test
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = dataset_dir
        self.dataset_dir_orig = osp.join(self.root, dataset_dir)
        self.dataset_dir_proc = osp.join(self.root, dataset_dir, 'processed')
        self.crop = crop
        self.resize = resize
        self.id_attr = id_attr
        self.debug = debug
        self.viewpoint_csv = viewpoint_csv
        self.viewpoint_list = viewpoint_list

        # Set directories and filenames for preprocessed annotations
        self.set_prep_annots()
        self.set_prep_images()

        # Preprocess and load dataset
        if self._preproc_db_exists(self.prep_images, self.prep_annots):
            # Load preprocessed dataset
            self.db = self._get_preproc_db(self.prep_annots)
        else:
            # Download dataset if not downloaded before
            if dataset_url is not None:
                self.download_dataset(self.dataset_dir_orig, dataset_url)
            # Load COCO annots and preproc
            db_coco = self._get_coco_db(self.split)
            self.db = self._preproc_db(
                db_coco=db_coco,
                split=self.split,
                annots_filename=self.prep_annots,
                expand=1.0,
                min_size=imsize,
                fliplr=False,
                viewpoint_list=self.viewpoint_list,
                excluded_names=excluded_names,
            )

        print(
            '=> load {} samples from {} / {} dataset'.format(
                len(self.db), self.name, self.split
            )
        )

        if self.split_test is not None:
            if self._preproc_db_exists(self.prep_images_test, self.prep_annots_test):
                self.db_test = self._get_preproc_db(self.prep_annots_test)
            else:
                # Load COCO test annots and preproc
                db_coco_test = self._get_coco_db(self.split_test)
                self.db_test = self._preproc_db(
                    db_coco=db_coco_test,
                    split=self.split_test,
                    annots_filename=self.prep_annots_test,
                    expand=1.0,
                    min_size=imsize,
                    fliplr=flip_test,
                    viewpoint_list=self.viewpoint_list,
                    excluded_names=excluded_names,
                )
            print(
                '=> load {} samples from {} / {} dataset'.format(
                    len(self.db_test), self.name, self.split_test
                )
            )

        if self.split_test is None:
            train, test = self._split_train_test_min_samples(
                data=self.db, name_idx='id_name', train_min_samples=train_min_samples
            )
        else:
            train = self._filter_min_samples(
                data=self.db, name_idx='id_name', min_samples=train_min_samples
            )
            test = self._filter_min_samples(
                data=self.db_test, name_idx='id_name', min_samples=test_min_samples
            )

        train = self.db_to_tuples(train, relabel=True, name_key='id_name')
        test = self.db_to_tuples(test, relabel=False, name_key='id_name')

        super(COCODataset, self).__init__(train, test, **kwargs)

    def set_prep_images(self):
        self.prep_images = osp.join(self.dataset_dir_proc, 'images', self.split)
        if self.split_test is not None:
            self.prep_images_test = osp.join(
                self.dataset_dir_proc, 'images', self.split_test
            )

    def set_prep_annots(self):
        self.prep_annots = osp.join(
            self.dataset_dir_proc, 'annots', '{}.json'.format(self.split)
        )
        if self.split_test is not None:
            self.prep_annots_test = osp.join(
                self.dataset_dir_proc, 'annots', '{}.json'.format(self.split_test)
            )

    def _preproc_db_exists(self, prep_images, prep_annots):
        """ Check if preprocessed dataset exists in a directory"""
        if osp.exists(prep_images) and osp.exists(prep_annots):
            return True
        else:
            os.makedirs(prep_images, exist_ok=True)
            os.makedirs(osp.split(prep_annots)[0], exist_ok=True)
            return False

    def _get_preproc_db(self, annots):
        """ Load annotation file for preprocessed database """
        with open(annots) as file:
            db = json.load(file)
        return db

    def _get_coco_db(self, split):
        """ Get database from COCO anntations """
        ann_file = osp.join(
            self.dataset_dir_orig,
#            '{}.coco'.format(self.name),
            'annotations',
            'instances_{}.json'.format(split),
        )
        if not osp.exists(ann_file):
            # below path is used on automated datasets
            ann_file = osp.join(
                self.dataset_dir,
                'annotations',
                'instances_{}.json'.format(split),
            )

        dataset = json.load(open(ann_file, 'r'))

        # Get image metadata
        imgs = {}
        if 'images' in dataset:
            for img in dataset['images']:
                imgs[img['id']] = img

        # Get annots for images from annotations and parts
        imgToAnns = defaultdict(list)
        if 'annotations' in dataset:
            for ann in dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)

        if 'parts' in dataset:
            for ann in dataset['parts']:
                imgToAnns[ann['image_id']].append(ann)

        image_set_index = list(imgs.keys())
        print('=> Found {} images in {}'.format(len(image_set_index), ann_file))

        # Get viewpoint annotations from a separate csv
        if self.viewpoint_csv is not None:
            uuid2view = np.genfromtxt(
                self.viewpoint_csv, dtype=str, skip_header=1, delimiter=','
            )
            print(
                '=> Found {} view annotations in {}'.format(
                    len(uuid2view), self.viewpoint_csv
                )
            )
            uuid2view = {a[0]: a[1] for a in uuid2view}
        else:
            uuid2view = None

        # Collect ground truth annotations
        gt_db = []
        for index in image_set_index:
            img_anns = imgToAnns[index]
            image_path = self._get_image_path(imgs[index]['file_name'])
            gt_db.extend(self._load_image_annots(img_anns, image_path, uuid2view))
        return gt_db

    def _load_image_annots(self, img_anns, image_path, uuid2view=None):
        """ Get COCO annotations for an image by index """
        rec = []
        for i, obj in enumerate(img_anns):
            if uuid2view is not None:
                viewpoint = uuid2view.get(obj['uuid'], 'None')
            else:
                viewpoint = obj['viewpoint']
            rec.append(
                {
                    'image_path': image_path,
                    'aa_bbox': obj['bbox'],
                    'name': obj['name'],
                    'viewpoint': viewpoint,
                    'obj_id': i,
                }
            )
        return rec

    def _get_image_path(self, filename):
        """ Get full path to image in COCO annotations by image filename """
        image_path = osp.join(
            self.dataset_dir_orig,
#            '{}.coco'.format(self.name),
            'images',
            self.split,
            filename,
        )
        return image_path

    def _preproc_db(
        self,
        db_coco,
        split,
        annots_filename,
        expand,
        min_size,
        fliplr,
        viewpoint_list,
        excluded_names,
        print_freq=100,
    ):
        """Preprocess images by cropping area around bounding box
        and resizing to a smaller size for faster augmentation and loading
        """
        print('Preprocessing database...')
        prep_gt_db = []
        excluded_records = []
        for i, db_rec in enumerate(db_coco):
            # Keep only records from specific viewpoints
            # if viewpoint_list is not None and db_rec['viewpoint'] not in viewpoint_list:
            #     excluded_records.append(db_rec)
            #     continue

            # # Exclude names from the name exclusion list
            # if excluded_names is not None and db_rec['name'] in excluded_names:
            #     excluded_records.append(db_rec)
            #     continue

            # Read image
            image = imageio.imread(db_rec['image_path'])
            if image is None:
                print('=> fail to read {}'.format(db_rec['image_path']))
                raise ValueError('Fail to read {}'.format(db_rec['image_path']))

            aa_bbox = db_rec['aa_bbox']

            if self.crop:
                # Get box around axis-aligned bounding box
                x1, y1, bw, bh = increase_bbox(
                    aa_bbox, expand, image.shape[:2], type='xyhw'
                )

                # Crop image and coordinates
                image_cropped = image[y1 : y1 + bh, x1 : x1 + bw]
                if min(image_cropped.shape) < 1:
                    print(
                        'Skipped image {} Cropped to zero size.'.format(
                            db_rec['image_path']
                        )
                    )
                    continue
                else:
                    image = image_cropped

                # Shift coordinates to new origin
                aa_bbox = to_origin(aa_bbox, (x1, y1))

            if self.resize:
                # Compute output size
                if image.shape[0] <= image.shape[1]:
                    out_size = (min_size, int(image.shape[1] * min_size / image.shape[0]))
                else:
                    out_size = (int(image.shape[0] * min_size / image.shape[1]), min_size)

                # Resize coordinates
                aa_bbox = resize_coords(aa_bbox, image.shape[:2], out_size)

                # Resize image
                image = skimage_transform.resize(
                    image, out_size, order=3, anti_aliasing=True
                )

            if fliplr:
                image = np.fliplr(image)

            # Save image to processed folder
            im_filename = osp.basename(db_rec['image_path'])
            new_filename = osp.join(
                self.dataset_dir_proc,
                'images',
                split,
                '{}_{}{}'.format(
                    osp.splitext(im_filename)[0],
                    db_rec['obj_id'],
                    osp.splitext(im_filename)[1],
                ),
            )
            imageio.imwrite(new_filename, img_as_ubyte(image))

            prep_gt_db.append(
                {
                    'image_path': new_filename,
                    'aa_bbox': aa_bbox,
                    'obj_id': db_rec['obj_id'],
                    'name': db_rec['name'],
                    'viewpoint': db_rec['viewpoint'],
                    'id_name': '_'.join([db_rec[att] for att in self.id_attr]),
                }
            )

            if i % print_freq == 0:
                print('Processed {} images'.format(i))

            if self.debug and i >= 300:
                break

        # Save as json
        with open(annots_filename, 'w', encoding='utf-8') as f:
            json.dump(prep_gt_db, f, ensure_ascii=False, indent=4)

        print('Excluded {} records'.format(len(excluded_records)))

        return prep_gt_db

    def _filter_min_samples(self, data, name_idx, min_samples):
        """Filter dataset to keep annotations for names with the number
        of annotations per name more than the minimum number.
        """
        assert isinstance(
            min_samples, int
        ), 'min_samples should be int type , got type {}'.format(type(min_samples))

        # Analyse unique names
        names = np.array([item[name_idx] for item in data])
        unames, counts = np.unique(names, return_counts=True)
        print('Found {} unique names in {} records'.format(len(unames), len(names)))

        filter_names = unames[counts >= min_samples]
        print('Selecting names with more than {} examples'.format(min_samples))

        new_set = []
        for record in data:
            if record[name_idx] in filter_names:
                new_set.append(record)
        print('Selected {} examples'.format(len(new_set)))

        return new_set

    def _split_train_test_min_samples(self, data, name_idx, train_min_samples=10):
        """Split dataset into train and test subsets by min number of samples.
        If number of bounding boxes per name is less then a threshold,
        then move it to test set.
        Otherwise, it is moved to train set.
        """
        # Analyse unique names:
        names = np.array([item[name_idx] for item in data])
        unames, counts = np.unique(names, return_counts=True)
        print('Found {} unique names in {} records'.format(len(unames), len(names)))

        train_names = unames[counts > train_min_samples]

        print(
            'Selecting names with more than {} examples for training'.format(
                train_min_samples
            )
        )

        train_set = []
        test_set = []
        for record in data:
            if record[name_idx] in train_names:
                train_set.append(record)
            else:
                test_set.append(record)

        print(
            'Selected {} training and {} test examples'.format(
                len(train_set), len(test_set)
            )
        )
        return train_set, test_set

    def db_to_tuples(self, data, relabel=False, name_key='name'):
        """Convert database record to tuples of data for further processing in
        training/testing.

        {'image_path', 'aa_bbox', 'obj_id', 'name', 'viewpoint', 'id_name'}
        => (image_path, person_id)
        """
        # Collect unique names and relabel to integer labels
        if relabel:
            name_contaiter = set()
            for record in data:
                name = record[name_key]
                name_contaiter.add(name)
            name2label = {name: label for label, name in enumerate(name_contaiter)}

        output = []
        for record in data:
            name = record[name_key]
            img_path = record['image_path']
            if relabel:
                name = name2label[name]
            output.append((img_path, name))

        return output
