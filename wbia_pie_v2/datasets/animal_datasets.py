# -*- coding: utf-8 -*-
from .coco_dataset import COCODataset
import yaml
import os


from datetime import date

# used on some data labels
CURRENT_YEAR = date.today().year

# This dynamically creats a COCODataset object from looking at a config file
class ConfigDataset(COCODataset):

    def __init__(self, config_fpath, year=CURRENT_YEAR, **kwargs):
        conf = _read_and_validate_dataset_config(config_fpath)
        # name is the config filename without extension or containing folders
        id_attr = ['name']
        if conf['data'].get('use_viewpoint', True):
            id_attr.append('viewpoint')
        mode = kwargs.get('mode', 'train')
        super(ConfigDataset, self).__init__(
            name=os.path.splitext(os.path.split(config_fpath)[1])[0],
            dataset_url=conf['data'].get('dataset_url', None),
            dataset_dir=conf['data']['coco_dir'],
            # a year label is common on coco datasets
            split=conf['data'].get('split', f'{mode}{year}'),
            crop=conf['data'].get('crop', True),
            resize=conf['data'].get('resize', True),
            imsize=min(conf['data']['height'], conf['data']['width']),
            train_min_samples=conf['data'].get('train_min_samples', 3),
            test_min_samples=conf['data'].get('test_min_samples', 3),
            viewpoint_list=conf['data'].get('viewpoint_list', ['left', 'right']),
            debug=conf['data'].get('debug', False),
            **kwargs
        )


def _read_and_validate_dataset_config(config_fpath):
    with open(config_fpath, "r") as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    assert 'data' in conf and 'coco_dir' in conf['data'], 'Error: ConfigDataset needs data.coco_dir to be defined in config file.'
    return conf


class WhaleShark(COCODataset):
    def __init__(self, **kwargs):
        super(WhaleShark, self).__init__(
            name='whaleshark',
            dataset_dir='whaleshark',
            dataset_url='https://lilablobssc.blob.core.windows.net/whale-shark-id/whaleshark.coco.tar.gz',
            split='train2020',
            crop=True,
            resize=True,
            imsize=256,
            train_min_samples=5,
            id_attr=['name', 'viewpoint'],
            viewpoint_list=['left', 'right'],
            debug=False,
            **kwargs
        )


class WhaleSharkCropped(COCODataset):
    def __init__(self, **kwargs):
        super(WhaleSharkCropped, self).__init__(
            name='whaleshark_cropped',
            dataset_dir='whaleshark_cropped',
            dataset_url='https://www.dropbox.com/s/4tky8z0g4ob6qfx/coco.whaleshark_cropped.tar.gz?dl=1',
            split='test2021',
            viewpoint_csv='reid-data/whaleshark_cropped/original/whaleshark_cropped.coco/annotations/annotViewpointMap.csv',
            crop=False,
            resize=True,
            imsize=256,
            train_min_samples=5,
            id_attr=['name', 'viewpoint'],
            viewpoint_list=['left', 'right'],
            debug=False,
            **kwargs
        )


class MantaRayCropped(COCODataset):
    def __init__(self, **kwargs):
        super(MantaRayCropped, self).__init__(
            name='mantaray_cropped',
            dataset_dir='mantaray_cropped.coco',
            dataset_url=None,
            split='train2021',
            crop=False,
            resize=False,
            imsize=300,
            train_min_samples=5,
            id_attr=['name'],
            debug=False,
            **kwargs
        )


class GrayWhale(COCODataset):
    def __init__(self, **kwargs):
        super(GrayWhale, self).__init__(
            name='graywhale',
            dataset_dir='graywhale',
            dataset_url='',
            split='train_right',
            split_test='test_left',
            crop=True,
            flip_test=True,
            resize=True,
            imsize=256,
            train_min_samples=3,
            test_min_samples=3,
            id_attr=['name', 'viewpoint'],
            viewpoint_list=['left', 'right'],
            debug=False,
            excluded_names='____',
            **kwargs
        )


class HyenaBothsides(COCODataset):
    def __init__(self, **kwargs):
        super(HyenaBothsides, self).__init__(
            name='hyena_bothsides',
            dataset_dir='/data/db/_ibsdb/_ibeis_cache/hyena_bothsides',
            dataset_url=None,
            split='train2021',
            split_test='test2021',
            crop=True,
            resize=True,
            imsize=256,
            train_min_samples=3,
            test_min_samples=3,
            id_attr=['name', 'viewpoint'],
            debug=False,
            excluded_names='____',
            **kwargs
        )


class WildHorseFace(COCODataset):
    def __init__(self, **kwargs):
        super(WildHorseFace, self).__init__(
            name='wildhorse_face',
            dataset_dir='wildhorses_combined',
            dataset_url='',
            split='train2021',
            crop=True,
            flip_test=False,
            resize=True,
            imsize=300,
            train_min_samples=3,
            id_attr=['name', 'viewpoint'],
            viewpoint_list=['front'],
            debug=False,
            excluded_names='____',
            **kwargs
        )


