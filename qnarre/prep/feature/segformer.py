import numpy as np
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ImageFeatureExtractionMixin,
    ImageInput,
    is_torch_tensor,
)
from ...utils import logging


logger = logging.get_logger(__name__)


class SegformerFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize=True,
        size=512,
        resample=Image.BILINEAR,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        reduce_labels=False,
        **kw,
    ):
        super().__init__(**kw)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.reduce_labels = reduce_labels

    def __call__(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput = None,
        return_tensors=None,
        **kw,
    ):
        valid_images = False
        valid_segmentation_maps = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if (
                len(images) == 0
                or isinstance(images[0], (Image.Image, np.ndarray))
                or is_torch_tensor(images[0])
            ):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example),"
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        # Check that segmentation maps has a valid type
        if segmentation_maps is not None:
            if isinstance(segmentation_maps, (Image.Image, np.ndarray)) or is_torch_tensor(
                segmentation_maps
            ):
                valid_segmentation_maps = True
            elif isinstance(segmentation_maps, (list, tuple)):
                if (
                    len(segmentation_maps) == 0
                    or isinstance(segmentation_maps[0], (Image.Image, np.ndarray))
                    or is_torch_tensor(segmentation_maps[0])
                ):
                    valid_segmentation_maps = True

            if not valid_segmentation_maps:
                raise ValueError(
                    "Segmentation maps must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example),"
                    "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
                )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]
            if segmentation_maps is not None:
                segmentation_maps = [segmentation_maps]

        # reduce zero label if needed
        if self.reduce_labels:
            if segmentation_maps is not None:
                for idx, map in enumerate(segmentation_maps):
                    if not isinstance(map, np.ndarray):
                        map = np.array(map)
                    # avoid using underflow conversion
                    map[map == 0] = 255
                    map = map - 1
                    map[map == 254] = 255
                    segmentation_maps[idx] = Image.fromarray(map.astype(np.uint8))

        # transformations (resizing + normalization)
        if self.do_resize and self.size is not None:
            images = [
                self.resize(image=image, size=self.size, resample=self.resample) for image in images
            ]
            if segmentation_maps is not None:
                segmentation_maps = [
                    self.resize(map, size=self.size, resample=Image.NEAREST)
                    for map in segmentation_maps
                ]

        if self.do_normalize:
            images = [
                self.normalize(image=image, mean=self.image_mean, std=self.image_std)
                for image in images
            ]

        # return as BatchFeature
        data = {"pixel_values": images}

        if segmentation_maps is not None:
            labels = []
            for map in segmentation_maps:
                if not isinstance(map, np.ndarray):
                    map = np.array(map)
                labels.append(map.astype(np.int64))
            # cast to np.int64
            data["labels"] = labels

        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
