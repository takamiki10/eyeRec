"""Image normalization utilities."""

import numpy as np


class ImageNormalizer:
    """Basic placeholder normalizer."""

    def normalize(self, image):
        if image is None:
            return None
        return np.asarray(image)
