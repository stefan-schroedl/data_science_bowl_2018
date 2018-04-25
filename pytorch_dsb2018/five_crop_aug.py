#!/usr/bin/env python

import imgaug as ia


class FiveCrop(ia.augmenters.Augmenter):

    """Subclass of img_aug.Augmenter to randomly crop one of the corners or the center"""

    def __init__(self, size, name=None, deterministic=False, random_state=None):
        super(FiveCrop, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        if ia.is_single_integer(size):
            self.size = (size, size)
        elif isinstance(size, (list, tuple)) and len(size) == 2 and isinstance(size[0], (int, long)) and isinstance(size[1], (int, long)):
            self.size = size
        else:
            raise ValueError('Invalid size parameter: %s' % str(size))

        self.choice = ia.parameters.DiscreteUniform(0, 4)


    def get_parameters(self):
        return [self.size, self.interpolation]


    def center_crop(self, img, output_size):

        h, w = img.shape[0:2]

        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return img[i:(i+th), j:(j+tw)]


    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in xrange(nb_images):
            seed = seeds[i]
            img = images[i]
            h, w = img.shape[0:2]

            crop_h, crop_w = self.size

            if crop_w > w or crop_h > h:
                raise ValueError("Requested crop size {} is bigger than input size {}".format(self.size,
                                                                                              (h, w)))
            random_state = ia.new_random_state(seed)
            which = self.choice.draw_sample(random_state=random_state)
            if which == 0:
                # top left
                image_cr = img[0:crop_h, 0:crop_w]
            elif which == 1:
                # top right
                image_cr = img[0:crop_h, (w-crop_w):w]
            elif which == 2:
                # bottom left
                image_cr = img[(h-crop_h):h, 0:crop_w]
            elif which == 3:
                # bottom right
                image_cr = img[(h-crop_h):h, (w - crop_w):w]
            else:
                # center
                image_cr = self.center_crop(img, (crop_h, crop_w))

            result.append(image_cr)

        return result


    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            seed = seeds[i]
            h, w = keypoints_on_image.shape[0:2]
            crop_h, crop_w = self.size[0:2]

            random_state = ia.new_random_state(seed)
            which = self.choice.draw_sample(random_state=random_state)
            if which == 0:
                shifted = keypoints_on_image
            elif which == 1:
                shifted = keypoints_on_image.shift(x=-crop_w)
            elif which == 2:
                shifted = keypoints_on_image.shift(y=-crop_h)
            elif which == 3:
                shifted = keypoints_on_image.shift(x=-crop_w, y=-crop_h)
            else:
                i = int(round((h - crop_h) / 2.))
                j = int(round((w - crop_w) / 2.))
                shifted = keypoints_on_image.shift(x=-i, y=-j)

            shifted.shape = (crop_h, crop_w)

            result.append(shifted)

        return result
