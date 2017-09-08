import os
import chainer
from chainer import config
import cv2
import numpy as np
np.random.seed(555)
from contextlib import ExitStack


class DatasetPreProcessor(chainer.dataset.DatasetMixin):
    def __init__(self):
        self.pairs = self.read_paths()
        self.counter = 0
        self.image_size_in_batch = [None, None]  # height, width

    def __len__(self):
        return len(self.pairs)

    def read_paths(self):
        path_label_pairs = []
        for image_path, label in self.path_label_pair_generator():
            path_label_pairs.append((image_path, label))
        return path_label_pairs

    def path_label_pair_generator(self):
        with ExitStack () as stack:
            f_image = stack.enter_context(open(config.image_pointer_path, 'r'))
            f_label = stack.enter_context(open(config.labels_file_path, 'r'))

            for image_file_name, label in zip(f_image, f_label):
                image_file_name = image_file_name.rstrip()
                image_full_path = os.path.join(config.image_dir_path, image_file_name)
                if os.path.isfile(image_full_path):
                    yield image_full_path, label.rstrip()
                else:
                    raise RuntimeError("file is not fined: {}.".format(image_full_path))

    def __init_batch_counter(self):
        if config.train and self.counter==config.training_params.batch_size:
            self.counter = 0
            self.image_size_in_batch = [None, None]

    def __set_image_size_in_batch(self, image):
        if self.counter==1:
            resized_h, resized_w = image.shape[:2]
            self.image_size_in_batch = [resized_h, resized_w]

    def get_example(self, index):
        self.counter += 1
        if config.debug_mode:
            if self.counter>15:
                assert False, 'stop test'

        path, label = self.pairs[index]
        image = cv2.imread(path)
        src_image = image.copy()

        if config.debug_mode:
            show_image(image)
            # cv2.imwrite("/Users/naoki_shimada/Downloads/Origin.jpg", image)

        # gray transform if converse_gray is True
        image = self.color_trancefer(image)
        h, w, ch = image.shape

        if image is None:
            raise RuntimeError("invalid image: {}".format(path))

        # resizing image
        if config.do_resize:
            if self.counter>1:
                # augmentas is ordered w,h in resize method of openCV
                scale = self.image_size_in_batch[1]/w, self.image_size_in_batch[0]/h
                image = self.resize_image(image, scale)
            else:
                image = self.resize_image(image)
        elif config.crop_params.flag:
            image = self.crop_image(image)

        # augmentat image
        if config.aug_params.do_augment:
            image = self.augment_image(image)

        if config.debug_mode:
            u_image = image.astype(np.uint8)
            show_image(u_image)
            cv2.imwrite(config.data_root_path+"/data/{}".format(os.path.basename(path)), u_image)
            print(image.shape)
            print('label:', label)
            # assert False, 'terminate'

        # store image size
        # because dimension must be equeal per batch
        self.__set_image_size_in_batch(image)

        # image normalize
        image = getattr(self.image_normalizer, \
            config.im_norm_type.method)(image, config.im_norm_type.opts)

        if config.debug_mode:
            u_image = image.astype(np.uint8)
            show_image(u_image)
            # cv2.imwrite("/Users/naoki_shimada/Downloads/GCN.jpg", show_image)
            print(image.shape)
            print('label:', label)
            # assert False, 'terminate'

        if config.detect_edge:
            edges = self.detect_edge(image)
            if config.debug_mode:
                u_image = edges.reshape(edges.shape[0], edges.shape[1]).astype(np.uint8)
                show_image(u_image)

            image = cv2.merge((image, edges))

        # transpose for chainer
        image = image.transpose(2, 0, 1)
        # initialize batch counter
        self.__init_batch_counter()

        batch_inputs = image.astype(np.float32), np.array(label, dtype=np.int32)

        if config.with_feature_val:
            cell_diameter = \
                compute_cell_diameter(src_image, debug=config.debug_mode)
            min_nucleus_diameter, max_nucleus_diameter = \
                compute_nucleus_diameter(src_image, debug=config.debug_mode)
            features = np.array( \
                [cell_diameter, min_nucleus_diameter, max_nucleus_diameter],
                dtype=np.float32)
            batch_inputs += (features, )

        return batch_inputs

    def color_trancefer(self, image):
        h, w, _ = image.shape
        if config.converse_gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape((h,w,1))
        else:
            image = image.astype(np.float32)
        return image

    def augment_image(self, image):
        if confg.aug_params.params.do_scale and self.counter==1:
            image = self.scaling(image)

        if config.aug_params.params.do_flip:
            image = self.flip(image)

        if config.aug_params.params.change_britghtness:
            image = self.random_brightness(image)

        if config.aug_params.params.change_contrast:
            image = self.random_contrast(image)

        if config.aug_params.params.do_rotate:
            image = self.rotate_image(image)

        if config.aug_params.params.do_shift:
            image = self.shift_image(image)

        if config.aug_params.params.do_blur:
            image = self.random_blur(image)

        return image

    def resize_image(self, image, scale=None):
        xh, xw = image.shape[:2]

        if scale is None:
            # if scale is not difinded, calculate scale as closest multiple number.
            h_scale = (xh//config.multiple)*config.multiple/xh
            w_scale = (xw//config.multiple)*config.multiple/xw
            scale = w_scale, h_scale
        elif isinstance(scale, numbers.Number):
            scale = scale, scale
        elif isinstance(scale, tuple) and len(scale)>2:
            raise RuntimeError("Error occurred with variable ot scale in resize_image method")

        new_sz = (int(xw*scale[0])+1, int(xh*scale[1])+1)  # specification of opencv, argments is recepted (w, h)
        image = cv2.resize(image, new_sz)

        xh, xw = image.shape[:2]
        m0, m1 = xh % config.multiple, xw % config.multiple
        d0, d1 = np.random.randint(m0+1), np.random.randint(m1+1)
        image = image[d0:(image.shape[0] - m0 + d0), d1:(image.shape[1] - m1 + d1)]

        if len(image.shape)==2:
            return image.reshape((image.shape[0], image.shape[1], 1))
        else:
            return image

    def flip(self, image):
        do_flip_xy = np.random.randint(0, 2)
        do_flip_x = np.random.randint(0, 2)
        do_flip_y = np.random.randint(0, 2)

        if do_flip_xy: # Transpose X and Y axis
            image = image[::-1, ::-1, :]
        elif do_flip_x: # Flip along Y-axis
            image = image[::-1, :, :]
        elif do_flip_y: # Flip along X-axis
            image = image[:, ::-1, :]
        return image

    def scaling(self, image):
        do_scale = np.random.randint(0, 2)
        if do_scale:
            scale = config.aug_params.params.scale[ \
                np.random.randint(0,len(config.aug_params.params.scale))]
            return self.resize_image(image, scale)
        else:
            return image

    def random_brightness(self, image, lower=0.2, upper=2.0, seed=None):
        brightness_flag = np.random.randint(0, 2)
        if brightness_flag:
            h, w, ch = image.shape
            gamma = np.random.uniform(lower, upper)
            image = 255 * np.power(image, 1.0/gamma)*255**(-1.0/gamma)
            return self.__reshpe_channel(image, (h,w,ch))
        else:
            return image

    def random_contrast(self, image, lower=1.0, upper=20.0, seed=None):
        def __change(one_channel):
            f = np.random.uniform(lower, upper)
            return 255.0/(1+np.exp(-f*(one_channel-128)/255))

            # mean = one_channel.mean()
            # max_val = one_channel.max()
            # return max_val/(1+np.exp(-f*(one_channel-mean)/max_val))
            # return (one_channel - mean) * f + mean

        contrast_flag = np.random.randint(0, 2)
        if contrast_flag:
            h, w, ch = image.shape
            image = cv2.merge(tuple(__change(d_ch) for d_ch in cv2.split(image)))
            return self.__reshpe_channel(image, (h,w,ch))
        else:
            return image

    def shift_image(self, image):
        do_shift_xy = np.random.randint(0, 2)
        do_shift_x = np.random.randint(0, 2)
        do_shift_y = np.random.randint(0, 2)

        if do_shift_xy:
            lr_shift = config.aug_params.params.lr_shift[ \
                np.random.randint(0,len(config.aug_params.params.lr_shift))]
            ud_shift = config.aug_params.params.ud_shift[ \
                np.random.randint(0,len(config.aug_params.params.ud_shift))]
        elif do_shift_y:
            lr_shift = 0
            ud_shift = config.aug_params.params.ud_shift[ \
                np.random.randint(0,len(config.aug_params.params.ud_shift))]
        elif do_shift_x:
            lr_shift = config.aug_params.params.lr_shift[ \
                np.random.randint(0,len(config.aug_params.params.lr_shift))]
            ud_shift = 0

        if do_shift_xy or do_shift_y or do_shift_y:
            h, w, ch = image.shape
            affine_matrix = np.float32([[1,0,lr_shift],[0,1,ud_shift]])  # 横、縦
            image = cv2.warpAffine(image, affine_matrix, (w,h))
            return self.__reshpe_channel(image, (h,w,ch))
        else:
            return image

    def rotate_image(self, image):
        do_rotate = np.random.randint(0, 2)
        if do_rotate:
            h, w, ch = image.shape
            rotation_angle = config.aug_params.params.rotation_angle[ \
                np.random.randint(0,len(config.aug_params.params.rotation_angle))]
            affine_matrix = cv2.getRotationMatrix2D((h/2, w/2), rotation_angle, 1)

            image = cv2.warpAffine(image, affine_matrix, (w,h))
            return self.__reshpe_channel(image, (h,w,ch))
        else:
            return image

    def random_blur(self, image, average_square=None):
        do_blur = np.random.randint(0, 2)
        if do_blur:
            h, w, ch = image.shape
            # the larger the number, the blurred.
            # original: (25, 25)
            average_square = (10, 10) if average_square is None else average_square
            # calculate moving average and output
            image = cv2.blur(image, average_square)
            return self.__reshpe_channel(image, (h,w,ch))
        else:
            return image

    def crop_image(self, image):
        h, w, ch = image.shape
        top = int((w-config.crop_params.size)/2)
        left = int((h-config.crop_params.size)/2)
        return image[left:left+config.crop_params.size,top:top+config.crop_params.size,:]

    def __reshpe_channel(self, image, im_shape ):
        if len(image.shape)==2:
            return image.reshape(im_shape)
        else:
            return image
