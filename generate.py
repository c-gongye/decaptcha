#!/usr/bin/env python
import os
import sys
import random
import time
import argparse
import numpy as np
from captcha.image import ImageCaptcha

class PreProcessor(object):
    
    @staticmethod
    def L_only(length):
        @next
        def _(s, img):
            img = img.convert("L")
            # TODO: fallback to 40*60 for single char?
            return list(s), [np.array(img.crop((i * 27, 0, i * 27 + 27, 60))).flatten() for i in range(length)]
        return _

    @staticmethod
    def binary_only(length):
        from scipy import ndimage
        @next
        def _(s, img):
            img = img.convert("L")
            r = [np.array(img.crop((i * 27, 0, i * 27 + 27, 60))) for i in range(length)]
            r = map(lambda im: ndimage.binary_opening((im > im.mean()).astype(np.float)).flatten(), r)
            # TODO: fallback to 40*60 for single char?
            return list(s), r
        return _

def generate(length, img_captcha):
    @next
    def _(seq):
        s = ("%%0%dd" % length) % (random.random() * (10 ** length))
        return s, img_captcha.generate_image(s)
    return _

def save_image(path):
    @next
    def _(s, img):
        fname = os.path.join(path, "%s-%s.png" % (s, 10000000 * random.random()))
        img.save(fname)
        return s, img
    return _

def next(f):
    def exe(*args):
        r = __exe.next(*f(*args))
        return r
    __exe = exe
    return exe

def process(count, chain, save_X, save_y):
    X = []
    y = []
    for i in range(count):
        _ = chain(i)
        for _X in _[1]:
            X.append(_X)
        for _y in _[0]:
            y.append(int(_y))
    with open(save_X, "wb") as f:
        np.save(f, np.array(X))
    with open(save_y, "wb") as f:
        np.save(f, np.array(y))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate captcha and save to ndarray')
    parser.add_argument('N', default=1000, help="number of captcha to generate")
    parser.add_argument('-l', '--length', default=4, help="length of digits per captcha")
    parser.add_argument('-s', '--save', metavar="PATH", default=None,
        help="path to save generated captcha, by default, images are not saved")
    parser.add_argument('--X', default="X.npy", help="path to save the ndarray of the data points")
    parser.add_argument('--y', default="y.npy", help="path to save the ndarray of the labels")

    parser.add_argument('-p', '--process', default="L_only",
            choices=[f for f in PreProcessor.__dict__.keys() if not f.startswith("_")],
            help="algorithms to process images")

    args = parser.parse_args()

    if args.save:
        if os.path.exists(args.save) and os.path.isdir(args.save) and os.listdir(args.save):
            print("ERROR: save path \"%s\" exists and is not empty" % args.save)
            sys.exit(0)
        elif not os.path.exists(args.save):
            os.makedirs(args.save)
    
    length = int(args.length)
    img_captcha = ImageCaptcha(width=40 * length, height=60, font_sizes = (45, ))

    # setup call chains
    chain = []
    chain.append(generate(length, img_captcha))
    if args.save:
        chain.append(save_image(args.save))
    chain.append(getattr(PreProcessor, args.process)(length))
    chain.append(lambda x, y:(x, y))

    for i in range(len(chain) - 1):
        chain[i].next = chain[i + 1]
    
    # start process
    s = time.time()
    process(int(args.N), chain[0], args.X, args.y)
    print("generated %s samples in %.2fs" % (args.N, time.time() - s))

    
