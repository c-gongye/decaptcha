#!/usr/bin/env python
import os
import shutil

import joblib
import numpy as np
from scipy import misc

TMPDIR = '.tmp-demo'

if __name__ == '__main__':

    clf = [joblib.load("digit%dmodel.out" % i) for i in range(4)]

    print("model loaded")

    while True:
        raw_input()
        if os.path.exists(TMPDIR):
            shutil.rmtree(TMPDIR)
        os.system("python ./generate.py 1 -l 4 -p binary_only -s %s >/dev/null" % TMPDIR)
        img = os.listdir(TMPDIR)[0]
        if os.system("./imgcat %s 2>/dev/null" % os.path.join(TMPDIR, img)) > 0:
            print("generated captcha saved at %s" % os.path.join(TMPDIR, img))
            import matplotlib.pyplot as plt
            f = misc.imread(os.path.join(TMPDIR, img))
            plt.imshow(f)
            plt.show()
        # load processed data
        prediction = ''

        X = np.load('X.npy')
        for i in range(4):
            prediction += str(clf[i].predict(X)[0])

        print("Prediction result is: %s" % prediction)
