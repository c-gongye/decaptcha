#!/usr/bin/env python
import joblib
import numpy as np
import os
import shutil

TMPDIR = '.tmp-demo'

if __name__ == '__main__':

    if os.path.exists(TMPDIR):
        shutil.rmtree(TMPDIR)
    os.system("python ./generate.py 1 -l 4 -p binary_only -s %s" % TMPDIR)
    img = os.listdir(TMPDIR)[0]
    if os.system("imgcat %s 2>/dev/null" % os.path.join(TMPDIR, img)) > 0:
        print("generated captcha saved at %s" % os.path.join(TMPDIR, img))
        from PIL import ImageShow
        v = ImageShow._viewers
        if v:
            v[0].show_file(os.path.join(TMPDIR, img))
        
    # load processed data
    X = np.load('X.npy')
    prediction = ''

    for i in range(4):
        tmp_clf = joblib.load("digit%dmodel.out"%(i))
        prediction += str(tmp_clf.predict(X)[0])

    print "Prediction result is: ",prediction

