import time
import cv2
import numpy as np
import service.main as s
def emotions_detector(img_array):

    if len (img_array.shape)==2:
        img_array=cv2.cvtColor (img_array, cv2.COLOR_GRAY2RGB)

    time_init=time.time()

    output_names=['dense_1']

    # test_image = cv2.imread(im_path) we will be getting direct image
    test_image = cv2.resize(img_array,(224,224))
    im = np.float32(test_image)
    img_array = np.expand_dims(im, axis = 0)

    time_processing=time.time()-time_init

    pred=s.m.run(output_names, {"input": img_array})
    time_elapsed=time.time()-time_init

    classes={0:'No_DR', 1:'Mild', 2:'Moderate', 3:'Severe', 4:'Proliferat_DR'}
    result=np.argmax(pred[0][0])
    return {'emotion': classes[result],
            'time_elapsed': time_elapsed,
            'time_processing': time_processing}