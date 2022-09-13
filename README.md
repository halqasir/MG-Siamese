# MG-Siamese

To insert shape priors in our model, we resort to a Siamese architecture that learns a mapping that projects the images and the masks into a feature embedding space, each mask corresponding to the specific class to be tested. Euclidean distances are evaluated in the feature space at test time to decide to which class the input image belongs.

*Requirements*
- tensorflow
- matplotlib
- numpy
- scipy
- cv2
- pickle
- h5py
- skimage
- easydict

*Train the model*
``` 
python train.py --cfg $cfg_file
``` 


*Test the model*
```  
python test.py --cfg $cfg_file
``` 
