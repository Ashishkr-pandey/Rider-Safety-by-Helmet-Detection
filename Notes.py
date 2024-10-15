# Dataset is a Collection of Different data in tabular form(or in any other form but i don't know).
# https://github.com/thtrieu/darkflow
# Below link is above link's error
# https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst
# https: // www.codespeedy.com/yolo-object-detection-from -image-with -opencv-and -python/
# Load above link for notes with code

# [model](https://drive.google.com/open?id=16yH9M_ovw0cJG4gVKuXTkz_cwYxJtwAk)
# [cfg](https://drive.google.com/open?id=1GiWyY1EHUWgkBvo8tGuwM4yoplaZZGza)
'''
# YOLO
->It stands for You Only Look Once.
->It is mostly used for real time object detection.
->In it one Classifier is present which see's the dataset & identify the Object. example of dataset is:- PASCAL VOC Dataset
->Basically YOLO first focus on our FPS(frame per second) & in case of 30fps it divide
video frame in 13 row's & 13 coloumn's & then it see 4-5 box together to see the object
if present it show.
->Every object has 1 vector(maybe wrong)
->Vectors can be 13*13 in case of 30 fps.
'''
'''
# Config file(.cfg)
->This file format is of Configuration File.
->It mainly used in how application or computer program is going to be interacting to system.
# Weigth file(.weights)
->Weigth file is the Trained Model that detects the Object.
# Names File(.names)
->Name files contains the Name of the Objects which has to detected.
->In our project, obj is the name file & we are detecting the Helmet that's why i give 'helmet' in obj.names
'''
