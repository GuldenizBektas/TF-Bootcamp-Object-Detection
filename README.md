# TF-Bootcamp-Object-Detection

![image](https://github.com/GuldenizBektas/TF-Bootcamp-Object-Detection/blob/main/Object_Detection.png)

This notebook contains Object Detection model with Tensorflow Object Detection API. It's created for TF Bootcamp.

I'll write down roughly the steps you'll go through.

This is a custom object detection model (Mask Detection). This means I created the image dataset. This description is in the notebook either.

We're going to make a mask detection model with Tensorflow Object Detection API, using SSD MobNet model in TF Hub.
First, to train and test your detector you need to create your own dataset. You can do this by simply using a code block to take your images with and without mask on and the longest and the hardest part, labelling them. You'll use labelimg library for this.
Let's break it down to two steps to create your own dataset.
1. Take your images.
  We're going to make a mask detection so while you take your images you should put your mask on at first, and them take    it out to have dataset without and with mask on. You can do sign language prediction or other object's detection but according to that labels may change. We'll get into that later.
  
   ⭐️ uuid library is used to give an unique id to images.
   
```
import cv2 as cv
import uuid
cap = cv.VideoCapture(0)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    imgname="./Images/Mask/{}.jpg".format(str(uuid.uuid1()))
    cv.imwrite(imgname, frame)
    cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
   
```

2. Label images with labelimg library. Go to your terminal or command prompt and type labelimg. The interface of library will open.

![image](https://github.com/GuldenizBektas/TF-Bootcamp-Object-Detection/blob/main/labelimg.png)

Something like this. Then at the left top click Open Dir and choose the image dictionary. Program will create XML files of every image so we need to choose save directory for this too. Click Change Save Dir and choose the same directory since we want them in the same place. Now we're already there.
To define every image choose Create RectBox botton at the left bar. Now you can define every object manually. After that it will ask you the label. In our case we'll write "mask" and "nomask" to the particular images.
🤯 A tip: Cmd+W allow you to draw a label. A and D allow you to go next or previous image.
And now you can split the first X images and their corresponding XML rotations as train set and the rest of them will be test set.

-------

The files and the directories will be created with the code within the notebook if it's not already exists. Here's the diagram of the file system:

```
tensorflow -
          |_ models
          |_protoc
          |_scripts
          |_workspace -
                       |_annotations -
                                      |_label_map.pbtxt
                       |_images -
                                 |_train
                                 |_test
                       |_models -
                                 |_my_ssd_mobnet
                       |_pre_trained_model -
                                            |_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8
```

The notes taken from the Michigan Universities Object Detection lecture. Check that out: https://www.youtube.com/watch?v=TB-fdISzpHQ&t=3744s

And watch the videos:
- https://www.youtube.com/watch?v=IOI0o3Cxv9Q&t=1335s
- https://www.youtube.com/watch?v=T9KfYaS9hwQ&t=414s
