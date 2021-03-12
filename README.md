![](https://user-images.githubusercontent.com/13696193/43165867-fe02e3b2-8f62-11e8-9fd0-cc7c86b11772.png)




# Yolo_Autolabel



### Requirements 📋

Python 3.6 or later.


To install run:

```
pip install -r requirements.txt
```
or simply
```
pip install opencv_python
pip install PySimpleGUI
```

## How to use 

Run:
```
$ python autolabeler.py
```
Now select:
![tutorial_1](/github_images/img1.png)


This will produce this output:
![tutorial_2](/github_images/img2.png)

The images with the detected objects (if 'Save image' is selected) and the labels for every image (if Save label is selected)

Now to export the dataset:

![tutorial_3](/github_images/img3.png)

This will produce this output:

```
Ruta_imagen,N_frame,Nombre_video,Width,Height
./output/video_1_frame_0.png,0,video_1.mp4,1920,1080
./output/video_1_frame_278.png,278,video_1.mp4,1920,1080
./output/video_1_frame_556.png,556,video_1.mp4,1920,1080
./output/video_1_frame_834.png,834,video_1.mp4,1920,1080
```

# Project tree

.
 * train
   * image1.jpg
   * image1.txt
   * ...
 * obj.data
 * obj.names
 * train.txt



## Autores ✒️
* **Oscar Mauriaca** - *Desarrollo* - [xaerincl](https://github.com/xaerincl)
