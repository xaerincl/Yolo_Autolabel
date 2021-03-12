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

![tutorial_4](/github_images/img4.png)


```
export
|
+--obj_train_data
|  |
|  +--soccer.txt
|  |
|  +--dining_table.txt
|
|  +--... (rest of the labels)
|
+--obj.data
| 
+--obj.names
| 
+--train.txt
| 
+--images.zip
| 
+--upload.zip
| 
+--labels_to_cvat.txt
```



images and upload.zip files are ready to be uploaded to CVAT if you need to edit the labels or export into another format.


## How to upload labels to CVAT


Create project
![tutorial_5](/github_images/img5.png)
copy .txt into raw
done
submit
create task
![tutorial_5](/github_images/img6.png)
upload images.zip
upload annotations yolo 1.1
upload upload.zip
![tutorial_5](/github_images/img7.png)

ready to go!




## Autores ✒️
* **Oscar Mauriaca** - *Desarrollo* - [xaerincl](https://github.com/xaerincl)
