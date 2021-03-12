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
```



## Autores ✒️
* **Oscar Mauriaca** - *Desarrollo* - [xaerincl](https://github.com/xaerincl)
