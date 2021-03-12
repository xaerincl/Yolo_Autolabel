import PySimpleGUI as sg
import numpy as np
import io
import cv2
import shutil
import os



# --------------------------------- Define Layout ---------------------------------

layout =    [

        [sg.Text('Step 1: Save labels .txt and/or save detected images')],
        [sg.Text('Path to .weight '), sg.In('yolov3.weights',size=(40,1), key='-weight-'), sg.FileBrowse()],
        [sg.Text('Path to .cfg    '), sg.In('yolov3.cfg',size=(40,1), key='-cfg-'), sg.FileBrowse()],
        [sg.Text('Path to .names  '), sg.In('coco.names',size=(40,1), key='-names-'), sg.FileBrowse()],
        [sg.Text('Image folder    '), sg.In('imagenes',size=(40,1), key='-img_folder-'), sg.FolderBrowse()],
        [sg.Text('Save labels     '), sg.In('labels',size=(40,1), key='-label_folder-'), sg.FolderBrowse()],
        [sg.Text('Save infer imgs '), sg.In('detections',size=(40,1), key='-img_infer-'), sg.FolderBrowse()],
        [sg.Text('Confidence'), sg.Slider(range=(0,1),orientation='h', resolution=.1, default_value=.3, size=(15,15), key='-confidence-')],
        [sg.Text('Threshold'), sg.Slider(range=(0,1), orientation='h', resolution=.1, default_value=.5, size=(15,15), key='-threshold-')],
        [sg.CB(text = 'GPU', default = False, enable_events = True, key = '-GPU-'), 
            sg.CB('Save label', default = True, enable_events = True, key = '-LABEL-'), 
            sg.CB('Save image', default = False, enable_events = True, key = '-img_save-')],
        [sg.Button('Run')],
        [sg.Text('_'*60)],
        [sg.Txt('Step 2: Export dataset in YOLO format ready to train')],
        [sg.Text('Export dataset: '), sg.In('export',size=(40,1), key='-export-'), sg.FolderBrowse()],
        [sg.Button('Export')]
            ]


# --------------------------------- Create Window ---------------------------------
window = sg.Window('Yolo Automatic Labeler', layout,resizable=True, return_keyboard_events=True, use_default_focus=False, finalize = True)

# --------------------------------- Event Loop ---------------------------------
def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            #shutil.copy2(s, d)
            shutil.move(s,d)

def create_raw_labels():



    with open(values['-names-']) as file_object:
        content = file_object.readlines()

    with open('labels_to_cvat.txt', "w") as file_object:
        for i,c in enumerate(content):
            if i == 0:
                file_object.write('[' + '\n')

            file_object.write('  {' + '\n')
            file_object.write('    "name": ' + '"' + c.rstrip() + '",'  + '\n')
            file_object.write('    "id": ' + str(i+1) + ',' + '\n')
            file_object.write('    "color": '+  '"' + '#3c6faa' + '",' + '\n')
            file_object.write('    "attributes": []' + '\n')
            if i == len(content)-1:
                file_object.write('  }' + '\n')
                file_object.write(']')
            else:
                file_object.write('  },' + '\n')



def exportar_dataset(fnames):

    #create folder
    if not os.path.exists(values['-export-']):
        os.makedirs(values['-export-'])

    if not os.path.exists(os.path.join(values['-export-'], 'train')):
        os.makedirs(os.path.join(values['-export-'], 'train'))

    # create obj.data
    with open(values['-cfg-'], 'r') as searchfile:
        for line in searchfile:
            if 'classes' in line:
                break
    n_classes = line.split('=')[1].rstrip()


    name = 'obj.data'
    classes = 'classes = '+ str(n_classes)
    train = 'train = data/train.txt'
    names = 'names = data/obj.names'
    backup = 'backup = backup/'
    with open(os.path.join(values['-export-'], name), "w") as file_object:
        file_object.write(classes + "\n" + train + "\n" + names + "\n" + backup)

    # create obj.name

    orig = values['-names-']
    dest = os.path.join(values['-export-'], 'obj.names')
    shutil.copy(orig,dest)


    # create train.txt

    prefix = 'data/train/'
    with open(os.path.join(values['-export-'], 'train.txt'), "w") as file_object:
        for i in fnames:
            file_object.write(prefix + i + '\n')

    # move objects

    copytree(values['-img_folder-'], os.path.join(values['-export-'], 'train'))
    try:
        copytree(values['-label_folder-'], os.path.join(values['-export-'], 'train'))
    except:
        pass

    print('Done')



    #


def yolo_get_label(id_class, img, detect):

    name = img[:-4]+'.txt'
    x_r = round(float(detect[0]),6)
    y_r = round(float(detect[1]),6)
    w_r = round(float(detect[2]),6)
    h_r = round(float(detect[3]),6)

    with open(os.path.join(values['-label_folder-'], name), "a+") as file_object:
        file_object.seek(0)
        data = file_object.read(100)
        if len(data) > 0 :
            file_object.write("\n")
        file_object.write(f'{id_class} {x_r} {y_r} {w_r} {h_r}')


def yolo_infer(img_list):
    if not os.path.exists(values['-label_folder-']):
        os.makedirs(values['-label_folder-'])

    if values['-img_save-']:
        if not os.path.exists(values['-img_infer-']):
            os.makedirs(values['-img_infer-'])

    get_label = True
    image_show = True
    try:
        net = cv2.dnn.readNetFromDarknet(values['-cfg-'], values['-weight-'])
    except Exception as E:
        print(f'** Error {E} **')


    if values['-GPU-']:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except:
            pass

    try:
        LABELS = open(values['-names-']).read().strip().split('\n')
    except Exception as E:
        print(f'** Error {E} **')

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    for img in img_list:
        with open(os.path.join(values['-label_folder-'], img[:-4]+'.txt'), "w") as file_object:
            # creates the label .txt file
            pass

        image = cv2.imread(os.path.join(values['-img_folder-'], img))
        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        boxes = []
        confidences = []
        classIDs = []
        detections = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > values['-confidence-']:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    detections.append(detection[0:4])
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, values['-confidence-'], values['-threshold-'])
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                if values['-LABEL-']:
                    yolo_get_label(classIDs[i], img, detections[i])

                if values['-img_save-']:
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
                    cv2.imwrite(os.path.join(values['-img_infer-'], img),image)

    create_raw_labels()
    print('Done')
    pass



while True:

    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Cancel'):
        break

    if event == 'Run':
        folder = values['-img_folder-']
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []
        fnames = [f for f in file_list if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(('.png', '.PNG', '.jpg'))]
        fnames = sorted(fnames)
        yolo_infer(fnames)

    if event == 'Export':
        print('Export dataset')
        exportar_dataset(fnames)

# --------------------------------- Close & Exit ---------------------------------
window.close()
print('Exiting')