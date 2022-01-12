from flask import Flask, render_template, request, url_for, redirect, flash
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = 'ritesh'

def load_yolo():
    net = cv2.dnn.readNet("model/yolov3.weights", "model/yolov3.cfg")
    classes = []
    with open("model/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()] 
    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels

def detect_objects(img, net, outputLayers):			
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def count_people(classes, boxes, confs, class_ids): 
    person_counter=0
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            if label == 'person':
                person_counter+=1
    return person_counter

def image_detect(img_path): 
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    peoples = count_people(classes, boxes, confs, class_ids)
    return peoples
    


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/track')
def track():
    count1 = image_detect('./static/images/coach1.png')
    count2 = image_detect('./static/images/coach2.png')

    if count1 < count2:
        coach1 = 'greenyellow'
        coach2 = 'red'
    elif count1 > count2:
        coach1 = 'red'
        coach2 = 'greenyellow'
    elif (count1 == count2) and (count1 <= 10):
        coach1 = 'greenyellow'
        coach2 = 'greenyellow'
    elif (count1 == count2) and (count1 > 10):
        coach1 = 'red'
        coach2 = 'red'
    

    context = {
        'coach1': coach1,
        'coach2': coach2
    }

    return render_template('track.html', **context)


@app.route('/ml', methods=['GET', 'POST'])
def ml():
    if request.method == 'POST':
        pic1 = request.files['pic1']
        pic2 = request.files['pic2']

        output = open('./static/images/coach1.png', "wb")
        output.write(pic1.read())
        output.close()

        output = open('./static/images/coach2.png', "wb")
        output.write(pic2.read())
        output.close()

        return redirect(url_for('ml'))
    return render_template('ml.html')

if __name__ == "__main__":
    app.run(debug=True)