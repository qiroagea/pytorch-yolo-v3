from __future__ import division
import time
from util import *
from darknet import Darknet
import argparse
import pickle as pkl
from voice import Player


oroka = Player()
oroka.set("voice/orokana2.wav")


def get_test_input(input_dim, cuda):
    image = cv2.imread("imgs/messi.jpg")
    image = cv2.resize(image, (input_dim, input_dim))
    img_ = image[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    # noinspection PyArgumentList
    img_ = Variable(img_)
    if cuda:
        img_ = img_.cuda()
    return img_


def prep_img(image, inpDim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    origIm = image
    dimension = origIm.shape[1], origIm.shape[0]
    image = cv2.resize(origIm, (inpDim, inpDim))
    img_ = image[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, origIm, dimension


def write(x):
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    if label == "person":
        c1 = x[1:3].int()
        c2 = x[3:5].int()
        c1 = c1.numpy()
        c2 = c2.numpy()
        x = c1[0] + c2[0]
        y = c1[1] + c2[1]
        c = ((y-x)**2)**(1/2)
        print(label, c)
        if c < 100:
            oroka.play()


def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument(
        "--reso",
        dest='reso',
        help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
        default="160",
        type=str
    )
    return parser.parse_args()


if __name__ == '__main__':
    cfgfile = "cfg/yolov3-tiny.cfg"
    weightsfile = "yolov3-tiny.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    # noinspection PyRedeclaration
    num_classes = 80
    bbox_attrs = 5 + num_classes

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    videofile = 'video.avi'

    cap = cv2.VideoCapture(0)

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:

            img, orig_im, dim = prep_img(frame, inp_dim)

            if CUDA:
                # noinspection PyUnboundLocalVariable
                im_dim = im_dim.cuda()
                img = img.cuda()

            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

            if type(output) == int:
                frames += 1
                continue

            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim

            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]

            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

            list(map(lambda x: write(x), output))

            frames += 1
        else:
            break
