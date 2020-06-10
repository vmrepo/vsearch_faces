
import cv2 as cv
import numpy as np
import os
from time import time
from pytube import YouTube

#CASCADE = cv.CascadeClassifier('./face_detectors/haarcascade_frontalface_default.xml')
NET = cv.dnn.readNetFromCaffe('./face_detectors/deploy.prototxt', './face_detectors/res10_300x300_ssd_iter_140000_fp16.caffemodel')
#NET = cv.dnn.readNetFromTensorflow('./face_detectors/opencv_face_detector_uint8.pb', './face_detectors/opencv_face_detector.pbtxt')

MINDURATION = 3000
MAXDURATION = 15000
#TYPECUT:
#0 - any fixed period
#1 - fixed or zoom-translation period between changed
#2 - fixed period between changed
TYPECUT = 1

def netdetect(img):

    detect_confidence_threshold = 0.3#0.7

    h, w = img.shape[:2]

    blob = cv.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))

    NET.setInput(blob)
    detections = NET.forward()
    rects = []

    for i in range(0, detections.shape[2]):
        # filter out weak detections by ensuring the predicted
        # probability is greater than a minimum threshold
        if detections[0, 0, i, 2] > detect_confidence_threshold:
            # compute the (x, y)-coordinates of the bounding box for
            # the object, then update the bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            rects.append((startX, startY, endX - startX, endY - startY))

    return rects

class VideoCut():

    frames = []
    ndetected = 0

    def getframes(self):
        return self.frames;

    def countframes(self):
        return len(self.frames)

    def reset(self):
        self.frames = []
        self.ndetected = 0

    def processframe(self, img):
        self.frames.append(img)
        #rects = CASCADE.detectMultiScale(cv.cvtColor(img, cv.COLOR_BGR2GRAY), scaleFactor = 1.2, minNeighbors = 5)
        rects = netdetect(img)
        if len(rects):
            self.ndetected += 1

    def isuseful(self):
        return self.ndetected / len(self.frames) >= 0.5

def intersection(x1, y1, angle1, x2, y2, angle2):
    #returned:
    #x, y: infinity for parallel
    #f: -1 - both direct from intersection point; 1 - both direct to intersection point; 0 - different directions;

    eps = np.pi / 180

    if abs(angle1 - angle2) < eps:
        return np.inf, np.inf, 1

    if abs(angle1 - angle2 + np.sign(angle2) * np.pi) < eps:
        return np.inf, np.inf, 0

    if abs(np.pi / 2 - abs(angle1)) < eps:

       k2 = np.tan(angle2)
       b2 = y2 - x2 * k2
       x = x1
       y = k2 * x + b2

    elif abs(np.pi / 2 - abs(angle2)) < eps:

       k1 = np.tan(angle1)
       b1 = y1 - x1 * k1
       x = x2
       y = k1 * x + b1

    else:

       k1 = np.tan(angle1)
       b1 = y1 - x1 * k1
       k2 = np.tan(angle2)
       b2 = y2 - x2 * k2
       x = (b2 - b1) / (k1 - k2)
       y = k1 * x + b1

    a1 = np.arctan2(y - y1, x - x1)
    a2 = np.arctan2(y - y2, x - x2)

    if abs(angle1 - a1) < eps and abs(angle2 - a2) < eps:
        f = 1
    elif not abs(angle1 - a1) < eps and not abs(angle2 - a2) < eps:
        f = -1
    else:
        f = 0

    return x, y, f

def detectback(size, flow):
    #tested on size 426x240

    point_count = 100
    threshold_factor = 0.4
    eps_value_fixed = 1
    eps_threshold_near = 0.2
    eps_angle_near = 45 * np.pi / 180
    eps_angle_far = 20 * np.pi / 180

    ret = 0#0 - change back; 1 - fixed back; 2 - zoom-translation back

    #points count for fixed
    fixeds = 0

    #hypothesis fields: x; y; -1 - unzoom, 1 - zoom; list indexes of points (for x - infinity: y as angle; zoom as 1)
    #hypotheses list
    hypotheses = []
    #best hypothesis index
    hypothesis_idx = -1

    points = []

    k = 0 
    while (True):

        #stop condition and return:
        if k == point_count:
            #print(len(hypotheses))
            #print(hypotheses[hypothesis_idx])
            a = [fixeds, len(hypotheses[hypothesis_idx][3]) if hypothesis_idx != -1 else 0]
            #print(a)
            s = list(reversed(sorted(a)))
            #print(s)
            if s[0] / point_count > threshold_factor:
                ret = np.array(a).argmax() + 1
            break
        k += 1

        x, y = np.random.randint(0, size[0]), np.random.randint(0, size[1])
        fx, fy = flow[y, x].T
        angle = np.arctan2(fy, fx)
        value = np.sqrt(fx*fx + fy*fy)

        #checking fixed
        if value < eps_value_fixed:
            fixeds += 1

        #new point index
        idx_new = len(points)

        #flags for points for adding new hypotheses
        use_points = [True for _ in points]

        #search suitable hypotheses
        for i in range(len(hypotheses)):

            #checking direction

            if hypotheses[i][0] != np.inf:
                dx, dy = (hypotheses[i][0] - x) / size[0], (hypotheses[i][1] - y) / size[1]
                r = np.sqrt(dx * dx + dy * dy)
                a = np.arctan2(dy, dx)
                da = eps_angle_near if r < eps_threshold_near else eps_angle_far
            else:
                a = hypotheses[i][1]
                da = eps_angle_far

            if abs(a - angle) < da or abs(a - angle + np.sign(angle) * np.pi) < da:

                #reset adding hypothesis flags
                for j in range(len(hypotheses[i][3])):
                    use_points[j] = False

                hypotheses[i][3].append(idx_new)
                if len(hypotheses[i][3]) > len(hypotheses[hypothesis_idx][3]):
                    hypothesis_idx = i

        #all points that participate in suitable hypotheses should be excluded to add new hypotheses for the intersection of movements
        #add new hypotheses for the intersection of the motion of existing points
        for i in range(len(use_points)):
            if use_points[i]:
                xc, yc, fc = intersection(x, y, angle, points[i][0], points[i][1], points[i][2])
                if fc != 0:
                    hypotheses.append([xc, yc if xc != np.inf else angle, fc, [i, idx_new]])

        points.append([x, y, angle])

    return ret

def savevideo(title, count, size, fps, frames):

    path = os.path.join('./cuts/', title)

    if not os.path.exists(path):
        os.mkdir(path)

    filename = os.path.join(path, '%s_%03d.avi' % (title, count))

    fourcc = cv.VideoWriter_fourcc(*'XVID')

    out = cv.VideoWriter(filename, fourcc, fps, size)

    for frm in frames:
        out.write(frm)

    out.release()

    print('saved', filename)

def vsearch(title, cam):

    userbreak = False

    cut = VideoCut()
    iscandidate = True
    count = 0

    cap = cv.VideoCapture(cam)

    if cap is None or not cap.isOpened():
        print('bad', cam)
        return False, userbreak

    size = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    delay = int(1000 / fps)
    timecode = 0
    timecode_log = 0
    period_log = 1000

    gray_prev = None
    flow = None
    r_prev = None
    r = None
    r_past = None

    names = ['changed', 'fixed', 'zoom-translation']

    print('opened', cam)

    t0 = time()
    nframes = 0

    while True:

        success, img = cap.read()

        if not success:
            break

        nframes += 1

        if timecode - timecode_log >= period_log:
            print(timecode, names[r])
            timecode_log = timecode

        h, w = img.shape[:2]

        gray = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (426, 240))

        if not gray_prev is None:

            flow = cv.calcOpticalFlowFarneback(gray_prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            r = detectback((426, 240), flow)
            if r != r_prev:
                r_past = r_prev
                print(timecode, names[r])
            r_prev = r

        gray_prev = gray

        if TYPECUT == 0:

            if iscandidate:

                if r != 1 and (not r is None):

                    if cut.countframes() * delay >= MINDURATION:
                        if cut.isuseful():
                            savevideo(title, count, size, fps, cut.getframes())
                            count += 1
                    cut.reset()
                    iscandidate = False

                else:
                    if cut.countframes() * delay >= MAXDURATION:
                        if cut.isuseful():
                            savevideo(title, count, size, fps, cut.getframes())
                            count += 1
                        cut.reset()
            else:
                if r == 1:
                    iscandidate = True

        if TYPECUT == 1:

            if r == 0 and (not r is None):

                if cut.countframes() * delay >= MINDURATION:
                    if cut.isuseful():
                        savevideo(title, count, size, fps, cut.getframes())
                        count += 1
                cut.reset()

            else:
                if cut.countframes() * delay >= MAXDURATION:
                    if cut.isuseful():
                        savevideo(title, count, size, fps, cut.getframes())
                        count += 1
                    cut.reset()

        if TYPECUT == 2:

            if iscandidate:

                if r != 1 and (not r is None):

                    if r == 0:
                        if cut.countframes() * delay >= MINDURATION:
                           if cut.isuseful():
                                savevideo(title, count, size, fps, cut.getframes())
                                count += 1
                    else:
                        iscandidate = False

                    cut.reset()

                else:
                    if cut.countframes() * delay >= MAXDURATION:
                        if cut.isuseful():
                            savevideo(title, count, size, fps, cut.getframes())
                            count += 1
                        cut.reset()
            else:
                if r == 0:
                    iscandidate = True

        if iscandidate:
            cut.processframe(img.copy())

        if not r is None:
            if not r_past is None:
                cv.putText(img, 'past: ' + names[r_past], (0, h - 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv.putText(img, 'now: ' + names[r], (0, h - 2), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        #?cv.imshow(title, img)

        ch = cv.waitKey(5)
        if ch == 27:#ESC
            userbreak = True
            print('userbreak')
            break

        timecode += delay

    t1 = time()

    if cut.countframes() * delay >= MINDURATION:
        if cut.isuseful():
            savevideo(title, count, size, fps, cut.getframes())

    cap.release()
    #?cv.destroyAllWindows()

    print('closed', cam)
    print('frames:', nframes)
    print('elapsed:', t1 - t0)

    return True, userbreak

def main():

    userbreak = False

    with open('ytbs.txt') as f:
        ytbs = f.read().splitlines()

    for i in range(len(ytbs)):

        print(ytbs[i])

        if not ytbs[i][0] in ['+', '-']:

            yt = YouTube(ytbs[i])

            for s in yt.streams.all():
                print(s)

            stream = yt.streams.get_by_itag(22)

            if not stream is None:

                cam = stream.url

                r, userbreak = vsearch('%03d' % (i), cam)

                if not userbreak:
                    ytbs[i] = ('+' if r else '-') + ytbs[i]

            else:
                    ytbs[i] = '-' + ytbs[i]

            with open('ytbs.txt', 'w') as f:
                f.write('\n'.join(ytbs))

            if userbreak:
                break

if __name__ == '__main__':
    main()
