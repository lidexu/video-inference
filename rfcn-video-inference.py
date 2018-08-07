from Queue import Queue
import threading
import cv2
import time, random
from rfcn_dcn_inference_JH_logProcess import process_image_fun

class Producer(threading.Thread):
    def __init__(self, frame_queue, videoFile=None):
        super(Producer, self).__init__()
        self.frame_queue = frame_queue
        self.videoFile = videoFile

    def run(self):
        print('in producer')
        cap = cv2.VideoCapture(self.videoFile)
        print('cap is open', cap.isOpened())
        while True:
            ret, image = cap.read()
            print('get frame = ', ret)
            if(ret == True):
                self.frame_queue.put(image)
            else:
                cap = cv2.VideoCapture(self.videoFile)
                print('cap is open', cap.isOpened())
    

class Consumer(threading.Thread):
    def __init__(self, frame_queue, savefile, visualizePath):
        super(Consumer, self).__init__()
        self.frame_queue = frame_queue
        self.savefile = savefile
        self.visualizePath = visualizePath
    
    def run(self):
        print('in consumer')
    
        while True:
            print('frame_queue size', self.frame_queue.qsize())
            frame = self.frame_queue.get()

            process_image_fun(urlFlag=0, imagesPath=frame, fileOp=self.savefile, vis=self.visualizePath)
            # cv2.imshow('cap video', frame)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

def main():
    print("RUN PROGRAM")
    savePath = '/'
    savefileop = open(savePath, 'a+')
    visualizePath = '/'

    frame_queue = Queue()

    producer = Producer(frame_queue)
    producer.daemon = True
    producer.start()

    print('run Consumer')
    consumer = Consumer(frame_queue, savefileop, visualizePath)
    consumer.daemon = True
    consumer.start()

    producer.join()
    consumer.join()

if __name__ == '__main__':
    main()
