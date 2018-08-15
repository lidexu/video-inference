from Queue import Queue
import threading
import cv2
import time, random
from rfcn_video import process_image_fun
from rfcn_video import init_detect_model
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='video inference')
    parser.add_argument('--videoFile', dest='videoFile', 
                        default=None, required=True, help='', type=str)
    parser.add_argument('--interval', dest='interval',
                        default=1, required=False, help='', type=int)
    
    args = parser.parse_args()
    return args

args = parse_args()


def main():
    print("RUN PROGRAM")
    videoFile = args.videoFile
    frame_inter = args.interval
    savePath = '/workspace/inference/result/reslut.json'
    savefileop = open(savePath, 'a+')
    visualizePath = '/workspace/inference/result/vis_result'
    cap = cv2.VideoCapture(videoFile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
    print(fourcc)
    write_fps = fps
    videoWriter = cv2.VideoWriter(
        '/workspace/inference/result/bkResult.avi',  fourcc, write_fps, size)

    print('cap is open', cap.isOpened())
    count = 0
    frame_infer = 0
    model_params_list = init_detect_model()
    while True:
        print(frame_infer)
        ret, image = cap.read()
        if(ret == False):
            break
        if(count == frame_infer):

            im = process_image_fun(
                imagesPath=image, fileOp=savefileop, vis=visualizePath, model_params_list=model_params_list, count=count)
        #print(im)

            #cv2.waitKey(2)

            videoWriter.write(im)
            frame_infer = frame_infer + frame_inter

        # cv2.imshow('cap video', frame)
        count =  count + 1
        if(count % 50 == 0):
            print("Now process the %d image:"%count)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()     # close all the widows opened inside the program
    cap.release()        # release the video read/write handler
    videoWriter.release()

    #frame_queue = Queue()

    # producer = Producer(frame_queue, videoFile)
    # producer.daemon = True
    # producer.start()

    # print('run Consumer')
    # consumer = Consumer(frame_queue, savefileop, visualizePath)
    # consumer.daemon = True
    # consumer.start()

    # producer.join()
    # consumer.join()


if __name__ == '__main__':
    main()
















# class Producer(threading.Thread):
#     def __init__(self, frame_queue, videoFile=None):
#         super(Producer, self).__init__()
#         self.frame_queue = frame_queue
#         self.videoFile = videoFile

#     def run(self):
#         print('in producer')
#         cap = cv2.VideoCapture(self.videoFile)
#         print('cap is open', cap.isOpened())
#         while True:
#             ret, image = cap.read()
#             print('get frame = ', ret)
#             if(ret == True):
#                 self.frame_queue.put(image)
#             else:
#                 cap = cv2.VideoCapture(self.videoFile)
#                 print('cap is open', cap.isOpened())
    

# class Consumer(threading.Thread):
#     def __init__(self, frame_queue, savefile, visualizePath):
#         super(Consumer, self).__init__()
#         self.frame_queue = frame_queue
#         self.savefile = savefile
#         self.visualizePath = visualizePath
    
#     def run(self):
#         print('in consumer')
    
#         while True:
#             print('frame_queue size', self.frame_queue.qsize())
#             frame = self.frame_queue.get()

#             process_image_fun(urlFlag=0, imagesPath=frame, fileOp=self.savefile, vis=self.visualizePath)
#             # cv2.imshow('cap video', frame)
            
#             if cv2.waitKey(5) & 0xFF == ord('q'):
#                 cv2.destroyAllWindows()
#                 break

# # def main():
# #     print("RUN PROGRAM")
# #     savePath = '/'
# #     savefileop = open(savePath, 'a+')
# #     visualizePath = '/'

# #     frame_queue = Queue()

# #     producer = Producer(frame_queue)
# #     producer.daemon = True
# #     producer.start()

# #     print('run Consumer')
# #     consumer = Consumer(frame_queue, savefileop, visualizePath)
# #     consumer.daemon = True
# #     consumer.start()

# #     producer.join()
# #     consumer.join()
