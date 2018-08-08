from Queue import Queue
import threading
import cv2
import time, random
from rfcn_video import process_image_fun
from rfcn_video import init_detect_model

def main():
    print("RUN PROGRAM")
    videoFile = '/workspace/mnt/group/terror/lidexu/Git/video-inference/test/testSample_terror.mp4'
    savePath = '/workspace/mnt/group/terror/lidexu/Git/video-inference/test/reslut.json'
    savefileop = open(savePath, 'a+')
    visualizePath = '/workspace/mnt/group/terror/lidexu/Git/video-inference/test/vis-result'
    cap = cv2.VideoCapture(videoFile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    print(fourcc)
    videoWriter = cv2.VideoWriter(
        '/workspace/mnt/group/terror/lidexu/Git/video-inference/test/bkResult.avi',  fourcc, fps, (600, 1000))

    print('cap is open', cap.isOpened())
    count = 0
    model_params_list = init_detect_model()
    while True:
        ret, image = cap.read()
        if(ret == False):
            break
        im = process_image_fun(
            imagesPath=image, fileOp=savefileop, vis=visualizePath, model_params_list=model_params_list, count=count)
        #print(im)

        cv2.waitKey(10)

        videoWriter.write(im)

        # cv2.imshow('cap video', frame)
        count += 1

        if cv2.waitKey(5) & 0xFF == ord('q'):
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
