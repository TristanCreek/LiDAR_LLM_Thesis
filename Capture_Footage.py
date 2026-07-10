# Code adapted from StackOverflow accessed on 10/4/2021
# https://stackoverflow.com/questions/55141315/storing-rtsp-stream-as-video-file-with-opencv-videowriter
import time
import datetime
import os
from threading import Thread
import cv2

dirOut = "path\\test\\videos"
fps = 15
cameraIP = "10.0.3.43"
video_length = 30
user = 'admin'
pwd = 'afrladmin'


class RTSPVideoWriterObject(object):
    def __init__(self, src):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))
        self.last_cap = time.time()
        self.write = 0

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                if(time.time() - self.last_cap >= 1/fps):
                    self.last_cap = time.time()
                    self.write = 1
                    (self.status, self.frame) = self.capture.read()

    def save_vid(self):
        # Set up codec and output video settings
        filename = f"{datetime.datetime.now():%Y-%m-%dT%H%M%S}" + '.avi'
        file = os.path.join(dirOut, filename)
        codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.output_video = cv2.VideoWriter(file, codec, fps, (self.frame_width, self.frame_height))
        # Save obtained frame into video output file
        for i in range(video_length * fps):
            while True:
                if self.write:
                    self.output_video.write(self.frame)
                    self.write = 0
                    break
        self.output_video.release()

    def show_frame(self):
        # Display frames in main program
        if self.status:
            cv2.imshow('frame', self.frame)

        key = cv2.waitKey(1)
        # Record video of length according to global variable video_length
        # Pauses live footage while recording
        if key == ord('v'):
            self.save_vid()
        # Press Q on keyboard to stop recording
        elif key == ord('q'):
            self.capture.release()
            self.output_video.release()
            cv2.destroyAllWindows()
            exit(1)

if __name__ == '__main__':
    # Connects with High Resolution 1920x1080
    src = 'rtsp://' + user + ":" + pwd + "@" + cameraIP + ":554/cam/realmonitor?channel=1&subtype=0"
    video_stream_widget = RTSPVideoWriterObject(src)
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass