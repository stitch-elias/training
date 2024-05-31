import numpy as np
import cv2

window_size = 5
keypoint_queue = []
def process_3d_keypoints(frame_3d_keypoints):
    keypoint_queue.append(frame_3d_keypoints)

    if len(keypoint_queue)>window_size:
        keypoint_queue.pop(0)
    smoothed_keypoints = np.mean(keypoint_queue,axit=0)

    return smoothed_keypoints

class KalmanFiterWrapper:
    def __init__(self,input_dim,init_error,init_process_var,init_measure_var):
        self.input_dim = input_dim
        self.init_error = init_error
        self.init_process_var = init_process_var
        self.init_measure_var = init_measure_var

        self.kf = cv2.KalmanFilter(self.input_dim,self.input_dim)
        self.kf.transitionMatrix = np.eye(self.input_dim,dtype=np.float32)
        self.kf.measurementMatrix = np.eye(self.input_dim,dtype=np.float32)
        self.kf.processNoiseCov = self.init_process_var*np.eye(self.input_dim,dtype=np.float32)
        self.kf.measurementNoiseCov = self.init_measure_var*np.eye(self.input_dim,dtype=np.float32)
        self.kf.errorCovPost = self.init_error*np.eye(self.input_dim,dtype=np.float32)
        self.statepose = self.kf.statePost

    def filter(self,observation):
        return self.kf.correct(observation)

    def predict(self):
        return self.kf.predict()

class OneEuroFileter:
    def __init__(self,t0,x0,dx0=0,min_cutoff=1.,beta=0,d_cutoff=1.):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_prev = x0
        self.dx_prev = float(dx0)
        self.t_prev = t0

    def smoothing_factor(self,t_e,cutoff):
        r = 2*np.pi*cutoff*t_e
        return r/(r+1)

    def exponetial_smoothing(self,a,x,x_prev):
        return a*x+(1-a)*x_prev

    def filter_signal(self,t,x):
        t_e = t-self.t_prev

        a_d = self.smoothing_factor(t_e,self.d_cutoff)
        dx = (x-self.x_prev)/t_e
        dx_hat = self.exponetial_smoothing(a_d,dx,self.dx_prev)

        cutoff = self.min_cutoff+self.beta*abs(dx_hat)
        a = self.smoothing_factor(t_e,cutoff)
        x_hat = self.exponetial_smoothing(a,x,self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat
