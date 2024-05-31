import mediapipe as mp
import cv2
import numpy as np

MEDIAPIPE_POSE_KEYPOINTS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
]   # 33

MEDIAPIPE_HAND_KEYPOINTS = [
    "wrist", "thumb1", "thumb2", "thumb3", "thumb4",
    "index1", "index2", "index3", "index4",
    "middle1", "middle2", "middle3", "middle4",
    "ring1", "ring2", "ring3", "ring4",
    "pinky1", "pinky2", "pinky3", "pinky4"
]   # 21

ALL_KEYPOINTS = MEDIAPIPE_POSE_KEYPOINTS + ['left_' + s for s in MEDIAPIPE_HAND_KEYPOINTS] + ['right_' + s for s in MEDIAPIPE_HAND_KEYPOINTS]

MEDIAPIPE_POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)]

WEIGHTS = {
    'left_ear': 0.04,
    'right_ear': 0.04,
    'left_shoulder': 0.18,
    'right_shoulder': 0.18,
    'left_elbow': 0.02,
    'right_elbow': 0.02,
    'left_wrist': 0.01,
    'right_wrist': 0.01,
    'left_hip': 0.2,
    'right_hip': 0.2,
    'left_knee': 0.03,
    'right_knee': 0.03,
    'left_ankle': 0.02,
    'right_ankle': 0.02,
}

import time

cap = cv2.VideoCapture("dance.mp4")

cTime = 0
pTime = time.time()

mp_pose = mp.solutions.pose.Pose(min_tracking_confidence=0.8,min_detection_confidence=0.5)
mp_hands = mp.solutions.hands.Hands(max_num_hands=2)
# mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5)

def intrinsic_from_fov(fov,width,height):
    normed_int = np.array([
        [0.5/(np.tan(fov/2)*(width/max(width,height))),0.,0.5],
        [0.,0.5/(np.tan(fov/2)*(height/max(width,height))),0.5],
        [0.,0.,1.]
    ],dtype=np.float32)
    return normed_int*np.array([width,height,1],dtype=np.float32).reshape(3,1)

def get_camera_space_lamdmarks(image_landmarks,world_landmarks,visible,rvec,tvec,k):
    _,rvec,tvec = cv2.solvePnP(world_landmarks[visible],image_landmarks[visible],k,np.zeros(5),rvec=rvec,tvec=tvec,useExtrinsicGuss=tvec is not None)

    rmat,_ = cv2.Rodrigues(rvec)

    kpts3d_cam = world_landmarks@rmat.T+tvec.T

    kpts3d_cam_z = kpts3d_cam[:,2].reshape(-1,1)
    kpts3d_cam[:,:2] = (np.concatenate([image_landmarks,np.ones((image_landmarks.shape[0],1))],axis=1)@np.linalg.inv(k).T*kpts3d_cam_z)[:,:2]
    return kpts3d_cam,rvec,tvec



k = intrinsic_from_fov(np.pi/4,180,180)
barycenter_weight = np.array([WEIGHTS.get(kp,0.) for kp in MEDIAPIPE_POSE_KEYPOINTS])


while True:
    ctime = time.time()
    fps = 1 / (ctime - pTime)
    pTime = cTime

    success, frame = cap.read()

    if not success:
        break

    frame.flags.writeable = False

    result = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB))
    image_landmarks = np.array([[i.x,i.y] for i in result.pose_landmarks.landmark])
    world_landmarks = np.array([[i.x,i.y] for i in result.pose_landmarks.landmark])
    visible = [i.visibility>0.2 for i in result.pose_landmarks.landmark]

    kpts3d,rvec,tvec = get_camera_space_lamdmarks(image_landmarks,world_landmarks,visible,None,None,k)

    pose_kpts2d = image_landmarks
    barycenter = np.average(kpts3d,axis=0,weights=barycenter_weight)

    pose_kpts3d = kpts3d-barycenter
    pose_rvec = rvec
    pose_tvec = tvec
    t = 0
    barycenter_history = [(barycenter,t)]
    pose_history = [(kpts3d,t)]

    barycenter_list = [barycenter for bartcenter_,t in barycenter_history if abs(t-0)<1]
    

    if cv2.waitKey(10) & 0XFF == ord('q'):
        break
cap.release()


