import cv2
import mediapipe as mp
import numpy as np
from IK_Solver import SkeletonIKSolver
import torch
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mlp


MEDIAPIPE_POSE_KEYPOINTS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
]   # 33

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
# WEIGHTS = {
#     'left_ear': 0.04,
#     'right_ear': 0.04,
#     'left_shoulder': 0.1,
#     'right_shoulder': 0.1,
#     'left_elbow': 0.02,
#     'right_elbow': 0.02,
#     'left_wrist': 0.01,
#     'right_wrist': 0.01,
#     'left_hip': 0.18,
#     'right_hip': 0.18,
#     'left_knee': 0.03,
#     'right_knee': 0.03,
#     'left_ankle': 0.12,
#     'right_ankle': 0.12,
# }

def intrinsic_from_fov(fov,width,height):
    normed_int = np.array([
        [0.5/(np.tan(fov/2)*(width/max(width,height))),0.,0.5],
        [0.,0.5/(np.tan(fov/2)*(height/max(width,height))),0.5],
        [0.,0.,1.]
    ],dtype=np.float32)
    return normed_int*np.array([width,height,1],dtype=np.float32).reshape(3,1)

def moving_least_square(x, y, w):
    # 1-D MLS: x: (..., N), y: (..., N), w: (..., N)
    p = np.stack([np.ones_like(x), x], axis=-2)             # (..., 2, N)
    M = p @ (w[..., :, None] * p.swapaxes(-2, -1))
    a = np.linalg.solve(M, (p @ (w * y)[..., :, None]))
    a = a.squeeze(-1)
    return a

def mls_smooth(input_t, input_y, query_t: float, smooth_range: float):
    # 1-D MLS: input_t: (N), input_y: (..., N), query_t: scalar
    if len(input_y) == 1:
        return input_y[0]
    input_t = np.array(input_t) - query_t
    input_y = np.stack(input_y, axis=-1)
    broadcaster = (None,)*(len(input_y.shape) - 1)
    w = np.maximum(smooth_range - np.abs(input_t), 0)
    coef = moving_least_square(input_t[broadcaster], input_y, w[broadcaster])
    return coef[..., 0]



# mp.solutions.drawing_utils用于绘制
mp_drawing = mp.solutions.drawing_utils

# 参数：1、颜色，2、线条粗细，3、点的半径
DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 1, 1)
DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 1, 1)

# mp.solutions.pose，是人的骨架
mp_pose = mp.solutions.pose
# 参数：1、是否检测静态图片，2、姿态模型的复杂度，3、结果看起来平滑（用于video有效），4、检测阈值，5、跟踪阈值
pose_mode = mp_pose.Pose(model_complexity=2,min_detection_confidence=0.5, min_tracking_confidence=0.5)

#mp.solutions.hands，是人的手
# mp_hands = mp.solutions.hands
# 参数：1、是否检测静态图片，2、手的数量，3、检测阈值，4、跟踪阈值
# hands_mode = mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.1)

# cap = cap = cv2.VideoCapture(0)  # 0表示第一个摄像头
bone_euler_sequence, scale_sequence, location_sequence = [], [], []

barycenter_weight = np.array([WEIGHTS.get(kp, 0.) for kp in MEDIAPIPE_POSE_KEYPOINTS])

cap = cv2.VideoCapture('/home/qiu/下载/dance2.mp4')
frame_rate = cap.get(cv2.CAP_PROP_FPS)

frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
pose_rvec, pose_tvec = None, None
pose_kpts2d = pose_kpts3d = None
smooth_range = 10 * (1 / frame_rate)
smooth_range_barycenter = 30 * (1 / frame_rate)
barycenter_history= []
pose_history = []
left_hand_history = []
right_hand_history = []

kpts3ds = []
frame_delta=1.0 / 30.0
t = 0

k = intrinsic_from_fov(np.pi / 3, frame_width, frame_height)
# Initialize the skeleton IK solver
skeleton_ik_solver = SkeletonIKSolver(
    model_path='/home/qiu/ppj/blender-4.1.0-linux-x64/tmp2skeleton',
    track_hands=False,
    smooth_range=15 * (1 / frame_rate),
)


plt.ion()
fig = plt.figure()

while cap.isOpened():
    success, image = cap.read()
    # try:
    #     im_height,im_width, c = image.shape
    # except:
    #     break
    if not success:
        print("Ignoring empty camera frame.")
        break
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pose_kpts2d = pose_kpts3d = barycenter =None

    '''
    mp_holistic.PoseLandmark类中共33个人体骨骼点
    '''
    # 处理RGB图像
    results = pose_mode.process(image1)
    # # 绘制
    # mp_drawing.draw_landmarks(
    #     image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)

    image_landmarks = np.array(
        [[lm.x*frame_width, lm.y*frame_height] for lm in results.pose_landmarks.landmark])
    world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark])
    visible = np.array([lm.visibility > 0.2 for lm in results.pose_landmarks.landmark])
    # world_landmarks[11:,2]*=2
    world_landmarks[11:,2]+=world_landmarks[7,2]-world_landmarks[11,2]


    # print(world_landmarks[7])
    # exit()
    if visible.sum()<6:
        continue
    # get transformation matrix from world coordinate to camera coordinate
    _, rvec, tvec = cv2.solvePnP(world_landmarks[visible], image_landmarks[visible], k, np.zeros(5), rvec=pose_rvec,
                                 tvec=pose_tvec, useExtrinsicGuess=pose_rvec is not None)
    if tvec[2]<0:
        continue
    rmat, _ = cv2.Rodrigues(rvec)

    # get camera coordinate of all keypoints
    kpts3d_cam = world_landmarks @ rmat.T + tvec.T

    # force projected x, y to be identical to visibile image_landmarks
    kpts3d_cam_z = kpts3d_cam[:, 2].reshape(-1, 1)
    kpts3d_cam[:, :2] = (np.concatenate([image_landmarks, np.ones((image_landmarks.shape[0], 1))],
                                        axis=1) @ np.linalg.inv(k).T * kpts3d_cam_z)[:, :2]
    # kpts3d_cam[11:,2]/=1.1
    kpts3d = kpts3d_cam


    pose_kpts2d = image_landmarks
    barycenter = np.average(kpts3d, axis=0, weights=barycenter_weight)
    pose_kpts3d = kpts3d - barycenter
    pose_rvec, pose_tvec = rvec, tvec
    barycenter_history.append((barycenter, t))
    pose_history.append((kpts3d, t))
    barycenter_list = [barycenter for barycenter, t_ in barycenter_history if
                       abs(t_ - t) < smooth_range_barycenter]
    barycenter_t = [t_ for barycenter, t_ in barycenter_history if abs(t_ - t) < smooth_range_barycenter]
    if len(barycenter_t) == 0:
        barycenter = np.zeros(3)
    else:
        barycenter = mls_smooth(barycenter_t, barycenter_list, t, smooth_range_barycenter)

    # Get smoothed pose keypoints
    pose_kpts3d_list = [kpts3d for kpts3d, t_ in pose_history if abs(t_ - t) < smooth_range]
    pose_t = [t_ for kpts3d, t_ in pose_history if abs(t_ - t) < smooth_range]
    pose_kpts3d = None if not any(abs(t_ - t) < frame_delta * 0.6 for t_ in pose_t) else mls_smooth(
        pose_t, pose_kpts3d_list, t, smooth_range)

    all_kpts3d = pose_kpts3d if pose_kpts3d is not None else np.zeros((len(MEDIAPIPE_POSE_KEYPOINTS), 3))
    all_valid = np.full(len(MEDIAPIPE_POSE_KEYPOINTS), pose_kpts3d is not None)

    skeleton_ik_solver.fit(torch.from_numpy(all_kpts3d).float(),torch.from_numpy(all_valid).bool(),t)
    bone_euler = skeleton_ik_solver.get_smoothed_bone_euler(t)
    location = skeleton_ik_solver.get_smoothed_location(t)
    scale = skeleton_ik_solver.get_scale()

    bone_euler_sequence.append(bone_euler.numpy())
    location_sequence.append(location.detach().numpy())
    scale_sequence.append(scale*0.1)

    kpts3ds.append((all_kpts3d, all_valid))

    # kpts3d_homo = kpts3d @ k.T
    # kpts2d = kpts3d_homo[:, :2] / kpts3d_homo[:, 2:]
    # 处理RGB图像
    # results = hands_mode.process(image1)
    # 绘制
    # if results.multi_hand_landmarks:
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         mp_drawing.draw_landmarks(
    #             image, hand_landmarks, mp_hands.HAND_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)

    # for a, b in MEDIAPIPE_POSE_CONNECTIONS:
    #     if all_valid[a] == 0 or all_valid[b] == 0:
    #         continue
    #     cv2.line(image, (int(kpts2d[a, 0]), int(kpts2d[a, 1])), (int(kpts2d[b, 0]), int(kpts2d[b, 1])),
    #              (0, 255, 0), 1)
    # for i in range(kpts2d.shape[0]):
    #     if all_valid[i] == 0:
    #         continue
    #     cv2.circle(image, (int(kpts2d[i, 0]), int(kpts2d[i, 1])), 2, (0, 0, 255), -1)
    # cv2.imshow('MediaPipe Pose', cv2.resize(image,(image.shape[1]//4,image.shape[0]//4)))
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


    # fig.clf()
    # ax = fig.add_axes(Axes3D(fig))
    # ax.view_init(0, 0,-90)
    # ax.scatter(pose_kpts3d[:,0], pose_kpts3d[:,1], pose_kpts3d[:,2])
    # for a, b in MEDIAPIPE_POSE_CONNECTIONS:
    #     if all_valid[a] == 0 or all_valid[b] == 0:
    #         continue
    #     ax.plot(np.array([pose_kpts3d[a,0],pose_kpts3d[b,0]]), np.array([pose_kpts3d[a,1],pose_kpts3d[b,1]]), np.array([pose_kpts3d[a,2],pose_kpts3d[b,2]]), label='parametric curve')
    # plt.pause(0.001)

    t += 1.0 / frame_rate
# plt.ioff()
# plt.show()
# exit()
# print("Save animation result...")
with open('tmp/bone_animation_data.pkl', 'wb') as fp:
    pickle.dump({
        'fov': np.pi / 3,
        'frame_rate': frame_rate,
        'bone_names': skeleton_ik_solver.optimizable_bones,
        'bone_euler_sequence': bone_euler_sequence,
        'location_sequence': location_sequence,
        'scale': np.mean(scale_sequence),
        'all_bone_names': skeleton_ik_solver.all_bone_names
    }, fp)

pose_mode.close()
cv2.destroyAllWindows()
cap.release()

{"callBackUrl":"https://tst-qhfz123.etonedu.cn/ems/nzp-ems-exif-server/rest/etd/face-feature/save-or-update","photoUrl":"https://cos-test-qhfz123.etonedu.cn:9000/qhfz-pub/saas_bizedu/course/public/2024/04/16/EF20240416160858165BJEGB/%e6%9c%ac%e4%ba%ba.jpg","roleType":1,"schoolId":2111795,"serverUrl":"http://119.91.23.117:81","userId":11544245,"userName":"陈玉涛"}