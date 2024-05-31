MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS = [
    'Forehead_M_01', 'Eyeild_L_Dn_01', 'Eye_L_01', 'Eyeild_L_Up_01', 'Eyeild_R_Dn_01', 'Eye_R_01', 'Eyeild_R_Up_01',
    'Hair_F_L_01', 'Hair_F_R_01', 'Lip_L_01', 'Lip_R_01',
    'Bip001-L-UpperArm', 'Bip001-R-UpperArm', 'Bip001-L-Forearm', 'Bip001-R-Forearm', 'Bip001-L-Hand', 'Bip001-R-Hand', 'Bip001-L-Finger31',
    'Bip001-R-Finger31', 'Bip001-L-Finger11', 'Bip001-R-Finger11', 'Bip001-L-Finger01', 'Bip001-R-Finger01',
    'Bip001-L-Thigh', 'Bip001-R-Thigh', 'Bip001-L-Calf', 'Bip001-R-Calf', 'Bip001-L-Foot', 'Bip001-R-Foot'
]
# MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS = [
#     'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
#     'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
#     'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky1',
#     'right_pinky1', 'left_index1', 'right_index1', 'left_thumb', 'right_thumb',
#     'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
#     'left_toe', 'right_toe'
# ]
MEDIAPIPE_KEYPOINTS_WITH_HANDS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky_dummy',
    'right_pinky_dummy', 'left_index_dummy', 'right_index_dummy', 'left_thumb_dummy', 'right_thumb_dummy',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_toe', 'right_toe',

    "left_wrist_dummy", "left_thumb1", "left_thumb2", "left_thumb3", "left_thumb4",
    "left_index1", "left_index2", "left_index3", "left_index4",
    "left_middle1", "left_middle2", "left_middle3", "left_middle4",
    "left_ring1", "left_ring2", "left_ring3", "left_ring4",
    "left_pinky1", "left_pinky2", "left_pinky3", "left_pinky4",

    "right_wrist_dummy", "right_thumb1", "right_thumb2", "right_thumb3", "right_thumb4",
    "right_index1", "right_index2", "right_index3", "right_index4",
    "right_middle1", "right_middle2", "right_middle3", "right_middle4",
    "right_ring1", "right_ring2", "right_ring3", "right_ring4",
    "right_pinky1", "right_pinky2", "right_pinky3", "right_pinky4",
]
DEFAULT_BONES = [
    "Dummy_Root",
    "Bip001",
    "Bip001-Pelvis",
    "Bip001-Spine",
    "Bip001-Spine1",
    "Bip001-Spine2",
    "Bip001-Neck",
    "Bip001-L-Clavicle",
    "Bip001-L-UpperArm",
    "Bip001-L-Forearm",
    "Bip001-L-Hand",
    "Bip001-L-Finger0",
    "Bip001-L-Finger01",
    "Bip001-L-Finger1",
    "Bip001-L-Finger11",
    "Bip001-L-Finger2",
    "Bip001-L-Finger21",
    "Bip001-L-Finger3",
    "Bip001-L-Finger31",
    "Clavicle_Twist_L",
    "Acc_Armband_01",
    "Bip001-R-Clavicle",
    "Bip001-R-UpperArm",
    "Bip001-R-Forearm",
    "Bip001-R-Hand",
    "Bip001-R-Finger0",
    "Bip001-R-Finger01",
    "Bip001-R-Finger1",
    "Bip001-R-Finger11",
    "Bip001-R-Finger2",
    "Bip001-R-Finger21",
    "Bip001-R-Finger3",
    "Bip001-R-Finger31",
    "Clavicle_Twist_R",
    "Bip001-Head",
    "FXDummy_Head",
    "Eye_L_01",
    "Lip_R_02",
    "Lip_L_01",
    "Forehead_01",
    "Jaw_01",
    "Lip_R_01",
    "Forehead_M_01",
    "Eye_R_01",
    "Eyeild_R_Dn_01",
    "Eyeild_R_Up_01",
    "Eyeild_R_Up_02",
    "Lip_L_02",
    "Hair_F_R_01",
    "Hair_F_R_02",
    "Hair_F_01",
    "Hair_F_02",
    "Hair_F_L_01",
    "Hair_F_L_02",
    "Hair_R_01",
    "Hair_R_02",
    "Hair_R_03",
    "Hair_R_04",
    "Hair_L_01",
    "Hair_L_02",
    "Hair_L_03",
    "Hair_L_04",
    "Hair_Acc_R_Dn_01",
    "Hair_Acc_R_Dn_02",
    "Hair_Acc_L_Up_01",
    "Hair_Acc_L_Up_02",
    "Hair_Acc_R_UP_01",
    "Hair_Acc_R_UP_02",
    "Hair_Acc_L_Dn_01",
    "Hair_Acc_L_Dn_02",
    "Eyeild_L_Up_01",
    "Eyeild_L_Up_02",
    "Eyeild_L_Dn_01",
    "Acc_Necktie_01",
    "Acc_Necktie_02",
    "Bip001-L-Thigh",
    "Bip001-L-Calf",
    "Bip001-L-Foot",
    "Bip001-R-Thigh",
    "Bip001-R-Calf",
    "Bip001-R-Foot",
    "Skirt_B_01",
    "Skirt_B_02",
    "Skirt_F_01",
    "Skirt_F_02",
    "Skirt_R_01",
    "Skirt_R_02",
    "Skirt_L_01",
    "Skirt_L_02"
]
# DEFAULT_BONES = [
#     'left_eye', 'right_eye', 'left_ear', 'right_ear', 'head', 'neck', 'left_collar', 'right_collar', 'left_shoulder',
#     'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
#     'pelvis', 'spine1', 'spine2', 'spine3', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
#     'right_ankle', 'left_toe', 'right_toe',
#     'left_index1', 'left_index2', 'left_index3', 'left_index4', 'left_middle1', 'left_middle2', 'left_middle3',
#     'left_middle4', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_pinky4', 'left_ring1', 'left_ring2',
#     'left_ring3', 'left_ring4', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'left_thumb4',
#     'right_index1', 'right_index2', 'right_index3', 'right_index4', 'right_middle1', 'right_middle2', 'right_middle3',
#     'right_middle4', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_pinky4', 'right_ring1', 'right_ring2',
#     'right_ring3', 'right_ring4', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'right_thumb4'
# ]
OPTIMIZABLE_BONES = [
    'Bip001-L-Hand', 'Bip001-Neck', 'Bip001-L-Clavicle', 'Bip001-R-Clavicle', 'Bip001-L-UpperArm', 'Bip001-R-UpperArm', 'Bip001-L-Forearm', 'Bip001-R-Forearm',
    'Bip001-L-Hand', 'Bip001-R-Hand',
    'Bip001-Pelvis', 'Bip001-Spine', 'Bip001-Spine1', 'Bip001-Spine2', 'Bip001-L-Thigh', 'Bip001-R-Thigh', 'Bip001-L-Calf', 'Bip001-R-Calf', 'Bip001-L-Foot',
    'Bip001-R-Foot',
    'Bip001-L-Finger1', 'Bip001-L-Finger11', 'Bip001-L-Finger2', 'Bip001-L-Finger21', 'Bip001-L-Finger3',
    'Bip001-L-Finger31', 'Bip001-L-Finger0', 'Bip001-L-Finger01',
    'Bip001-R-Finger1', 'Bip001-R-Finger11', 'Bip001-R-Finger2', 'Bip001-R-Finger21', 'Bip001-R-Finger3',
    'Bip001-R-Finger31', 'Bip001-R-Finger0', 'Bip001-R-Finger01',
]
# OPTIMIZABLE_BONES = [
#     'head', 'neck', 'left_collar', 'right_collar', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
#     'left_wrist', 'right_wrist',
#     'pelvis', 'spine1', 'spine2', 'spine3', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
#     'right_ankle',
#     'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_ring1',
#     'left_ring2', 'left_ring3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_thumb1', 'left_thumb2',
#     'left_thumb3',
#     'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_ring1',
#     'right_ring2', 'right_ring3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_thumb1', 'right_thumb2',
#     'right_thumb3',
# ]
ALIGN_LOCATION_WITH = ['Bip001-L-UpperArm', 'Bip001-R-UpperArm']
ALIGN_SCALE_WITH = [('Bip001-L-UpperArm', 'Bip001-R-UpperArm'), ('Bip001-L-Thigh', 'Bip001-R-Thigh'), ('Bip001-L-UpperArm', 'Bip001-L-Thigh'),
                    ('Bip001-R-UpperArm', 'Bip001-R-Thigh')]
# ALIGN_LOCATION_WITH = ['left_shoulder', 'right_shoulder']
# ALIGN_SCALE_WITH = [('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip'), ('left_shoulder', 'left_hip'),
#                     ('right_shoulder', 'right_hip')]
TARGET_KEYPOINT_PAIRS_WITH_HANDS = [
    # Head
    ('Hair_F_L_01', 'Hair_F_R_01'),

    # Body
    ('Bip001-L-UpperArm', 'Bip001-L-Thigh'),
    ('Bip001-R-UpperArm', 'Bip001-R-Thigh'),
    ('Bip001-L-Thigh', 'Bip001-R-Thigh'),
    ('Bip001-L-UpperArm', 'Bip001-R-UpperArm'),

    # Arms
    ('Bip001-L-Forearm', 'Bip001-L-UpperArm'),
    ('Bip001-L-Hand', 'Bip001-L-Forearm'),
    ('Bip001-R-Forearm', 'Bip001-R-UpperArm'),
    ('Bip001-R-Hand', 'Bip001-R-Forearm'),

    # Left hand
    ('Bip001-L-Hand', 'Bip001-L-Finger31'),
    ('Bip001-L-Hand', 'Bip001-L-Finger11'),
    ('Bip001-L-Finger31', 'Bip001-L-Finger11'),
    # Left hand fingers
    ('Bip001-L-Finger0', 'Bip001-L-Finger01'),
    ('Bip001-L-Finger1', 'Bip001-L-Finger11'),
    ('Bip001-L-Finger2', 'Bip001-L-Finger21'),
    ('Bip001-L-Finger3', 'Bip001-L-Finger31'),

    # Right Hand
    ('Bip001-R-Hand', 'Bip001-R-Finger31'),
    ('Bip001-R-Hand', 'Bip001-R-Finger11'),
    ('Bip001-R-Finger31', 'Bip001-R-Finger11'),
    # Right hand fingers
    ('Bip001-R-Finger0', 'Bip001-R-Finger01'),
    ('Bip001-R-Finger1', 'Bip001-R-Finger11'),
    ('Bip001-R-Finger2', 'Bip001-R-Finger21'),
    ('Bip001-R-Finger3', 'Bip001-R-Finger31'),

    # Legs
    ('Bip001-L-Thigh', 'Bip001-L-Calf'),
    ('Bip001-L-Calf', 'Bip001-L-Foot'),
    ('Bip001-R-Thigh', 'Bip001-R-Calf'),
    ('Bip001-R-Calf', 'Bip001-R-Foot'),

    # hands to hips
    ('Bip001-L-Hand', 'Bip001-L-Thigh'),
    ('Bip001-R-Hand', 'Bip001-R-Thigh'),

    # hand to hand
    ('Bip001-L-Hand', 'Bip001-R-Hand'),
]
# TARGET_KEYPOINT_PAIRS_WITH_HANDS = [
#     # Head
#     ('left_ear', 'right_ear'),
#
#     # Body
#     ('left_shoulder', 'left_hip'),
#     ('right_shoulder', 'right_hip'),
#     ('left_hip', 'right_hip'),
#     ('left_shoulder', 'right_shoulder'),
#
#     # Arms
#     ('left_elbow', 'left_shoulder'),
#     ('left_wrist', 'left_elbow'),
#     ('right_elbow', 'right_shoulder'),
#     ('right_wrist', 'right_elbow'),
#
#     # Left hand
#     ('left_wrist', 'left_pinky1'),
#     ('left_wrist', 'left_index1'),
#     ('left_pinky1', 'left_index1'),
#     # Left hand fingers
#     ('left_thumb1', 'left_thumb2'),
#     ('left_thumb2', 'left_thumb3'),
#     ('left_thumb3', 'left_thumb4'),
#     ('left_index1', 'left_index2'),
#     ('left_index2', 'left_index3'),
#     ('left_index3', 'left_index4'),
#     ('left_middle1', 'left_middle2'),
#     ('left_middle2', 'left_middle3'),
#     ('left_middle3', 'left_middle4'),
#     ('left_ring1', 'left_ring2'),
#     ('left_ring2', 'left_ring3'),
#     ('left_ring3', 'left_ring4'),
#     ('left_pinky1', 'left_pinky2'),
#     ('left_pinky2', 'left_pinky3'),
#     ('left_pinky3', 'left_pinky4'),
#
#     # Right Hand
#     ('right_wrist', 'right_pinky1'),
#     ('right_wrist', 'right_index1'),
#     ('right_pinky1', 'right_index1'),
#     # Right hand fingers
#     ('right_thumb1', 'right_thumb2'),
#     ('right_thumb2', 'right_thumb3'),
#     ('right_thumb3', 'right_thumb4'),
#     ('right_index1', 'right_index2'),
#     ('right_index2', 'right_index3'),
#     ('right_index3', 'right_index4'),
#     ('right_middle1', 'right_middle2'),
#     ('right_middle2', 'right_middle3'),
#     ('right_middle3', 'right_middle4'),
#     ('right_ring1', 'right_ring2'),
#     ('right_ring2', 'right_ring3'),
#     ('right_ring3', 'right_ring4'),
#     ('right_pinky1', 'right_pinky2'),
#     ('right_pinky2', 'right_pinky3'),
#     ('right_pinky3', 'right_pinky4'),
#
#     # Legs
#     ('left_hip', 'left_knee'),
#     ('left_knee', 'left_ankle'),
#     ('right_hip', 'right_knee'),
#     ('right_knee', 'right_ankle'),
#
#     # hands to hips
#     ('left_wrist', 'left_hip'),
#     ('right_wrist', 'right_hip'),
#
#     # hand to hand
#     ('left_wrist', 'right_wrist'),
# ]
TARGET_KEYPOINT_PAIRS_WITHOUT_HANDS = [
    # Head
    ('Hair_F_L_01', 'Hair_F_R_01'),

    # Body
    ('Bip001-L-UpperArm', 'Bip001-L-Thigh'),
    ('Bip001-R-UpperArm', 'Bip001-R-Thigh'),
    ('Bip001-L-Thigh', 'Bip001-R-Thigh'),
    ('Bip001-L-UpperArm', 'Bip001-R-UpperArm'),

    # Arms
    ('Bip001-L-Forearm', 'Bip001-L-UpperArm'),
    ('Bip001-L-Hand', 'Bip001-L-Forearm'),
    ('Bip001-R-Forearm', 'Bip001-R-UpperArm'),
    ('Bip001-R-Hand', 'Bip001-R-Forearm'),

    # Left hand
    ('Bip001-L-Hand', 'Bip001-L-Finger31'),
    ('Bip001-L-Hand', 'Bip001-L-Finger11'),
    ('Bip001-L-Finger31', 'Bip001-L-Finger11'),

    # Right Hand
    ('Bip001-R-Hand', 'Bip001-R-Finger31'),
    ('Bip001-R-Hand', 'Bip001-R-Finger11'),
    ('Bip001-R-Finger31', 'Bip001-R-Finger11'),

    # Legs
    ('Bip001-L-Thigh', 'Bip001-L-Calf'),
    ('Bip001-L-Calf', 'Bip001-L-Foot'),
    ('Bip001-R-Thigh', 'Bip001-R-Calf'),
    ('Bip001-R-Calf', 'Bip001-R-Foot'),

    # hands to hips
    ('Bip001-L-Hand', 'Bip001-L-Thigh'),
    ('Bip001-R-Hand', 'Bip001-R-Thigh'),

    # hand to hand
    ('Bip001-L-Hand', 'Bip001-R-Hand'),
]
# TARGET_KEYPOINT_PAIRS_WITHOUT_HANDS = [
#     # Head
#     ('left_ear', 'right_ear'),
#
#     # Body
#     ('left_shoulder', 'left_hip'),
#     ('right_shoulder', 'right_hip'),
#     ('left_hip', 'right_hip'),
#     ('left_shoulder', 'right_shoulder'),
#
#     # Arms
#     ('left_elbow', 'left_shoulder'),
#     ('left_wrist', 'left_elbow'),
#     ('right_elbow', 'right_shoulder'),
#     ('right_wrist', 'right_elbow'),
#
#     # Left hand
#     ('left_wrist', 'left_pinky'),
#     ('left_wrist', 'left_index'),
#     ('left_pinky', 'left_index'),
#
#     # Right Hand
#     ('right_wrist', 'right_pinky'),
#     ('right_wrist', 'right_index'),
#     ('right_pinky', 'right_index'),
#
#     # Legs
#     ('left_hip', 'left_knee'),
#     ('left_knee', 'left_ankle'),
#     ('right_hip', 'right_knee'),
#     ('right_knee', 'right_ankle'),
#
#     # hands to hips
#     ('left_wrist', 'left_hip'),
#     ('right_wrist', 'right_hip'),
#
#     # hand to hand
#     ('left_wrist', 'right_wrist'),
# ]
BONE_CONSTRAINTS = {
    # Euler angles in world space, where human stands z up, -y back
    'Bip001-Head': ((-30, 30), (-30, 30), (-45, 45)),
    'Bip001-Neck': ((-30, 30), (-30, 30), (-45, 45)),

    'Bip001-L-Clavicle': ((-15, 15), (-15, 15), (-15, 15)),
    'Bip001-R-Clavicle': ((-15, 15), (-15, 15), (-15, 15)),

    'Bip001-L-UpperArm': ((-45, 45), (-60, 75), (-135, 45)),
    'Bip001-r-UpperArm': ((-45, 45), (-75, 60), (-45, 135)),

    'Bip001-L-Forearm': ((-150, 90), (-5, 5), (-135, 5)),
    'Bip001-r-Forearm': ((-90, 150), (-5, 5), (-5, 135)),

    'Bip001-L-Hand': ((-5, 5), (-75, 75), (-30, 30)),
    'Bip001-R-Hand': ((-5, 5), (-75, 75), (-30, 30)),

    'Bip001-Spine': ((-5, 15), (-20, 20), (-20, 20)),
    'Bip001-Spine1': ((-5, 15), (-20, 20), (-20, 20)),
    'Bip001-Spine2': ((-5, 15), (-20, 20), (-20, 20)),

    'Bip001-L-Thigh': ((-90, 45), (-45, 60), (-45, 45)),
    'Bip001-R-Thigh': ((-90, 45), (-60, 45), (-45, 45)),
    'Bip001-L-Calf': ((-5, 135), (-15, 15), (-5, 5)),
    'Bip001-R-Calf': ((-5, 135), (-15, 15), (-5, 5)),
    'Bip001-L-Foot': ((-45, 90), (-15, 15), (-45, 45)),
    'Bip001-R-Foot': ((-45, 90), (-15, 15), (-45, 45)),

    'Bip001-L-Finger1': ((-3, 3), (-45, 135), (-45, 45)),
    'Bip001-L-Finger11': ((-3, 3), (-45, 120), (-3, 3)),
    'Bip001-L-Finger2': ((-3, 3), (-45, 135), (-45, 45)),
    'Bip001-L-Finger21': ((-3, 3), (-45, 120), (-3, 3)),
    'Bip001-L-Finger3': ((-3, 3), (-45, 135), (-45, 45)),
    'Bip001-L-Finger31': ((-3, 3), (-45, 120), (-3, 3)),
    'Bip001-L-Finger0': ((-3, 3), (-45, 135), (-45, 45)),
    'Bip001-L-Finger01': ((-3, 3), (-45, 120), (-3, 3)),

    'Bip001-R-Finger1': ((-3, 3), (-135, 45), (-45, 45)),
    'Bip001-R-Finger11': ((-3, 3), (-120, 45), (-3, 3)),
    'Bip001-R-Finger2': ((-3, 3), (-135, 45), (-45, 45)),
    'Bip001-R-Finger21': ((-3, 3), (-120, 45), (-3, 3)),
    'Bip001-R-Finger3': ((-3, 3), (-135, 45), (-45, 45)),
    'Bip001-R-Finger31': ((-3, 3), (-120, 45), (-3, 3)),
    'Bip001-R-Finger0': ((-3, 3), (-135, 45), (-45, 45)),
    'Bip001-R-Finger01': ((-3, 3), (-120, 45), (-3, 3)),
}

# BONE_CONSTRAINTS = {
#     # Euler angles in world space, where human stands z up, -y back
#     'head': ((-30, 30), (-30, 30), (-45, 45)),
#     'neck': ((-30, 30), (-30, 30), (-45, 45)),
#
#     'left_collar': ((-15, 15), (-15, 15), (-15, 15)),
#     'right_collar': ((-15, 15), (-15, 15), (-15, 15)),
#
#     'left_shoulder': ((-45, 45), (-60, 75), (-135, 45)),
#     'right_shoulder': ((-45, 45), (-75, 60), (-45, 135)),
#
#     'left_elbow': ((-150, 90), (-5, 5), (-135, 5)),
#     'right_elbow': ((-90, 150), (-5, 5), (-5, 135)),
#
#     'left_wrist': ((-5, 5), (-75, 75), (-30, 30)),
#     'right_wrist': ((-5, 5), (-75, 75), (-30, 30)),
#
#     'spine1': ((-5, 15), (-20, 20), (-20, 20)),
#     'spine2': ((-5, 15), (-20, 20), (-20, 20)),
#     'spine3': ((-5, 15), (-20, 20), (-20, 20)),
#
#     'left_hip': ((-90, 45), (-45, 60), (-45, 45)),
#     'right_hip': ((-90, 45), (-60, 45), (-45, 45)),
#     'left_knee': ((-5, 135), (-15, 15), (-5, 5)),
#     'right_knee': ((-5, 135), (-15, 15), (-5, 5)),
#     'left_ankle': ((-45, 90), (-15, 15), (-45, 45)),
#     'right_ankle': ((-45, 90), (-15, 15), (-45, 45)),
#
#     'left_index1': ((-3, 3), (-45, 135), (-45, 45)), 'left_index2': ((-3, 3), (-45, 135), (-3, 3)),
#     'left_index3': ((-3, 3), (-45, 120), (-3, 3)),
#     'left_middle1': ((-3, 3), (-45, 135), (-45, 45)), 'left_middle2': ((-3, 3), (-45, 135), (-3, 3)),
#     'left_middle3': ((-3, 3), (-45, 120), (-3, 3)),
#     'left_ring1': ((-3, 3), (-45, 135), (-45, 45)), 'left_ring2': ((-3, 3), (-45, 135), (-3, 3)),
#     'left_ring3': ((-3, 3), (-45, 120), (-3, 3)),
#     'left_pinky1': ((-3, 3), (-45, 135), (-45, 45)), 'left_pinky2': ((-3, 3), (-45, 135), (-3, 3)),
#     'left_pinky3': ((-3, 3), (-45, 120), (-3, 3)),
#     'left_thumb1': ((-3, 3), (-45, 135), (-45, 45)), 'left_thumb2': ((-3, 3), (-45, 135), (-3, 3)),
#     'left_thumb3': ((-3, 3), (-45, 120), (-3, 3)),
#
#     'right_index1': ((-3, 3), (-135, 45), (-45, 45)), 'right_index2': ((-3, 3), (-135, 45), (-3, 3)),
#     'right_index3': ((-3, 3), (-120, 45), (-3, 3)),
#     'right_middle1': ((-3, 3), (-135, 45), (-45, 45)), 'right_middle2': ((-3, 3), (-135, 45), (-3, 3)),
#     'right_middle3': ((-3, 3), (-120, 45), (-3, 3)),
#     'right_ring1': ((-3, 3), (-135, 45), (-45, 45)), 'right_ring2': ((-3, 3), (-135, 45), (-3, 3)),
#     'right_ring3': ((-3, 3), (-120, 45), (-3, 3)),
#     'right_pinky1': ((-3, 3), (-135, 45), (-45, 45)), 'right_pinky2': ((-3, 3), (-135, 45), (-3, 3)),
#     'right_pinky3': ((-3, 3), (-120, 45), (-3, 3)),
#     'right_thumb1': ((-3, 3), (-135, 45), (-45, 45)), 'right_thumb2': ((-3, 3), (-135, 45), (-3, 3)),
#     'right_thumb3': ((-3, 3), (-120, 45), (-3, 3)),
# }

import os
import json
import numpy as np
import itertools
from typing import List, Tuple, Dict, Union, Optional
import torch


def load_skeleton_data(path: str):
    with open(os.path.join(path, 'skeleton.json'), 'r') as f:
        skeleton = json.load(f)
    bone_names = skeleton['bone_names']
    bone_parents = skeleton['bone_parents']
    bone_matrix_world_rest = np.load(os.path.join(path, skeleton['bone_matrix_world']))
    bone_matrix = np.load(os.path.join(path, skeleton['bone_matrix_rel']))

    skeleton_remap = skeleton['bone_remap']
    skeleton_remap = {k: v for k, v in skeleton_remap.items() if v is not None}
    skeleton_remap.update({k: k for k in DEFAULT_BONES if k in bone_names})

    return bone_names, bone_parents, bone_matrix_world_rest, bone_matrix, skeleton_remap


def get_optimization_target(bone_parents: Dict[str, str], skeleton_remap: Dict[str, str], track_hand: bool):
    # bones to optimize
    optimizable_bones = [skeleton_remap[b] for b in OPTIMIZABLE_BONES if b in skeleton_remap]

    # target pairs
    if track_hand:
        kpt_pairs = [(a, b) for a, b in TARGET_KEYPOINT_PAIRS_WITH_HANDS if a in skeleton_remap and b in skeleton_remap]
    else:
        kpt_pairs = [(a, b) for a, b in TARGET_KEYPOINT_PAIRS_WITHOUT_HANDS if
                     a in skeleton_remap and b in skeleton_remap]
    joint_pairs = [(skeleton_remap[a], skeleton_remap[b]) for a, b in kpt_pairs]

    # Find bones that has target bones as children
    bone_subset = []
    for t in itertools.chain(*joint_pairs):
        bone_chain = [t]
        while bone_parents[t] is not None:
            t = bone_parents[t]
            bone_chain.append(t)
        for b in reversed(bone_chain):
            if b not in bone_subset:
                bone_subset.append(b)

    if track_hand:
        kpt_pairs_id = torch.tensor(
            [(MEDIAPIPE_KEYPOINTS_WITH_HANDS.index(a), MEDIAPIPE_KEYPOINTS_WITH_HANDS.index(b)) for a, b in kpt_pairs],
            dtype=torch.long)
    else:
        kpt_pairs_id = torch.tensor(
            [(MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS.index(a), MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS.index(b)) for a, b in
             kpt_pairs], dtype=torch.long)
    joint_pairs_id = torch.tensor([(bone_subset.index(a), bone_subset.index(b)) for a, b in joint_pairs],
                                  dtype=torch.long)

    return bone_subset, optimizable_bones, kpt_pairs_id, joint_pairs_id


def get_constraints(bone_names: List[str], bone_matrix_world_rest: np.ndarray, optimizable_bones: List[str],
                    skeleton_remap: Dict[str, str]):
    # Get constraints
    joint_constraints_id = []
    joint_constraints = []
    for k, c in BONE_CONSTRAINTS.items():
        if not (k in skeleton_remap and skeleton_remap[k] in optimizable_bones):
            continue
        b = skeleton_remap[k]
        constraint = []
        rest_mat = bone_matrix_world_rest[bone_names.index(b)]

        # Get local -> world axis
        for i in range(3):
            world_axis = np.argmax(np.abs(rest_mat[:3, i]))
            constr = c[world_axis]
            if rest_mat[world_axis, i] < 0:
                constr = -constr[1], -constr[0]
            constraint.append(constr)

        joint_constraints_id.append(optimizable_bones.index(b))
        joint_constraints.append(constraint)

    joint_constraints_id = torch.tensor(joint_constraints_id, dtype=torch.long)
    joint_constraints = torch.tensor(joint_constraints, dtype=torch.float32)

    return joint_constraints_id, torch.deg2rad(joint_constraints)


def get_align_location(bone_names: List[str], skeleton_remap: Dict[str, str]):
    align_location_kpts = torch.tensor([MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS.index(k) for k in ALIGN_LOCATION_WITH],
                                       dtype=torch.long)
    align_location_joints = torch.tensor([bone_names.index(skeleton_remap[k]) for k in ALIGN_LOCATION_WITH],
                                         dtype=torch.long)
    return align_location_kpts, align_location_joints


def get_align_scale(bone_names: List[str], skeleton_remap: Dict[str, str]):
    align_scale_pairs_kpts = torch.tensor(
        [(MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS.index(a), MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS.index(b)) for a, b in
         ALIGN_SCALE_WITH], dtype=torch.long)
    align_scale_pairs_joints = torch.tensor(
        [(bone_names.index(skeleton_remap[a]), bone_names.index(skeleton_remap[b])) for a, b in ALIGN_SCALE_WITH],
        dtype=torch.long)
    return align_scale_pairs_kpts, align_scale_pairs_joints

# TOP1：自私、别扭、不成熟，
# TOP2：女版
# TOP3：外向、腹黑


