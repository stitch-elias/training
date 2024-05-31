import torch
from skeleton_config import load_skeleton_data, get_optimization_target, get_constraints, get_align_location, get_align_scale, MEDIAPIPE_KEYPOINTS_WITH_HANDS, MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS
import torch.nn.functional as F
@torch.jit.script
def barrier(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    return torch.exp(4 * (x - b)) + torch.exp(4 * (a - x))
def moving_least_square(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
    # 1-D MLS: x: (..., N), y: (..., N), w: (..., N)
    p = torch.stack([torch.ones_like(x), x], dim=-2)             # (..., 2, N)
    M = p @ (w[..., :, None] * p.transpose(-2, -1))
    a = torch.linalg.solve(M, (p @ (w * y)[..., :, None]))
    a = a.squeeze(-1)
    return a

def eval_matrix_world(parents: torch.Tensor, matrix_bones: torch.Tensor, matrix_basis: torch.Tensor) -> torch.Tensor:
    "Deprecated"
    matrix_bones, matrix_basis = matrix_bones.unbind(), matrix_basis.unbind()
    matrix_world = []
    for i in range(len(matrix_bones)):
        local_mat = torch.mm(matrix_bones[i], matrix_basis[i])
        m = local_mat if parents[i] < 0 else torch.mm(matrix_world[parents[i]], local_mat)
        matrix_world.append(m)
    return torch.stack(matrix_world)

def mls_smooth(input_t, input_y, query_t: float, smooth_range: float):
    # 1-D MLS: input_t: (N), input_y: (..., N), query_t: scalar
    if len(input_y) == 1:
        return input_y[0]
    input_t = torch.tensor(input_t) - query_t
    input_y = torch.stack(input_y, axis=-1)
    broadcaster = (None,)*(len(input_y.shape) - 1)
    w = torch.maximum(smooth_range - torch.abs(input_t), torch.tensor(0))
    coef = moving_least_square(input_t[broadcaster], input_y, w[broadcaster])
    return coef[..., 0]

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (
            one, zero, zero, zero,
            zero, cos, -sin, zero,
            zero, sin, cos, zero,
            zero, zero, zero, one
        )
    elif axis == "Y":
        R_flat = (
            cos, zero, sin, zero,
            zero, one, zero, zero,
            -sin, zero, cos, zero,
            zero, zero, zero, one
        )
    elif axis == "Z":
        R_flat = (
            cos, -sin, zero, zero,
            sin, cos, zero, zero,
            zero, zero, one, zero,
            zero, zero, zero, one
        )
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (4, 4))


def euler_angle_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Code MODIFIED from pytorch3d
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3), XYZ
        convention: permutation of "X", "Y" or "Z", representing the order of Euler rotations to apply.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, euler_angles[..., 'XYZ'.index(c)])
        for c in convention
    ]
    # return functools.reduce(torch.matmul, matrices)
    return matrices[2] @ matrices[1] @ matrices[0]


class SkeletonIKSolver:
    def __init__(self, model_path: str, track_hands: bool = True, **kwargs):
        # load skeleton model data
        all_bone_names, all_bone_parents, all_bone_matrix_world_rest, all_bone_matrix, skeleton_remap = load_skeleton_data(
            model_path)

        self.keypoints = MEDIAPIPE_KEYPOINTS_WITH_HANDS if track_hands else MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS

        # skeleton structure info
        self.all_bone_names = all_bone_names
        self.all_bone_parents = all_bone_parents
        self.all_bone_parents_id = torch.tensor(
            [(all_bone_names.index(all_bone_parents[b]) if all_bone_parents[b] is not None else -1) for b in
             all_bone_parents], dtype=torch.long)
        self.all_bone_matrix: torch.Tensor = torch.from_numpy(all_bone_matrix).float()

        # Optimization target
        bone_subset, optimizable_bones, kpt_pairs_id, joint_pairs_id = get_optimization_target(all_bone_parents,
                                                                                               skeleton_remap,
                                                                                               track_hands)
        self.joint_pairs_a, self.joint_pairs_b = joint_pairs_id[:, 0], joint_pairs_id[:, 1]
        self.kpt_pairs_a, self.kpt_pairs_b = kpt_pairs_id[:, 0], kpt_pairs_id[:, 1]
        self.bone_parents_id = torch.tensor(
            [(bone_subset.index(all_bone_parents[b]) if all_bone_parents[b] is not None else -1) for b in bone_subset],
            dtype=torch.long)
        subset_id = [all_bone_names.index(b) for b in bone_subset]
        self.bone_matrix = self.all_bone_matrix[subset_id]

        # joint constraints
        joint_constraint_id, joint_constraint_value = get_constraints(all_bone_names, all_bone_matrix_world_rest,
                                                                      optimizable_bones, skeleton_remap)
        self.joint_contraint_id = joint_constraint_id
        self.joint_constraints_min, self.joint_constraints_max = joint_constraint_value[:, :,
                                                                 0], joint_constraint_value[:, :, 1]

        # align location
        self.align_location_kpts, self.align_location_bones = get_align_location(optimizable_bones, skeleton_remap)

        # align scale
        self.align_scale_pairs_kpt, self.align_scale_pairs_bone = get_align_scale(all_bone_names, skeleton_remap)
        rest_joints = torch.from_numpy(all_bone_matrix_world_rest)[:, :3, 3]
        self.align_scale_pairs_length = torch.norm(
            rest_joints[self.align_scale_pairs_bone[:, 0]] - rest_joints[self.align_scale_pairs_bone[:, 1]], dim=-1)

        # optimization hyperparameters
        self.lr = kwargs.get('lr', 1.0)
        self.max_iter = kwargs.get('max_iter', 24)
        self.tolerance_change = kwargs.get('tolerance_change', 1e-6)
        self.tolerance_grad = kwargs.get('tolerance_grad', 1e-4)
        self.joint_constraint_loss_weight = kwargs.get('joint_constraint_loss_weight', 1)
        self.pose_reg_loss_weight = kwargs.get('pose_reg_loss_weight', 0.1)
        self.smooth_range = kwargs.get('smooth_range', 0.3)

        # optimizable bone euler angles
        self.optimizable_bones = optimizable_bones
        self.gather_id = torch.tensor(
            [(optimizable_bones.index(b) + 1 if b in optimizable_bones else 0) for b in bone_subset], dtype=torch.long)[
                         :, None, None].repeat(1, 4, 4)
        self.all_gather_id = torch.tensor(
            [(optimizable_bones.index(b) + 1 if b in optimizable_bones else 0) for b in all_bone_names],
            dtype=torch.long)[:, None, None].repeat(1, 4, 4)
        self.optim_bone_euler = torch.zeros((len(optimizable_bones), 3), requires_grad=True)

        # smoothness
        self.euler_angle_history, self.location_history = [], []
        self.align_scale = torch.tensor(0.0)

    def fit(self, kpts: torch.Tensor, valid: torch.Tensor, frame_t: float):
        optimizer = torch.optim.LBFGS(
            [self.optim_bone_euler],
            line_search_fn='strong_wolfe',
            lr=self.lr,
            max_iter=100 if len(self.euler_angle_history) == 0 else self.max_iter,
            tolerance_change=self.tolerance_change,
            tolerance_grad=self.tolerance_grad
        )

        pair_valid = valid[self.kpt_pairs_a] & valid[self.kpt_pairs_b]
        kpt_pairs_a, kpt_pairs_b = self.kpt_pairs_a[pair_valid], self.kpt_pairs_b[pair_valid]
        joint_pairs_a, joint_pairs_b = self.joint_pairs_a[pair_valid], self.joint_pairs_b[pair_valid]

        kpt_dir = kpts[kpt_pairs_a] - kpts[kpt_pairs_b]
        kpt_pairs_length = torch.norm(kpts[self.align_scale_pairs_kpt[:, 0]] - kpts[self.align_scale_pairs_kpt[:, 1]],
                                      dim=-1)
        align_scale = (kpt_pairs_length / self.align_scale_pairs_length).mean()
        if align_scale > 0:
            self.align_scale = align_scale
            kpt_dir = kpt_dir / self.align_scale

        def _loss_closure():
            optimizer.zero_grad()
            optim_matrix_basis = euler_angle_to_matrix(self.optim_bone_euler, 'YXZ')
            matrix_basis = torch.gather(torch.cat([torch.eye(4).unsqueeze(0), optim_matrix_basis]), dim=0,
                                        index=self.gather_id)
            matrix_world = eval_matrix_world(self.bone_parents_id, self.bone_matrix, matrix_basis)
            joints = matrix_world[:, :3, 3]
            joint_dir = joints[joint_pairs_a] - joints[joint_pairs_b]
            dir_loss = F.mse_loss(kpt_dir, joint_dir)
            joint_prior_loss = barrier(self.optim_bone_euler[self.joint_contraint_id], self.joint_constraints_min,
                                       self.joint_constraints_max).mean()
            pose_reg_loss = self.optim_bone_euler.square().mean()
            loss = dir_loss + self.pose_reg_loss_weight * pose_reg_loss + self.joint_constraint_loss_weight * joint_prior_loss
            loss.backward()
            return loss

        if len(kpt_dir) > 0:
            optimizer.step(_loss_closure)

        optim_matrix_basis = euler_angle_to_matrix(self.optim_bone_euler, 'YXZ')
        matrix_basis = torch.gather(torch.cat([torch.eye(4).unsqueeze(0), optim_matrix_basis]), dim=0,
                                    index=self.all_gather_id)
        matrix_world = torch.tensor([align_scale, align_scale, align_scale, 1.])[None, :, None] * eval_matrix_world(
            self.bone_parents_id, self.bone_matrix, matrix_basis)
        location = kpts[self.align_location_kpts].mean(dim=0) - matrix_world[self.align_location_bones, :3, 3].mean(
            dim=0)
        opt = self.optim_bone_euler.detach()
        self.euler_angle_history.append((opt, frame_t))
        self.location_history.append((location, frame_t))

    def get_smoothed_bone_euler(self, query_t: float) -> torch.Tensor:
        input_euler, input_t = zip(
            *((e, t) for e, t in self.euler_angle_history if abs(t - query_t) < self.smooth_range))
        if len(input_t) <= 2:
            joints_smoothed = input_euler[-1]
        else:
            joints_smoothed = mls_smooth(input_t, input_euler, query_t, self.smooth_range)
        return joints_smoothed

    def get_scale(self) -> float:
        return self.align_scale

    def get_smoothed_location(self, query_t: float) -> torch.Tensor:
        input_location, input_t = zip(
            *((e, t) for e, t in self.location_history if abs(t - query_t) < self.smooth_range))
        if len(input_t) <= 2:
            location_smoothed = input_location[-1]
        else:
            location_smoothed = mls_smooth(input_t, input_location, query_t, self.smooth_range)
        return location_smoothed

    def eval_bone_matrix_world(self, bone_euler: torch.Tensor, location: torch.Tensor, scale: float) -> torch.Tensor:
        optim_matrix_basis = euler_angle_to_matrix(bone_euler, 'YXZ')
        matrix_basis = torch.gather(torch.cat([torch.eye(4).unsqueeze(0), optim_matrix_basis]), dim=0,
                                    index=self.all_gather_id)
        matrix_world = eval_matrix_world(self.all_bone_parents_id, self.all_bone_matrix, matrix_basis)

        # set scale and location
        matrix_world = torch.tensor([scale, scale, scale, 1.])[None, :, None] * matrix_world
        matrix_world[:, :3, 3] += location
        return matrix_world


def update_eval_matrix(bone_parents: torch.Tensor, bone_matrix_world: torch.Tensor,
                       updated_bones = None):
    bone_matrix_world_updated = bone_matrix_world.clone()
    for i, matrix in updated_bones.items():
        if matrix.shape == (3, 3):
            bone_matrix_world_updated[i, :3, :3] = matrix
        elif matrix.shape == (4, 4):
            bone_matrix_world_updated[i] = matrix
        else:
            raise ValueError('Invalid matrix shape')
    to_update = set(updated_bones.keys())
    for i in range(bone_matrix_world.shape[0]):
        if bone_parents[i].item() in to_update:
            bone_matrix_world_updated[i] = bone_matrix_world_updated[bone_parents[i]] @ (
                        bone_matrix_world[bone_parents[i]].inverse() @ bone_matrix_world[i])
    return bone_matrix_world_updated