"""Shared rollout utilities for Im2Flow2Act flow-conditioned policies.

Use run_flow_diffusion_agent_gt.py for GT zarr flow and
run_flow_diffusion_agent_generated.py for AnimateFlow-generated flow.
"""

import argparse
import collections
import json
import os
import sys

# Robosuite 1.4 can fail during import when numba cache files are not writable
# or when installed package files have no cache locator. Disable JIT for this
# evaluation script rather than failing before the environment is created.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import h5py
import imageio
import numpy as np
import torch
from tqdm import tqdm

# ── robomimic ─────────────────────────────────────────────────────────────────
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

# ── mimiclabs env registration ────────────────────────────────────────────────
# MimicLabs environments (e.g. MimicLabs_Lab2_Tabletop_Manipulation) must be
# imported before any robosuite.make() call so they are in the registry.
# We inject source paths so the im2flow2act env needs no pip-install of these.
try:
    for _p in [
        "/data/chuye/Documents/LIBERO",
        "/data/chuye/Documents/mimicgen",
        "/data/chuye/Documents/mimiclabs-dev",
    ]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from mimiclabs.mimiclabs.envs import *  # noqa: F401,F403  registers MimicLabs envs
except Exception as _e:
    print(f"[WARNING] Could not register MimicLabs envs: {_e}")

# ── im2flow2act ───────────────────────────────────────────────────────────────
try:
    from im2flow2act.common.utility.diffusion import load_flow_diffusion_model
    from im2flow2act.common.utility.file import read_pickle
    from im2flow2act.diffusion_policy.dataloader.diffusion_bc_dataset import (
        normalize_data,
        unnormalize_data,
    )
    from im2flow2act.diffusion_policy.dataloader.diffusion_flow_bc_dataset import (
        DiffusionFlowBCCloseLoopDataset,
        process_image,
    )
    from im2flow2act.tapnet.online_point_tracking import (
        build as tapir_build,
        construct_initial_features_and_state,
        inference as tapir_inference,
    )
    from im2flow2act.tapnet.utility.viz import (
        draw_point_tracking_sequence,
        viz_point_tracking_flow,
    )
    from im2flow2act.common.utility.viz import save_to_gif
    from im2flow2act.common.utility.arr import (
        complete_random_sampling,
        uniform_sampling,
    )
    from im2flow2act.tapnet.utility.utility import max_distance_moved
except ImportError as e:
    sys.exit(
        f"Cannot import im2flow2act: {e}\n"
        "Make sure the conda environment is activated and PYTHONPATH includes "
        "the im2Flow2Act repo root:\n"
        "  export PYTHONPATH=$PYTHONPATH:/data/chuye/Documents/im2Flow2Act"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_proprioception(obs):
    """Return 7-D proprioception: eef_pos (3) + eef_quat (4)."""
    return np.concatenate([
        obs["robot0_eef_pos"].astype(np.float32),
        obs["robot0_eef_quat"].astype(np.float32),
    ])


def get_rgb(obs, camera_name="agentview"):
    """Return (H, W, 3) uint8 RGB image."""
    return obs[f"{camera_name}_image"]


def get_depth(obs, camera_name="agentview", sim=None):
    """Return (H, W) float32 depth image in metres.

    Robosuite stores raw MuJoCo z-buffer values in [0, 1]. When a sim handle
    is provided we convert to real depth (metres) using its zfar/znear; this
    matches the metric-depth convention used by the zarr training data and by
    DiffusionFlowBCCloseLoopDataset.get_point_cloud.
    """
    depth = obs[f"{camera_name}_depth"]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    if sim is not None:
        # Only convert if the raw values look normalised (robosuite default).
        if depth.max() <= 1.0 + 1e-5:
            from robosuite.utils.camera_utils import get_real_depth_map
            depth = get_real_depth_map(sim, depth)
    return depth


def get_robosuite_sim(env):
    """Dig through robomimic wrappers to retrieve the underlying MjSim."""
    base = env
    seen = set()
    while id(base) not in seen:
        seen.add(id(base))
        if hasattr(base, "sim"):
            return base.sim
        for attr in ("env", "base_env", "_env"):
            if hasattr(base, attr):
                base = getattr(base, attr)
                break
        else:
            break
    return getattr(base, "sim", None)


def compute_sim_camera_calibration(env, camera_name, height, width):
    """Return (K 3×3, cam_pose 4×4) for `camera_name` in the mimiclabs sim.

    cam_pose is camera-to-world in the OpenCV convention (x right, y down,
    z forward) — matching what DiffusionFlowBCCloseLoopDataset.get_point_cloud
    / transform_pointcloud expect.
    """
    from robosuite.utils.camera_utils import (
        get_camera_intrinsic_matrix,
        get_camera_extrinsic_matrix,
    )
    sim = get_robosuite_sim(env)
    if sim is None:
        raise RuntimeError("Could not locate MjSim in env to build calibration.")
    K = get_camera_intrinsic_matrix(
        sim=sim, camera_name=camera_name,
        camera_height=height, camera_width=width,
    )
    cam_pose = get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name)
    return K.astype(np.float32), cam_pose.astype(np.float32)


def process_env_rgb(rgb, resize_shape):
    """Resize → process_image → float tensor (3, H, W)."""
    resized = cv2.resize(rgb, policy_wh(resize_shape))
    return process_image(resized)


def point_tracking_hw(point_tracking_img_size):
    """Return TAPIR frame size as (height, width)."""
    return int(point_tracking_img_size[0]), int(point_tracking_img_size[1])


def policy_wh(resize_shape):
    """Return policy image size as the (width, height) tuple cv2 expects."""
    return int(resize_shape[0]), int(resize_shape[1])


def norm_flow_to_pixels(flow_norm, width, height):
    """Convert normalized (x, y, visibility) flow to clipped pixel indices."""
    flow = flow_norm.copy()
    flow[..., 0] = flow[..., 0] * width
    flow[..., 1] = flow[..., 1] * height
    flow = flow.astype(np.int32)
    return clip_flow_xy(flow, width, height)


def clip_flow_xy(flow, width, height):
    """Clip only x/y channels; keep visibility unchanged."""
    flow[..., 0] = np.clip(flow[..., 0], 0, width - 1)
    flow[..., 1] = np.clip(flow[..., 1], 0, height - 1)
    return flow


def validate_policy_inputs(episode_flow_plan, initial_proprioception, stats, num_points):
    if episode_flow_plan.ndim != 3 or episode_flow_plan.shape[-1] != 3:
        raise ValueError(
            "episode_flow_plan must have shape (T, N, 3); got "
            f"{episode_flow_plan.shape}."
        )
    if episode_flow_plan.shape[1] != num_points:
        raise ValueError(
            f"Policy expects {num_points} flow points, got "
            f"{episode_flow_plan.shape[1]}."
        )
    prop_dim = stats["proprioception"]["min"].shape[0]
    if initial_proprioception.shape[-1] != prop_dim:
        raise ValueError(
            f"Policy expects proprioception dim {prop_dim}, got "
            f"{initial_proprioception.shape[-1]}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_plan_gif(rgb_initial, plan_flow_norm, save_path, draw_line=True):
    """Overlay the generated / GT flow plan on the initial RGB frame as a GIF.

    Parameters
    ----------
    rgb_initial    : (H, W, 3) uint8 — canvas to draw on (raw camera resolution).
    plan_flow_norm : (T, N, 3) float32 in [0,1] — plan used by the policy.
    """
    from einops import rearrange

    viz = plan_flow_norm.copy()
    viz = rearrange(viz, "T N C -> N T C")  # (N, T, 3)
    img_h, img_w = rgb_initial.shape[:2]
    viz[..., 0] = viz[..., 0] * img_w
    viz[..., 1] = viz[..., 1] * img_h

    frames = []
    for j in range(1, viz.shape[1] + 1):
        frame = draw_point_tracking_sequence(
            rgb_initial.copy(),
            viz[:, :j],
            draw_line=draw_line,
            thickness=2,
            radius=3,
        )
        frames.append(frame)
    save_to_gif(frames, save_path)


def save_plan_scatter(rgb_initial, plan_flow_norm, save_path, num_keyframes=4):
    """Save a single PNG showing the plan trajectory (start + keyframes + end)."""
    import matplotlib.pyplot as plt

    img_h, img_w = rgb_initial.shape[:2]
    T = plan_flow_norm.shape[0]
    keyframes = [0] + [int(round(T * i / (num_keyframes - 1))) for i in range(1, num_keyframes - 1)] + [T - 1]
    keyframes = sorted(set(np.clip(keyframes, 0, T - 1).tolist()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb_initial)
    colors = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]
    for i, t in enumerate(keyframes):
        xs = plan_flow_norm[t, :, 0] * img_w
        ys = plan_flow_norm[t, :, 1] * img_h
        ax.scatter(xs, ys, s=6, c=colors[i % len(colors)], label=f"t={t}")
    ax.set_title(f"Flow plan ({T} frames, {plan_flow_norm.shape[1]} pts)")
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def overlay_plan_on_rgb(rgb, plan_flow_pixel, t_index, radius=3):
    """Draw plan scatter at time-step t on a copy of rgb. plan_flow_pixel: (T, N, 2) int."""
    frame = rgb.copy()
    import matplotlib.cm as cm
    color_map = cm.get_cmap("jet")
    N = plan_flow_pixel.shape[1]
    for n in range(N):
        c = np.array(color_map(n / max(1, N - 1))[:3]) * 255
        c = (int(c[0]), int(c[1]), int(c[2]))
        x, y = plan_flow_pixel[t_index, n]
        cv2.circle(frame, (int(x), int(y)), radius, c, -1, lineType=16)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# GT flow from zarr
# ─────────────────────────────────────────────────────────────────────────────

def load_gt_flow_plan(zarr_root, episode_idx, num_points, num_frames, point_tracking_img_size):
    """
    Load and process the ground-truth SAM point-tracking flow for one episode
    from the zarr training data.

    Returns
    -------
    plan : (num_frames, num_points, 3) float32 in [0, 1]
    """
    ep = zarr_root[f"episode_{episode_idx}"]

    # point_tracking_sequence stored as (N, T, 3); transpose to (T, N, 3)
    pt_seq = np.transpose(ep["sam_point_tracking_sequence"][:], [1, 0, 2])  # (T, N, 3)
    moving_mask = ep["sam_moving_mask"][:]   # (N,)
    try:
        robot_mask = ep["robot_mask"][:]     # (N,)
    except Exception:
        robot_mask = np.ones_like(moving_mask, dtype=bool)

    # sample 128 object points (same logic as training)
    pt_seq = DiffusionFlowBCCloseLoopDataset.process_point_tracking_data(
        episode_point_tracking=pt_seq,
        episode_moving_mask=moving_mask,
        num_points_to_sample=num_points,
        point_tracking_img_size=point_tracking_img_size,
        robot_mask=robot_mask,
        equal_sampling=False,
        object_sampling=True,
        herustic_filter=[],
        ignore_robot_mask=False,
    )  # (T, num_points, 3) normalised to [0,1]

    episode_length = pt_seq.shape[0]

    # uniform-sample num_frames along time
    plan, _ = uniform_sampling(pt_seq[:episode_length], num_frames, return_indices=True)
    return plan.astype(np.float32)   # (num_frames, num_points, 3)


# ─────────────────────────────────────────────────────────────────────────────
# AnimateFlow-generated flow plan
# ─────────────────────────────────────────────────────────────────────────────

def load_flow_model(flow_model_path, flow_ckpt, vae_path_override=None):
    """Load the AnimateFlow pipeline (VAE + text-enc + UNet3D + LoRA).

    Args:
        vae_path_override: If set, overrides the vae_pretrained_model_path read
            from the experiment config. Useful when the config references a VAE
            training run that no longer exists on disk.
    """
    from im2flow2act.flow_generation.AnimationFlowPipeline import AnimationFlowPipeline
    from im2flow2act.flow_generation.inference import load_model as _load

    class _FakeCfg:
        model_path = flow_model_path
        model_ckpt = flow_ckpt
        vae_path = vae_path_override  # None means "use config value"

    vae, text_encoder, tokenizer, noise_scheduler, model = _load(_FakeCfg())
    pipeline = AnimationFlowPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        model=model,
        scheduler=noise_scheduler,
    )
    return pipeline


def generate_flow_plan(
    pipeline,
    initial_rgb,
    initial_depth,
    text,
    num_points,
    num_frames,
    grid_size,
    point_tracking_img_size,
    resize_shape,
    num_inference_steps=25,
    guidance_scale=8.0,
    workspace_depth=1.2,
    moving_threshold=20.0,
    filters=("mv",),
    debug_out=None,
):
    """
    Run AnimateFlow on initial_rgb and produce the flow plan used by the policy.

    Parameters
    ----------
    initial_rgb        : (H, W, 3) uint8  — raw camera image
    initial_depth      : (H, W) float32   — depth in metres
    text               : str              — task description
    num_points         : int              — points to track (e.g. 128)
    num_frames         : int              — plan time-steps (e.g. 32)
    grid_size          : int              — AnimateFlow grid (e.g. 32 → 32×32 grid)
    point_tracking_img_size : (H, W)      — TAPIR image resolution (e.g. (256,256))
    resize_shape       : (W, H)           — policy image resolution (e.g. (224,224))

    Returns
    -------
    plan : (num_frames, num_points, 3)  float32 in [0, 1]
    """
    from im2flow2act.flow_generation.inference import inference as _flow_inference

    resize_w, resize_h = policy_wh(resize_shape)
    pt_h, pt_w = point_tracking_hw(point_tracking_img_size)

    # ── 1. Build uniform grid of initial keypoints in resize_shape space ──────
    xs = np.linspace(0, resize_w - 1, grid_size)
    ys = np.linspace(0, resize_h - 1, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    # point_uv: (N, 2) in (x, y) order within the resize_shape frame
    point_uv = np.stack([xx.flatten(), yy.flatten()], axis=1).astype(np.float32)

    rgb_for_flow = cv2.resize(initial_rgb, (resize_w, resize_h))

    # ── 2. Run AnimateFlow inference ──────────────────────────────────────────
    with torch.no_grad():
        flows = _flow_inference(
            pipeline=pipeline,
            global_image=rgb_for_flow,
            point_uv=point_uv,
            text=text,
            height=grid_size,
            width=grid_size,
            video_length=num_frames,
            diff_flow=False,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # (N=grid_size², T=num_frames, 3) in [0,1]

    # ── 3. Rescale to point_tracking_img_size ────────────────────────────────
    #  AnimateFlow outputs normalised [0,1]; rescale to pixel space of TAPIR.
    episode_pt = flows.copy()  # (N, T, 3)
    episode_pt[:, :, :2] *= np.array([pt_w, pt_h])

    # ── 4. Downsample the spatial grid (32×32 → 16×16) ───────────────────────
    episode_pt = episode_pt.reshape(grid_size, grid_size, num_frames, 3)
    episode_pt = episode_pt[::2, ::2]                          # (16,16,T,3)
    episode_pt = episode_pt.reshape(-1, num_frames, 3)         # (256, T, 3)

    # fix visibility channel clipping
    episode_pt[:, :, 2] = np.where(episode_pt[:, :, 2] > 0.99, 1.0, episode_pt[:, :, 2])

    # Keep a snapshot of the full pre-filter flow for debugging.
    if debug_out is not None:
        pre_filter = episode_pt.copy()
        pre_filter_norm = pre_filter.copy()
        pre_filter_norm[..., 0] /= float(pt_w)
        pre_filter_norm[..., 1] /= float(pt_h)
        pre_filter_norm = np.clip(pre_filter_norm, 0.0, 1.0)
        debug_out["pre_filter"] = pre_filter_norm.transpose(1, 0, 2).astype(np.float32)

    # ── 5. Motion filter (mirrors FlowGenerator.apply_filter "mv") ───────────
    # This is the critical step: without it, 200+ static background points
    # drown the handful that actually move, and the policy's flow encoder
    # collapses to the same embedding regardless of the AnimateFlow output.
    if "mv" in filters:
        max_dists = max_distance_moved(episode_pt, t_threshold=-1)   # (N,)
        moving_mask = max_dists > moving_threshold
        if moving_mask.sum() < max(num_points // 4, 8):
            # Fall back to top-k by motion if threshold is too strict for this scene.
            k = max(num_points, int(0.5 * len(episode_pt)))
            topk_idx = np.argsort(-max_dists)[:k]
            moving_mask = np.zeros_like(max_dists, dtype=bool)
            moving_mask[topk_idx] = True
            print(
                f"  [mv-filter] threshold={moving_threshold} too strict — "
                f"using top-{k} by motion instead."
            )
        episode_pt = episode_pt[moving_mask]
        print(f"  [mv-filter] kept {len(episode_pt)}/{moving_mask.size} points")

        if debug_out is not None:
            post_mv = episode_pt.copy()
            post_mv[..., 0] /= float(pt_w)
            post_mv[..., 1] /= float(pt_h)
            post_mv = np.clip(post_mv, 0.0, 1.0)
            debug_out["post_mv"] = post_mv.transpose(1, 0, 2).astype(np.float32)

    # ── 6. Filter by valid (non-zero) and in-workspace depth ─────────────────
    depth_h, depth_w = initial_depth.shape[:2]
    init_flow_xy = episode_pt[:, 0, :2]  # (N_pts, 2) in [0,256) (x,y)

    cam_flow = init_flow_xy.copy()
    cam_flow[:, 0] = (cam_flow[:, 0] / pt_w * depth_w).astype(int)
    cam_flow[:, 1] = (cam_flow[:, 1] / pt_h * depth_h).astype(int)
    cam_flow = np.clip(
        cam_flow.astype(np.int32),
        a_min=0,
        a_max=np.array([depth_w - 1, depth_h - 1]),
    )

    depth_vals = initial_depth[cam_flow[:, 1], cam_flow[:, 0]]
    valid = (depth_vals > 0) & (depth_vals < workspace_depth)
    valid_idx = np.where(valid)[0]

    if len(valid_idx) == 0:
        print("  [WARN] No valid-depth points — using all generated points.")
        valid_idx = np.arange(len(episode_pt))

    # ── 7. Randomly sample num_points ────────────────────────────────────────
    plan_flow = complete_random_sampling(episode_pt[valid_idx], num_points)
    # plan_flow: (num_points, num_frames, 3) in point_tracking_img_size scale

    # ── 8. Normalise to [0,1] ─────────────────────────────────────────────────
    plan_flow = plan_flow.copy()
    plan_flow[..., 0] /= pt_w
    plan_flow[..., 1] /= pt_h
    plan_flow = np.clip(plan_flow, 0.0, 1.0).astype(np.float32)

    # ── 9. Transpose to (T, N, 3) ────────────────────────────────────────────
    plan_flow = plan_flow.transpose(1, 0, 2)  # (num_frames, num_points, 3)
    return plan_flow


# ─────────────────────────────────────────────────────────────────────────────
# Single-episode rollout
# ─────────────────────────────────────────────────────────────────────────────

def rollout_episode(
    model,
    noise_scheduler,
    stats,
    env,
    episode_flow_plan,       # (T, N, 3) float32 in [0,1]  — the flow plan
    initial_rgb,             # (H, W, 3) uint8
    initial_depth,           # (H, W) float32 metres
    initial_proprioception,  # (7,) float32
    camera_intrinsic,        # (3, 3)
    camera_pose_matrix,      # (4, 4)
    # --- policy hyperparams (from training config) ---
    obs_horizon,
    action_horizon,
    pred_horizon,
    action_dim,
    target_flow_horizon,
    num_inference_steps,
    num_points,
    point_tracking_img_size,
    resize_shape,
    normalize_pointcloud,
    camera_name,
    horizon,
    max_policy_steps=None,
    video_writer=None,
    video_skip=5,
    result_save_path=None,
    episode_idx=0,
):
    """
    Run one episode and return a stats dict with keys:
        Success_Rate, Return, Horizon
    """
    device = next(model.parameters()).device
    resize_w, resize_h = policy_wh(resize_shape)
    pt_h, pt_w = point_tracking_hw(point_tracking_img_size)
    if max_policy_steps is None:
        max_policy_steps = max(1, (horizon + action_horizon - 1) // action_horizon)
    validate_policy_inputs(
        episode_flow_plan=episode_flow_plan,
        initial_proprioception=initial_proprioception,
        stats=stats,
        num_points=num_points,
    )

    # ── Flow plan on GPU ──────────────────────────────────────────────────────
    ep_flow_plan_t = torch.from_numpy(episode_flow_plan).unsqueeze(0).to(device)  # (1,T,N,3)

    # ── Initial flow: t=0 row of the plan, back to pixel space ───────────────
    initial_flow_norm = episode_flow_plan[0].copy()   # (N, 3) in [0,1]
    initial_flow = norm_flow_to_pixels(initial_flow_norm, resize_w, resize_h)
    initial_flow_depth = initial_flow.copy()          # used for point cloud (under resize_shape scale)
    initial_flow_t = torch.tensor(initial_flow).unsqueeze(0).to(device)  # (1, N, 3)

    # ── Point cloud from initial depth ────────────────────────────────────────
    point_cloud, _ = DiffusionFlowBCCloseLoopDataset.get_point_cloud(
        depth_img=initial_depth,
        color_img=np.zeros((resize_h, resize_w, 3), dtype=np.uint8),
        cam_intr=camera_intrinsic,
        cam_pose=camera_pose_matrix,
        flow=initial_flow_depth.copy(),
    )
    point_cloud = point_cloud.astype(np.float32)
    if normalize_pointcloud:
        point_cloud = normalize_data(point_cloud, stats["point_cloud"])
    point_cloud_t = torch.tensor(point_cloud).unsqueeze(0).to(device)  # (1, N, 3)

    # ── TAPIR tracker initialisation ──────────────────────────────────────────
    online_predict_apply, online_init_apply, _, _ = tapir_build(
        num_points=num_points,
        img_size=point_tracking_img_size,
    )
    # query_points in (t, y, x) order for TAPIR
    query_points = np.array([
        [0, p[1] * pt_h, p[0] * pt_w]
        for p in initial_flow_norm
    ], dtype=np.float32)

    pt_frame = cv2.resize(initial_rgb, (pt_w, pt_h))
    query_features, causal_state = construct_initial_features_and_state(
        query_points=query_points,
        initial_frame=pt_frame,
        online_init_apply=online_init_apply,
    )
    current_flow_raw, causal_state = tapir_inference(
        query_features=query_features,
        causal_state=causal_state,
        current_frame=pt_frame,
        online_predict_apply=online_predict_apply,
    )   # (N, 1, 3)
    current_flow_raw = clip_flow_xy(current_flow_raw, pt_w, pt_h)

    # ── TAPIR logging (for visualization at the end of the episode) ───────────
    online_pt_track = [current_flow_raw.astype(np.int32)]   # list of (N,1,3)
    pt_track_frames = [pt_frame.copy()]                     # list of (H,W,3) at TAPIR res

    # Pre-compute the plan in pixel space (resize_shape) for overlay drawing.
    plan_pixel = norm_flow_to_pixels(episode_flow_plan, resize_w, resize_h)

    def _scale_flow_to_policy(cf_raw):
        """(N,1,3) in TAPIR-img space → (1,N,3) tensor in resize_shape space."""
        cf = cf_raw[:, 0, :].copy()
        cf[:, 0] = (cf[:, 0] / pt_w * resize_w).astype(np.int32)
        cf[:, 1] = (cf[:, 1] / pt_h * resize_h).astype(np.int32)
        cf = clip_flow_xy(cf.astype(np.int32), resize_w, resize_h)
        return torch.tensor(cf).unsqueeze(0).to(device)  # (1, N, 3)

    current_flow_t = _scale_flow_to_policy(current_flow_raw)

    # ── Observation queues ────────────────────────────────────────────────────
    initial_rgb_proc = process_env_rgb(initial_rgb, resize_shape)  # (3, H, W) float
    initial_frame_t = initial_rgb_proc.unsqueeze(0).to(device)     # (1, 3, H, W)

    img_deque = collections.deque(
        [initial_rgb_proc] * obs_horizon, maxlen=obs_horizon
    )
    prop_norm = normalize_data(
        initial_proprioception.reshape(1, -1), stats=stats["proprioception"]
    )
    prop_deque = collections.deque([prop_norm] * obs_horizon, maxlen=obs_horizon)

    # dummy target flow (not used at eval time; the model ignores it)
    target_flow_t = torch.zeros((1, target_flow_horizon), device=device)

    # ── Episode loop ──────────────────────────────────────────────────────────
    total_reward = 0.0
    success = False
    terminated = False
    step = 0
    video_count = 0

    policy_step = 0
    while step < horizon and policy_step < max_policy_steps:
        policy_step += 1
        prop_seq = (
            torch.from_numpy(np.concatenate(list(prop_deque), axis=0))
            .unsqueeze(0).to(device)
        )
        visual_seq = torch.stack(list(img_deque)).unsqueeze(0).to(device)

        # diffusion denoising
        noise = torch.randn(1, pred_horizon, action_dim, device=device)
        noisy = noise
        noise_scheduler.set_timesteps(num_inference_steps)
        for t in noise_scheduler.timesteps:
            with torch.no_grad():
                model_out = model(
                    noisy,
                    t.unsqueeze(0).to(device),
                    initial_frame_t,
                    visual_seq,
                    None,           # second camera (unused)
                    prop_seq,
                    ep_flow_plan_t,
                    initial_flow_t,
                    current_flow_t,
                    target_flow_t,
                    point_cloud_t,
                )
                noisy_residual = model_out[0]
            noisy = noise_scheduler.step(noisy_residual, t, noisy).prev_sample

        naction = noisy.detach().cpu().numpy()[0]  # (pred_horizon, action_dim)
        action_pred = unnormalize_data(naction, stats=stats["action"])
        actions = action_pred[obs_horizon - 1 : obs_horizon - 1 + action_horizon]
        env_action_dim = getattr(env, "action_dimension", actions.shape[-1])
        if actions.shape[-1] != env_action_dim:
            raise ValueError(
                f"Policy produced action dim {actions.shape[-1]}, but env "
                f"expects {env_action_dim}."
            )

        # execute action chunk
        for k in range(len(actions)):
            if step >= horizon:
                break

            act = actions[k]
            obs, reward, done, info = env.step(act)
            total_reward += reward
            step += 1

            # video (with overlay: plan target at this step + tracked points)
            if video_writer is not None and video_count % video_skip == 0:
                raw_frame = get_rgb(obs, camera_name)
                # Draw plan's current-step target and TAPIR-tracked points onto a
                # resize_shape canvas, then resize to raw for the video.
                canvas = cv2.resize(raw_frame, (resize_w, resize_h))
                plan_t = min(
                    int(step / max(1, horizon) * episode_flow_plan.shape[0]),
                    episode_flow_plan.shape[0] - 1,
                )
                # plan targets (yellow-ish) — uses plan_pixel at step t
                for n in range(plan_pixel.shape[1]):
                    x, y = plan_pixel[plan_t, n, :2]
                    cv2.circle(canvas, (int(x), int(y)), 2, (0, 255, 255), -1, lineType=16)
                # tracked points (red)
                tracked = current_flow_raw[:, 0, :2].copy()
                tracked[:, 0] = tracked[:, 0] / pt_w * resize_w
                tracked[:, 1] = tracked[:, 1] / pt_h * resize_h
                for n in range(tracked.shape[0]):
                    x, y = tracked[n]
                    cv2.circle(canvas, (int(x), int(y)), 2, (255, 0, 0), -1, lineType=16)
                canvas = cv2.resize(canvas, (raw_frame.shape[1], raw_frame.shape[0]))
                video_writer.append_data(canvas)
            video_count += 1

            # update observations
            rgb_new = get_rgb(obs, camera_name)
            img_deque.append(process_env_rgb(rgb_new, resize_shape))
            prop_deque.append(normalize_data(
                get_proprioception(obs).reshape(1, -1),
                stats=stats["proprioception"],
            ))

            # TAPIR tracking
            pt_frame = cv2.resize(rgb_new, (pt_w, pt_h))
            current_flow_raw, causal_state = tapir_inference(
                query_features=query_features,
                causal_state=causal_state,
                current_frame=pt_frame,
                online_predict_apply=online_predict_apply,
            )
            current_flow_raw = clip_flow_xy(current_flow_raw, pt_w, pt_h)
            online_pt_track.append(current_flow_raw.astype(np.int32))
            pt_track_frames.append(pt_frame.copy())
            current_flow_t = _scale_flow_to_policy(current_flow_raw)

            # success / terminal check
            info_success = info.get("is_success", {}) if isinstance(info, dict) else {}
            task_success = (
                info_success.get("task")
                if isinstance(info_success, dict) and "task" in info_success
                else env.is_success()["task"]
            )
            if task_success:
                success = True
                break
            if done:
                terminated = True
                break

        if success or terminated or step >= horizon:
            break

    # ── dump TAPIR online-tracking GIF for this episode ───────────────────────
    if result_save_path is not None and len(online_pt_track) > 1:
        try:
            online_pt = np.concatenate(online_pt_track, axis=1)   # (N, T, 3) in TAPIR-res
            frames_np = np.array(pt_track_frames)                 # (T, H, W, 3)
            viz_point_tracking_flow(
                frames_np,
                online_pt,
                point_per_key=len(online_pt),
                output_path=os.path.join(
                    result_save_path, f"episode_{episode_idx:03d}_tapir_tracking.gif"
                ),
            )
            np.save(
                os.path.join(
                    result_save_path,
                    f"episode_{episode_idx:03d}_tapir_tracking.npy",
                ),
                online_pt,
            )
        except Exception as _viz_e:
            print(f"  [WARN] Failed to save TAPIR tracking viz: {_viz_e}")

    return {
        "Success_Rate": float(success),
        "Return": total_reward,
        "Horizon": step,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser(mode=None):
    """Build a CLI parser.

    mode=None keeps the legacy combined CLI. mode="gt" and mode="generated"
    are used by the split entrypoint scripts.
    """
    if mode not in (None, "gt", "generated"):
        raise ValueError(f"Unknown mode: {mode}")

    p = argparse.ArgumentParser(description="Evaluate im2flow2act policy.")
    p.add_argument("--policy-path", required=True,
                   help="Path to the im2flow2act policy training directory.")
    p.add_argument("--policy-ckpt", type=int, required=True,
                   help="Policy checkpoint epoch.")
    p.add_argument("--hdf5", required=True,
                   help="Robomimic demo_im.hdf5 used to recreate the env.")
    p.add_argument("--camera-name", default="agentview",
                   help="Robosuite camera name (default: agentview).")

    zarr_required = mode == "gt"
    p.add_argument("--zarr", required=zarr_required, default=None,
                   help="Zarr store for GT flows, or calibration pkls when "
                        "--calibration-source zarr is used.")

    if mode is None:
        p.add_argument("--use-gt-flow", action="store_true",
                       help="Use GT zarr flow instead of generated flow.")
    else:
        p.set_defaults(use_gt_flow=(mode == "gt"))
        p.add_argument("--use-gt-flow", action="store_true",
                       help=argparse.SUPPRESS)

    if mode in (None, "generated"):
        p.add_argument("--flow-model-path", required=(mode == "generated"),
                       default=None, help="AnimateFlow training directory.")
        p.add_argument("--flow-ckpt", type=int, required=(mode == "generated"),
                       default=None, help="AnimateFlow checkpoint epoch.")
        p.add_argument("--vae-path", default=None,
                       help="Optional VAE checkpoint override.")
        p.add_argument("--task-description", default="hang mug on tree",
                       help="Task text fed to AnimateFlow.")
        p.add_argument("--flow-inference-steps", type=int, default=25,
                       help="DDIM steps for AnimateFlow inference.")
        p.add_argument("--guidance-scale", type=float, default=8.0,
                       help="Classifier-free guidance scale.")
        p.add_argument("--moving-threshold", type=float, default=20.0,
                       help="Minimum TAPIR-pixel motion for mv filtering.")
        p.add_argument("--workspace-depth", type=float, default=1.2,
                       help="Maximum valid initial depth in meters.")
    else:
        p.set_defaults(flow_model_path=None, flow_ckpt=None, vae_path=None,
                       task_description="", flow_inference_steps=25,
                       guidance_scale=8.0, moving_threshold=20.0,
                       workspace_depth=1.2)

    p.add_argument("--calibration-source", choices=["sim", "zarr"],
                   default="sim",
                   help="Use sim camera calibration or pkl files from --zarr.")
    p.add_argument("--n-rollouts", type=int, default=20,
                   help="Number of evaluation rollouts.")
    p.add_argument("--horizon", type=int, default=200,
                   help="Max env steps per rollout.")
    p.add_argument("--num-inference-steps", type=int, default=None,
                   help="DDIM steps for the diffusion policy. Defaults to the "
                        "training config when omitted.")
    p.add_argument("--max-policy-steps", type=int, default=None,
                   help="Optional cap on policy chunks. Defaults to horizon / "
                        "action_horizon.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--video-dir", default=None,
                   help="Save per-episode videos and flow visualizations.")
    p.add_argument("--result-path", default=None,
                   help="Save rollout statistics JSON to this path.")
    return p


def parse_args(mode=None):
    return build_arg_parser(mode).parse_args()


def run(args):
    if args.use_gt_flow and not args.zarr:
        sys.exit("--zarr is required for GT-flow evaluation.")
    if args.calibration_source == "zarr" and not args.zarr:
        sys.exit("--zarr is required when --calibration-source zarr is used.")

    # ── reproducibility ───────────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── load policy ───────────────────────────────────────────────────────────
    from im2flow2act.common.utility.model import load_config
    model_cfg = load_config(args.policy_path)
    print(f"Loading policy checkpoint epoch {args.policy_ckpt} ...")
    model, noise_scheduler = load_flow_diffusion_model(
        args.policy_path,
        args.policy_ckpt,
        use_ema=model_cfg.training.use_ema,
    )
    model.eval()

    stats = read_pickle(os.path.join(args.policy_path, "stats.pickle"))

    # build point_cloud stats (mirrors DiffusionFlowBCCloseLoopDataset.__init__)
    if "point_cloud" not in stats and "action" in stats:
        pc_stats = stats["action"].copy()
        pc_stats["min"] = pc_stats["min"][:3]
        pc_stats["max"] = pc_stats["max"][:3]
        stats["point_cloud"] = pc_stats

    # ── load camera calibration ───────────────────────────────────────────────
    # Default ("sim"): compute from the mimiclabs MuJoCo env at runtime. The
    # zarr pkls were generated on a different scene (UR5e im2flow2act sim), so
    # using them here back-projects depth into the wrong world frame.
    camera_intrinsic = None
    camera_pose_matrix = None
    if args.calibration_source == "zarr":
        camera_intrinsic = read_pickle(os.path.join(args.zarr, "camera_intrinsic.pkl"))
        camera_pose_matrix = read_pickle(os.path.join(args.zarr, "camera_pose_matrix.pkl"))
        print(f"[calibration] Using zarr pkls from {args.zarr}")

    # ── policy hyperparams ────────────────────────────────────────────────────
    ds_cfg = model_cfg.dataset
    obs_horizon          = ds_cfg.obs_horizon
    action_horizon       = ds_cfg.action_horizon
    pred_horizon         = ds_cfg.pred_horizon
    target_flow_horizon  = ds_cfg.target_flow_horizon
    action_dim           = model_cfg.action_dim
    num_points           = ds_cfg.num_points
    point_tracking_img_size = list(ds_cfg.point_tracking_img_size)   # [256, 256]
    resize_shape         = tuple(ds_cfg.camera_resize_shape)          # (224, 224)
    sample_frames        = ds_cfg.sample_frames                       # 32
    normalize_pointcloud = ds_cfg.normalize_pointcloud
    policy_inference_steps = (
        args.num_inference_steps
        if args.num_inference_steps is not None
        else getattr(model_cfg, "num_inference_steps", 16)
    )
    if stats["action"]["min"].shape[0] != action_dim:
        raise ValueError(
            f"Policy config action_dim={action_dim}, but stats action dim is "
            f"{stats['action']['min'].shape[0]}."
        )

    print(
        f"Policy config: obs_horizon={obs_horizon}, action_horizon={action_horizon}, "
        f"pred_horizon={pred_horizon}, num_points={num_points}, "
        f"resize_shape={resize_shape}"
    )

    # ── load flow model (generated mode) ─────────────────────────────────────
    flow_pipeline = None
    if not args.use_gt_flow:
        if args.flow_model_path is None or args.flow_ckpt is None:
            sys.exit("--flow-model-path and --flow-ckpt are required unless --use-gt-flow.")
        print(f"Loading AnimateFlow model (epoch {args.flow_ckpt}) ...")
        flow_pipeline = load_flow_model(args.flow_model_path, args.flow_ckpt,
                                        vae_path_override=args.vae_path)

    # ── open zarr (GT flow mode only) ─────────────────────────────────────────
    zarr_root = None
    if args.use_gt_flow:
        import zarr as zarr_lib
        zarr_root = zarr_lib.open(args.zarr, mode="r")
        print(f"Opened zarr store: {args.zarr}")

    # ── load demo states (GT flow mode: reset to demo initial state) ──────────
    demo_states = []   # list of {"states": ..., "model": ...}
    if args.use_gt_flow:
        with h5py.File(args.hdf5, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            for dk in demo_keys[: args.n_rollouts]:
                demo = f["data"][dk]
                entry = {"states": demo["states"][0]}
                if "model_file" in demo.attrs:
                    entry["model"] = demo.attrs["model_file"]
                demo_states.append(entry)
        if len(demo_states) < args.n_rollouts:
            sys.exit(
                f"Requested {args.n_rollouts} GT rollouts, but only "
                f"{len(demo_states)} demo states exist in {args.hdf5}."
            )
        print(f"Loaded {len(demo_states)} demo initial states for GT flow mode.")

    # ── initialize ObsUtils so env.get_observation() can process image keys ───
    # This is normally done inside FileUtils.restore_model(), but we load the
    # im2flow2act policy directly, so we must call it manually here.
    # Without this, OBS_KEYS_TO_MODALITIES is None and env.reset() crashes.
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs={
        "obs": {
            "low_dim": [],
            "rgb":   [f"{args.camera_name}_image"],
            "depth": [f"{args.camera_name}_depth"],
        }
    })

    # ── create robomimic environment ──────────────────────────────────────────
    print(f"Creating robomimic environment from {args.hdf5} ...")
    env_meta = FileUtils.get_env_metadata_from_dataset(args.hdf5)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        use_depth_obs=True,
    )
    print("Environment created.")

    # ── rollout loop ──────────────────────────────────────────────────────────
    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)

    all_stats = []
    warned_missing_sim = False

    for ep_i in tqdm(range(args.n_rollouts), desc="Rollouts"):
        # reset environment
        if args.use_gt_flow:
            env.reset()
            obs = env.reset_to(demo_states[ep_i])
        else:
            obs = env.reset()

        sim = get_robosuite_sim(env)
        if sim is None and not warned_missing_sim:
            print("[WARNING] Could not locate MjSim; depth stays in [0,1].")
            warned_missing_sim = True

        initial_rgb   = get_rgb(obs, args.camera_name)              # (H, W, 3) uint8
        if args.calibration_source == "sim":
            cam_h, cam_w = initial_rgb.shape[:2]
            camera_intrinsic, camera_pose_matrix = compute_sim_camera_calibration(
                env=env, camera_name=args.camera_name,
                height=cam_h, width=cam_w,
            )
            if ep_i == 0:
                print(f"[calibration] sim-derived K=\n{camera_intrinsic}")
        initial_depth = get_depth(obs, args.camera_name, sim=sim)    # (H, W) metres
        initial_prop  = get_proprioception(obs)                      # (7,) float32

        # ── flow plan ─────────────────────────────────────────────────────────
        if args.use_gt_flow:
            episode_flow_plan = load_gt_flow_plan(
                zarr_root=zarr_root,
                episode_idx=ep_i,
                num_points=num_points,
                num_frames=sample_frames,
                point_tracking_img_size=point_tracking_img_size,
            )
        else:
            debug_out = {} if args.video_dir else None
            episode_flow_plan = generate_flow_plan(
                pipeline=flow_pipeline,
                initial_rgb=initial_rgb,
                initial_depth=initial_depth,
                text=args.task_description,
                num_points=num_points,
                num_frames=sample_frames,
                grid_size=32,
                point_tracking_img_size=point_tracking_img_size,
                resize_shape=resize_shape,
                num_inference_steps=args.flow_inference_steps,
                guidance_scale=args.guidance_scale,
                workspace_depth=args.workspace_depth,
                moving_threshold=args.moving_threshold,
                filters=("mv",),
                debug_out=debug_out,
            )
        # episode_flow_plan: (sample_frames, num_points, 3)

        # ── visualize the plan the policy will be conditioned on ─────────────
        if args.video_dir:
            try:
                save_plan_gif(
                    rgb_initial=initial_rgb,
                    plan_flow_norm=episode_flow_plan,
                    save_path=os.path.join(args.video_dir, f"episode_{ep_i:03d}_plan.gif"),
                )
                save_plan_scatter(
                    rgb_initial=initial_rgb,
                    plan_flow_norm=episode_flow_plan,
                    save_path=os.path.join(args.video_dir, f"episode_{ep_i:03d}_plan.png"),
                )
                np.save(
                    os.path.join(args.video_dir, f"episode_{ep_i:03d}_plan.npy"),
                    episode_flow_plan,
                )
                # Pre/post filter debug dumps to diagnose the plan pipeline.
                if not args.use_gt_flow and isinstance(debug_out, dict):
                    for tag, arr in debug_out.items():
                        save_plan_scatter(
                            rgb_initial=initial_rgb,
                            plan_flow_norm=arr,
                            save_path=os.path.join(
                                args.video_dir,
                                f"episode_{ep_i:03d}_{tag}.png",
                            ),
                        )
                        np.save(
                            os.path.join(
                                args.video_dir,
                                f"episode_{ep_i:03d}_{tag}.npy",
                            ),
                            arr,
                        )
            except Exception as _viz_e:
                print(f"  [WARN] Failed to save plan viz: {_viz_e}")

        # ── per-episode video writer ──────────────────────────────────────────
        video_writer = None
        if args.video_dir:
            vp = os.path.join(args.video_dir, f"rollout_{ep_i:03d}.mp4")
            video_writer = imageio.get_writer(vp, fps=20, format="FFMPEG")

        try:
            with torch.no_grad():
                ep_stats = rollout_episode(
                    model=model,
                    noise_scheduler=noise_scheduler,
                    stats=stats,
                    env=env,
                    episode_flow_plan=episode_flow_plan,
                    initial_rgb=initial_rgb,
                    initial_depth=initial_depth,
                    initial_proprioception=initial_prop,
                    camera_intrinsic=camera_intrinsic,
                    camera_pose_matrix=camera_pose_matrix,
                    obs_horizon=obs_horizon,
                    action_horizon=action_horizon,
                    pred_horizon=pred_horizon,
                    action_dim=action_dim,
                    target_flow_horizon=target_flow_horizon,
                    num_inference_steps=policy_inference_steps,
                    num_points=num_points,
                    point_tracking_img_size=point_tracking_img_size,
                    resize_shape=resize_shape,
                    normalize_pointcloud=normalize_pointcloud,
                    camera_name=args.camera_name,
                    horizon=args.horizon,
                    max_policy_steps=args.max_policy_steps,
                    video_writer=video_writer,
                    result_save_path=args.video_dir,
                    episode_idx=ep_i,
                )
        finally:
            if video_writer is not None:
                video_writer.close()

        all_stats.append(ep_stats)
        s = "SUCCESS" if ep_stats["Success_Rate"] > 0 else "FAIL"
        print(
            f"[{ep_i + 1}/{args.n_rollouts}] {s} | "
            f"Return: {ep_stats['Return']:.2f} | Horizon: {ep_stats['Horizon']}"
        )

    # ── aggregate results ─────────────────────────────────────────────────────
    combined = TensorUtils.list_of_flat_dict_to_dict_of_list(all_stats)
    avg = {k: float(np.mean(v)) for k, v in combined.items()}
    avg["Num_Success"] = int(np.sum(combined["Success_Rate"]))

    print("\n" + "=" * 60)
    print("Per-Episode Results:")
    print("=" * 60)
    per_episode = []
    for i, s in enumerate(all_stats):
        tag = "SUCCESS" if s["Success_Rate"] > 0 else "FAIL"
        print(f"  Episode {i:3d}: {tag} | Return: {s['Return']:.2f} | Horizon: {int(s['Horizon'])}")
        per_episode.append({
            "episode": i,
            "success": bool(s["Success_Rate"] > 0),
            "return": float(s["Return"]),
            "horizon": int(s["Horizon"]),
        })
    print("=" * 60)
    print("Average stats:")
    print(json.dumps(avg, indent=4))

    # ── save JSON ─────────────────────────────────────────────────────────────
    output = {
        "policy_path": args.policy_path,
        "policy_ckpt": args.policy_ckpt,
        "mode": "gt_flow" if args.use_gt_flow else "generated_flow",
        "n_rollouts": args.n_rollouts,
        "horizon": args.horizon,
        "seed": args.seed,
        "average_stats": avg,
        "per_episode": per_episode,
    }
    save_path = args.result_path
    if save_path is None and args.video_dir:
        save_path = os.path.join(args.video_dir, "rollout_stats.json")
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(output, f, indent=4)
        print(f"\nSaved rollout stats to: {save_path}")


def main(mode=None):
    run(parse_args(mode))


if __name__ == "__main__":
    main()
