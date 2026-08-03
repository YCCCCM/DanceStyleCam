"""Reconstruction and body-attention objectives used by released CS training."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from data.normalization import MinMaxStats, NormalizerBundle


def _denormalize(value: torch.Tensor, stats: MinMaxStats) -> torch.Tensor:
    minimum = torch.as_tensor(stats.minimum, device=value.device, dtype=value.dtype)
    maximum = torch.as_tensor(stats.maximum, device=value.device, dtype=value.dtype)
    scale = maximum - minimum
    scale = torch.where(scale < 10 * torch.finfo(value.dtype).eps, torch.ones_like(scale), scale)
    return ((value.clamp(-1, 1) + 1.0) * 0.5) * scale + minimum


def _translate(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(matrix)
    result[:, :, :3] = matrix[:, :, :3]
    result[:, :, 3] = torch.matmul(vector.unsqueeze(2), matrix[:, :, :3]).squeeze(2) + matrix[:, :, 3]
    return result


def _rotate(matrix: torch.Tensor, angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    axis = F.normalize(axis, p=2, dim=-1)
    temporary = (1 - cosine).unsqueeze(-1) * axis
    rotation = torch.zeros_like(matrix)
    rotation[:, :, 0, 0] = cosine + temporary[:, :, 0] * axis[:, :, 0]
    rotation[:, :, 0, 1] = temporary[:, :, 0] * axis[:, :, 1] + sine * axis[:, :, 2]
    rotation[:, :, 0, 2] = temporary[:, :, 0] * axis[:, :, 2] - sine * axis[:, :, 1]
    rotation[:, :, 1, 0] = temporary[:, :, 1] * axis[:, :, 0] - sine * axis[:, :, 2]
    rotation[:, :, 1, 1] = cosine + temporary[:, :, 1] * axis[:, :, 1]
    rotation[:, :, 1, 2] = temporary[:, :, 1] * axis[:, :, 2] + sine * axis[:, :, 0]
    rotation[:, :, 2, 0] = temporary[:, :, 2] * axis[:, :, 0] + sine * axis[:, :, 1]
    rotation[:, :, 2, 1] = temporary[:, :, 2] * axis[:, :, 1] - sine * axis[:, :, 0]
    rotation[:, :, 2, 2] = cosine + temporary[:, :, 2] * axis[:, :, 2]
    result = torch.zeros_like(matrix)
    result[:, :, :3] = torch.matmul(rotation[:, :, :3, :3], matrix[:, :, :3])
    result[:, :, 3] = matrix[:, :, 3]
    return result


def _body_attention(
    camera: torch.Tensor,
    motion: torch.Tensor,
    bone_mask: torch.Tensor,
    inference_mask: torch.Tensor,
    normalizers: NormalizerBundle,
) -> tuple[torch.Tensor, torch.Tensor]:
    motion_world = _denormalize(motion, normalizers["pose"]).reshape(*motion.shape[:2], 60, 3)
    keypoints = motion_world.permute(2, 0, 1, 3)
    distance = _denormalize(camera[:, :, :1], normalizers["camera_distance"])
    position = _denormalize(camera[:, :, 1:4], normalizers["camera_position"])
    rotation = _denormalize(camera[:, :, 4:7], normalizers["camera_rotation"])
    fov = _denormalize(camera[:, :, 7:8], normalizers["camera_fov"])

    batch, frames, _ = rotation.shape
    one = torch.ones(batch, frames, 1, device=camera.device, dtype=camera.dtype)
    zero = torch.zeros_like(one)
    view = torch.eye(4, device=camera.device, dtype=camera.dtype).expand(batch, frames, 4, 4).clone()
    view = _translate(view, torch.cat((zero, zero, distance.abs()), dim=-1))
    rotate = torch.eye(4, device=camera.device, dtype=camera.dtype).expand(batch, frames, 4, 4).clone()
    rotate = _rotate(rotate, rotation[:, :, 1], torch.cat((zero, one, zero), dim=-1))
    rotate = _rotate(rotate, rotation[:, :, 2], torch.cat((zero, zero, -one), dim=-1))
    rotate = _rotate(rotate, rotation[:, :, 0], torch.cat((one, zero, zero), dim=-1))
    view = torch.matmul(view, rotate)
    eye = view[:, :, 3, :3] + position * torch.cat((one, one, -one), dim=-1)
    axis_z = F.normalize(-view[:, :, 2, :3], p=2, dim=-1)
    axis_y = F.normalize(view[:, :, 1, :3], p=2, dim=-1)
    axis_x = F.normalize(view[:, :, 0, :3], p=2, dim=-1)

    motion_to_eye = keypoints - eye
    projected_yz = motion_to_eye - axis_x * torch.sum(motion_to_eye * axis_x, dim=-1, keepdim=True)
    projected_xz = motion_to_eye - axis_y * torch.sum(motion_to_eye * axis_y, dim=-1, keepdim=True)
    cosine_yz = torch.sum(projected_yz * axis_z, dim=-1)
    cosine_xz = torch.sum(projected_xz * axis_z, dim=-1)
    cosine_fov = torch.cos(fov[:, :, 0] * 0.5 / 180.0 * math.pi)
    difference_x = (
        cosine_fov * torch.sqrt(torch.sum(projected_xz * projected_xz, dim=-1)) - cosine_xz
    ).permute(1, 2, 0)
    difference_y = (
        cosine_fov * torch.sqrt(torch.sum(projected_yz * projected_yz, dim=-1)) - cosine_yz
    ).permute(1, 2, 0)

    denominator = inference_mask.sum(dim=1)
    inside = (F.relu(difference_x * bone_mask) + F.relu(difference_y * bone_mask)) * inference_mask
    outside = F.relu(difference_x * (bone_mask - 1)) * F.relu(difference_y * (bone_mask - 1))
    outside = outside * inference_mask
    return (inside.sum(dim=1) / denominator).mean(), (outside.sum(dim=1) / denominator).mean()


def calculate_cs_losses(
    result: torch.Tensor,
    target: torch.Tensor,
    inference_mask: torch.Tensor,
    bone_mask: torch.Tensor,
    motion: torch.Tensor,
    normalizers: NormalizerBundle,
    history_len: int,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    denominator = inference_mask.sum(dim=1)
    reconstruction = (((result - target) ** 2) * inference_mask).sum(dim=1) / denominator

    target_velocity = target[:, 1:] - target[:, :-1]
    result_velocity = result[:, 1:] - result[:, :-1]
    velocity_mask = inference_mask.clone()
    velocity_mask[:, history_len : history_len + 1] = 0
    velocity = (((result_velocity - target_velocity) ** 2) * velocity_mask[:, 1:]).sum(dim=1) / denominator

    target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
    result_acceleration = result_velocity[:, 1:] - result_velocity[:, :-1]
    acceleration_mask = inference_mask.clone()
    acceleration_mask[:, history_len : history_len + 2] = 0
    acceleration = (
        ((result_acceleration - target_acceleration) ** 2) * acceleration_mask[:, 2:]
    ).sum(dim=1) / denominator
    inside, outside = _body_attention(result, motion, bone_mask, inference_mask, normalizers)
    components = {
        "reconstruction": weights["reconstruction"] * reconstruction.mean(),
        "velocity": weights["velocity"] * velocity.mean(),
        "acceleration": weights["acceleration"] * acceleration.mean(),
        "inside_body_attention": weights["inside_body_attention"] * inside,
        "outside_body_attention": weights["outside_body_attention"] * outside,
    }
    return sum(components.values()), components
