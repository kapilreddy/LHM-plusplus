# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2026-03-10 10:00:00
# @Function      : LHM++ Gradio App inference logic

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from core.utils.hf_hub import wrap_model_hub

# Default batch size for inference
DEFAULT_BATCH_SIZE = 40


def parse_app_configs(
    model_cards: Dict[str, Dict[str, str]],
) -> Tuple[DictConfig, DictConfig]:
    """Parse model configuration from environment variables and config files.

    Returns:
        A tuple of (cfg, cfg_train) containing merged configurations.
    """
    cli_cfg = OmegaConf.create()
    cfg = OmegaConf.create()

    app_model_name = os.environ.get("APP_MODEL_NAME")
    if app_model_name is None:
        raise NotImplementedError("APP_MODEL_NAME environment variable must be set")

    model_card = model_cards[app_model_name]
    model_path = model_card["model_path"]
    model_config = model_card["model_config"]

    cli_cfg.model_name = model_path

    if model_config is not None:
        cfg_train = OmegaConf.load(model_config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except AttributeError:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high

        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split("_")[-1],
        )

        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)

    cfg.motion_video_read_fps = 6
    cfg.merge_with(cli_cfg)
    cfg.setdefault("logger", "INFO")
    assert cfg.model_name is not None, "model_name is required"

    return cfg, cfg_train


def build_app_model(cfg: DictConfig) -> torch.nn.Module:
    """Build and load the LHM model from pretrained weights.

    Args:
        cfg: Configuration object containing model_name and other parameters.

    Returns:
        Loaded LHM model ready for inference.
    """
    from core.models import model_dict

    model_cls = wrap_model_hub(model_dict["human_lrm_a4o"])
    model = model_cls.from_pretrained(cfg.model_name)
    return model


@torch.no_grad()
def inference_results(
    model: torch.nn.Module,
    ref_img_tensors: torch.Tensor,
    smplx_params: Dict[str, torch.Tensor],
    motion_seq: Dict[str, Any],
    video_size: int = 40,
    ref_imgs_bool: Optional[torch.Tensor] = None,
    visualized_center: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cuda",
    return_alpha: bool = False,
):
    """Run inference on a motion sequence with batching to prevent OOM.

    Args:
        model: LHM model for human animation.
        ref_img_tensors: Reference image tensors of shape (N, C, H, W).
        smplx_params: SMPL-X parameters for the initial pose.
        motion_seq: Dictionary containing motion sequence data.
        video_size: Total number of frames to render.
        ref_imgs_bool: Boolean mask indicating which reference images to use.
        visualized_center: If True, crops output to subject bounds with 10% padding.
        batch_size: Number of frames to process in each batch.
        device: Device to run inference on.
        return_alpha: If True, also return the per-frame alpha mask the
            renderer already produces, so callers can composite the person
            onto an arbitrary background instead of the baked-in white.
            The mask is the same one used internally for visualized_center
            cropping; opt-in keeps the existing rgb-only callers working.

    Returns:
        If return_alpha is False (default): rendered RGB frames as a
        numpy array of shape (T, H, W, 3) -- backward-compatible.
        If return_alpha is True: tuple (rgb, alpha) where alpha has shape
        (T, H, W) in the same uint8 [0, 255] range as rgb.
    """
    offset_list = motion_seq.get("offset_list")
    ori_h, ori_w = motion_seq.get("ori_size", (512, 512))
    output_rgb = torch.ones((ori_h, ori_w, 3))
    ref_imgs_bool = torch.ones(
        ref_img_tensors.shape[0], dtype=torch.bool, device=device
    )

    model_outputs = model.infer_single_view(
        ref_img_tensors.unsqueeze(0).to(device),
        None,
        None,
        render_c2ws=motion_seq["render_c2ws"].to(device),
        render_intrs=motion_seq["render_intrs"].to(device),
        render_bg_colors=motion_seq["render_bg_colors"].to(device),
        smplx_params={k: v.to(device) for k, v in smplx_params.items()},
        ref_imgs_bool=ref_imgs_bool.unsqueeze(0),
    )

    if len(model_outputs) == 7:
        (
            gs_model_list,
            query_points,
            transform_mat_neutral_pose,
            gs_hidden_features,
            image_latents,
            motion_emb,
            pos_emb,
        ) = model_outputs
    else:
        (
            gs_model_list,
            query_points,
            transform_mat_neutral_pose,
            gs_hidden_features,
            image_latents,
            motion_emb,
        ) = model_outputs
        pos_emb = None

    batch_smplx_params = {
        "betas": smplx_params["betas"].to(device),
        "transform_mat_neutral_pose": transform_mat_neutral_pose,
    }

    frame_varying_keys = [
        "root_pose",
        "body_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "lhand_pose",
        "rhand_pose",
        "trans",
        "focal",
        "princpt",
        "img_size_wh",
        "expr",
    ]

    batch_rgb_list = []
    batch_mask_list = []
    # Raw gsplat alpha (separate from the neural-renderer mask in
    # batch_mask_list). Only populated when return_alpha=True so we don't
    # pay the extra collect/concat for the rgb-only callers.
    batch_gs_alpha_list = []
    num_batches = (video_size + batch_size - 1) // batch_size

    for batch_idx in range(0, video_size, batch_size):
        current_batch = batch_idx // batch_size + 1
        print(f"Processing batch {current_batch}/{num_batches}")

        batch_smplx_params.update(
            {
                key: motion_seq["smplx_params"][key][
                    :, batch_idx : batch_idx + batch_size
                ].to(device)
                for key in frame_varying_keys
            }
        )

        mask_seqs = (
            motion_seq.get("masks", [])[batch_idx : batch_idx + batch_size]
            if "masks" in motion_seq
            else None
        )

        anim_kwargs = {
            "gs_model_list": gs_model_list,
            "query_points": query_points,
            "smplx_params": batch_smplx_params,
            "render_c2ws": motion_seq["render_c2ws"][
                :, batch_idx : batch_idx + batch_size
            ].to(device),
            "render_intrs": motion_seq["render_intrs"][
                :, batch_idx : batch_idx + batch_size
            ].to(device),
            "render_bg_colors": motion_seq["render_bg_colors"][
                :, batch_idx : batch_idx + batch_size
            ].to(device),
            "gs_hidden_features": gs_hidden_features,
            "image_latents": image_latents,
            "motion_emb": motion_emb,
        }

        if pos_emb is not None:
            anim_kwargs["pos_emb"] = pos_emb
        if offset_list is not None:
            anim_kwargs["offset_list"] = offset_list[batch_idx : batch_idx + batch_size]
        if mask_seqs is not None:
            anim_kwargs["mask_seqs"] = mask_seqs
        if output_rgb is not None:
            anim_kwargs["output_rgb"] = output_rgb

        if return_alpha:
            anim_kwargs["return_gs_alpha"] = True
            batch_rgb, batch_mask, batch_gs_alpha = model.animation_infer(**anim_kwargs)
            batch_gs_alpha_list.append(
                (batch_gs_alpha.clamp(0, 1) * 255).to(torch.uint8).numpy()
            )
        else:
            batch_rgb, batch_mask = model.animation_infer(**anim_kwargs)
        batch_rgb_list.append((batch_rgb.clamp(0, 1) * 255).to(torch.uint8).numpy())
        batch_mask_list.append((batch_mask.clamp(0, 1) * 255).to(torch.uint8).numpy())
    
    print("End of inference")
    
    rgb = np.concatenate(batch_rgb_list, axis=0)
    # The neural mask is what visualized_center uses to find subject
    # bounds. Squeeze the trailing channel dim so it's plain (T, H, W).
    mask = np.concatenate(batch_mask_list, axis=0)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    # The raw gsplat alpha is what we hand back to callers asking for
    # alpha-keyed output. Same shape contract as mask.
    if return_alpha:
        gs_alpha = np.concatenate(batch_gs_alpha_list, axis=0)
        if gs_alpha.ndim == 4 and gs_alpha.shape[-1] == 1:
            gs_alpha = gs_alpha[..., 0]

    if visualized_center:
        h_indices, w_indices = np.where(mask > 0.25)[1:]
        if len(h_indices) > 0 and len(w_indices) > 0:
            top, bottom = h_indices.min(), h_indices.max()
            left, right = w_indices.min(), w_indices.max()

            center_y, center_x = (top + bottom) / 2, (left + right) / 2
            height, width = bottom - top, right - left
            new_height, new_width = height * 1.1, width * 1.1

            top_new = max(0, int(center_y - new_height / 2))
            bottom_new = int(center_y + new_height / 2)
            left_new = max(0, int(center_x - new_width / 2))
            right_new = int(center_x + new_width / 2)

            rgb = rgb[:, top_new:bottom_new, left_new:right_new]
            mask = mask[:, top_new:bottom_new, left_new:right_new]
            if return_alpha:
                gs_alpha = gs_alpha[:, top_new:bottom_new, left_new:right_new]

    if return_alpha:
        return rgb, gs_alpha
    return rgb
