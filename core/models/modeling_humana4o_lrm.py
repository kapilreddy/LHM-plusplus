# -*- coding: utf-8 -*-# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-10-15 13:26:46
# @Function      : Core Code of LHM++

import importlib
import math
import os
import pdb
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from diffusers.utils import is_torch_version
from einops import rearrange

from core.models.arcface_utils import ResNetArcFace
from core.models.encoders.pointtransformer.point_transformer_warp import (
    PointTransformerEncoder,
)
from core.models.transformer_block.conv_decoder import ConvDecoderOnly
from core.models.transformer_block.dpt_decoder import (
    EfficientDPTDecoder,
    PatchDPT4DecoderOnly,
    PatchDPTDecoderOnly,
    PatchEfficientDPTDecoder,
    PatchRopeEfficientDPTDecoder,
)
from core.models.utils import linear
from core.models.vggt_transformer import VGGTAggregator
from core.modules.embed import PointEmbed

logger = get_logger(__name__)


class AverageTimeCalculator:
    def __init__(self, window_size=10):
        self.window_size = window_size
        from collections import deque

        self.times = deque(maxlen=window_size)

    def add_time(self, time_value):
        self.times.append(time_value)

    def average_time(self):
        if len(self.times) == 0:
            return 0.0
        return sum(self.times) / len(self.times)


def scale_intrs(intrs, ratio_x, ratio_y):
    if len(intrs.shape) >= 3:
        intrs[..., 0, :] = intrs[..., 0, :] * ratio_x
        intrs[..., 1, :] = intrs[..., 1, :] * ratio_y
    else:
        intrs[0] = intrs[0] * ratio_x
        intrs[1] = intrs[1] * ratio_y
    return intrs


def pad_to_ratio(tensor: torch.Tensor, target_ratio: float) -> tuple:
    """
    Pad a [C, H, W] tensor to match a target aspect ratio while preserving its original content.
    The padding is applied symmetrically to maintain the center position.

    Args:
        tensor: Input tensor of shape [C, H, W]
        target_ratio: Desired width/height ratio (target_ratio = target_w / target_h)

    Returns:
        padded_tensor: Padded tensor
        pad_info: Tuple of (left_pad, right_pad, top_pad, bottom_pad)
    """
    C, H, W = tensor.shape
    current_ratio = W / H

    if current_ratio > target_ratio:
        # Image is wider than target - pad height
        new_h = int(W / target_ratio)
        pad_h = new_h - H
        top_pad = pad_h // 2
        bottom_pad = pad_h - top_pad
        left_pad = right_pad = 0
        padded_tensor = F.pad(
            tensor, (0, 0, top_pad, bottom_pad), mode="constant", value=0
        )
    else:
        # Image is taller than target - pad width
        new_w = int(H * target_ratio)
        pad_w = new_w - W
        left_pad = pad_w // 2
        right_pad = pad_w - left_pad
        top_pad = bottom_pad = 0
        padded_tensor = F.pad(
            tensor, (left_pad, right_pad, 0, 0), mode="constant", value=0
        )

    return padded_tensor, (left_pad, top_pad, W + left_pad, H + top_pad)


def _gs_alpha_to_2d(mask) -> np.ndarray:
    """Normalise the gsplat comp_mask -- which arrives as (B, V, H, W),
    (H, W, 1), or (H, W) depending on the renderer code path -- into a
    plain (H, W) float32 array clipped to [0, 1] so the caller can resize
    and place it without per-call shape juggling.
    """
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    a = np.asarray(mask, dtype=np.float32)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    while a.ndim > 2 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError(
            f"_gs_alpha_to_2d: cannot reduce shape {a.shape} to (H, W)"
        )
    return np.clip(a, 0.0, 1.0)


class ModelHumanA4OLRM(nn.Module):
    """
    ModelHumanA4OLRM

    This class implements the full Efficient Large Human Reconstruction Model (LHMPP),
    designed for single-view or multi-view large-scale human reconstruction tasks.
    It provides flexibility for a variety of transformer decoder backbones, 2D/3D
    feature embeddings, and supports various encoders (DINO, DINOv2, resunet, Sapiens, etc.).
    The model can utilize additional modules like SMPL-X parametric human models,
    point cloud embeddings, GS-based rendering networks, and optionally integrates
    face identification and super-resolution options.

    """

    def __init__(
        self,
        transformer_dim: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_type="cond",
        tf_grad_ckpt=False,
        encoder_grad_ckpt=False,
        encoder_freeze: bool = True,
        encoder_type: str = "dino",
        encoder_model_name: str = "facebook/dino-vitb16",
        encoder_feat_dim: int = 768,
        num_pcl: int = 2048,
        pcl_dim: int = 512,
        human_model_path=None,
        smplx_subdivide_num=2,
        smplx_type="smplx",
        gs_query_dim=None,
        gs_use_rgb=False,
        gs_sh=3,
        gs_mlp_network_config=None,
        gs_xyz_offset_max_step=1.8 / 32,
        gs_clip_scaling=0.2,
        shape_param_dim=100,
        expr_param_dim=50,
        fix_opacity=False,
        fix_rotation=False,
        use_face_id=False,
        facesr=False,
        use_stylegan2_prior=False,
        **kwargs,
    ):
        """
        Initialize ModelHumanA4OLRM.

        Args:
            transformer_dim (int): The dimensionality of transformer features.
            transformer_layers (int): Number of transformer layers.
            transformer_heads (int): Number of heads in transformer.
            transformer_type (str, optional): Transformer type. Defaults to "cond".
            tf_grad_ckpt (bool, optional): Enable gradient checkpointing for transformer. Defaults to False.
            encoder_grad_ckpt (bool, optional): Enable gradient checkpointing for encoder. Defaults to False.
            encoder_freeze (bool, optional): Whether to freeze image encoder. Defaults to True.
            encoder_type (str, optional): Which encoder to use ('dino', 'dino_v2', etc). Defaults to "dino".
            encoder_model_name (str, optional): Name or path of encoder weights. Defaults to "facebook/dino-vitb16".
            encoder_feat_dim (int, optional): Output features from encoder. Defaults to 768.
            num_pcl (int, optional): Number of points for point cloud embedding. Defaults to 2048.
            pcl_dim (int, optional): Latent embedding dimension of point cloud. Defaults to 512.
            human_model_path (str, optional): Path to directory containing parametric human models.
            smplx_subdivide_num (int, optional): Number of mesh subdivisions for SMPL-X. Defaults to 2.
            smplx_type (str, optional): SMPL-X variant type. Defaults to "smplx".
            gs_query_dim (int, optional): Dimensionality of the Gaussian Splatting query.
            gs_use_rgb (bool, optional): Whether to use RGB for GS rendering. Defaults to False.
            gs_sh (int, optional): Spherical harmonics degree for GS. Defaults to 3.
            gs_mlp_network_config (dict, optional): MLP config for GS renderer.
            gs_xyz_offset_max_step (float, optional): Max offset step for GS xyz. Defaults to 1.8 / 32.
            gs_clip_scaling (float, optional): Scaling used for clipping in GS. Defaults to 0.2.
            shape_param_dim (int, optional): Shape parameter vector dimension. Defaults to 100.
            expr_param_dim (int, optional): Expression parameter vector dimension. Defaults to 50.
            fix_opacity (bool, optional): Fix opacity in GS rendering. Defaults to False.
            fix_rotation (bool, optional): Fix rotation in GS rendering. Defaults to False.
            use_face_id (bool, optional): Enable face identity/recognition. Defaults to False.
            facesr (bool, optional): Use face super-resolution module. Defaults to False.
            use_stylegan2_prior (bool, optional): Use StyleGAN2 prior for face. Defaults to False.
            **kwargs: Additional configurable arguments spanning transformer decoder, embedding, rendering, etc.

        """

        super().__init__()

        self.gradient_checkpointing = tf_grad_ckpt
        self.encoder_gradient_checkpointing = encoder_grad_ckpt

        # attributes
        self.encoder_feat_dim = encoder_feat_dim

        # modules
        # images encoder  default dino-v2
        self.encoder = self._encoder_fn(encoder_type)(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
            encoder_feat_dim=encoder_feat_dim,
            encoder_params=kwargs.get("encoder_params", {}),
        )

        # using sapiens to extract human motion features

        # learnable points embedding
        skip_decoder = False
        self.latent_query_points_type = kwargs.get(
            "latent_query_points_type", "embedding"
        )
        if self.latent_query_points_type == "embedding":
            self.num_pcl = num_pcl  # 2048
            self.pcl_embeddings = nn.Embedding(num_pcl, pcl_dim)  # 1024
        elif self.latent_query_points_type.startswith("smplx"):
            latent_query_points_file = os.path.join(
                human_model_path, "smplx_points", f"{self.latent_query_points_type}.npy"
            )
            pcl_embeddings = torch.from_numpy(np.load(latent_query_points_file)).float()
            print(
                f"==========load smplx points:{latent_query_points_file}, shape:{pcl_embeddings.shape}"
            )
            self.register_buffer("pcl_embeddings", pcl_embeddings)
            self.pcl_embed = PointEmbed(dim=pcl_dim)
        elif self.latent_query_points_type.startswith(
            "e2e_smplx_pt"
        ):  # point_transformer
            skip_decoder = True
            self.pcl_embed = PointTransformerEncoder(c=3, planes=[128, 256, pcl_dim])
        elif self.latent_query_points_type.startswith("e2e_smplx"):
            skip_decoder = True
            self.pcl_embed = PointEmbed(dim=pcl_dim)  # pcl dim 1024
        elif self.latent_query_points_type.startswith(
            "e2e_points"
        ):  # identity xyz + normal
            skip_decoder = True
        else:
            raise NotImplementedError

        print(f"==========skip_decoder:{skip_decoder}")

        # build images encoder
        if "aggregator" in kwargs:
            self.images_aggregator = VGGTAggregator(**kwargs["aggregator"])
        else:
            self.images_aggregator = None

        # query transformer
        self.transformer = self.build_transformer(
            transformer_type,
            transformer_layers,
            transformer_heads,
            transformer_dim,
            encoder_feat_dim,  # global feature + frame feature
            **kwargs,
        )

        # motion embedding
        input_dim = self.encoder_feat_dim
        mid_dim = pcl_dim // 4
        self.motion_embed_mlp = nn.Sequential(
            linear(input_dim, mid_dim),
            nn.SiLU(),
            linear(mid_dim, pcl_dim),
        )

        # build gs render
        self.renderer = self.build_gsrender(
            human_model_path,
            smplx_subdivide_num,
            smplx_type,
            transformer_dim,
            gs_query_dim,
            gs_use_rgb,
            gs_sh,
            gs_mlp_network_config,
            gs_xyz_offset_max_step,
            gs_clip_scaling,
            shape_param_dim,
            expr_param_dim,
            fix_opacity,
            fix_rotation,
            skip_decoder,
            **kwargs,
        )
        # build neural renderer (dense prediction: RGB + mask)
        if "neural_renderer" in kwargs:
            self.neural_renderer = self._build_neural_renderer(**kwargs)
            self.neural_renderer_patch_size = kwargs.get(
                "neural_renderer_patch_size", 14
            )
            self.neural_renderer_input = kwargs.get("neural_renderer_input", "feats")
        else:
            self.neural_renderer = None  # without neural renderer.
            self.neural_renderer_patch_size = 1
        
        # face_id
        self.use_face_id = use_face_id

        if self.use_face_id:
            self.id_face_net = ResNetArcFace()

    def _build_neural_renderer(self, **kwargs) -> nn.Module:
        """
        Build neural renderer (dense prediction) module according to config.

        This function selects and instantiates the appropriate decoder class
        based on the provided 'neural_renderer' configuration in kwargs.

        Returns:
            nn.Module: An instance of the selected neural renderer.

        Raises:
            NotImplementedError: If the specified decoder type is not supported.
        """
        neural_renderer_kwargs = kwargs["neural_renderer"]
        neural_renderer_type = neural_renderer_kwargs["type"]
        if neural_renderer_type == "efficientdpt":
            return EfficientDPTDecoder(**neural_renderer_kwargs)
        elif neural_renderer_type == "patch_efficientdpt":
            return PatchEfficientDPTDecoder(**neural_renderer_kwargs)
        elif neural_renderer_type == "patch_rope_efficientdpt":
            return PatchRopeEfficientDPTDecoder(**neural_renderer_kwargs)
        elif neural_renderer_type == "patch_dptonly":
            return PatchDPTDecoderOnly(**neural_renderer_kwargs)
        elif neural_renderer_type == "patch_4dptonly":
            return PatchDPT4DecoderOnly(**neural_renderer_kwargs)
        elif neural_renderer_type == "patch_convonly":
            return ConvDecoderOnly(**neural_renderer_kwargs)
        else:
            raise NotImplementedError(
                f"Unsupported neural_renderer type: {neural_renderer_type}"
            )

    def train(self, mode=True):
        super().train(mode)
        if self.use_face_id:
            # setting id_face_net to evaluation
            self.id_face_net.eval()

    def build_gsrender(
        self,
        human_model_path,
        smplx_subdivide_num,
        smplx_type,
        transformer_dim,
        gs_query_dim,
        gs_use_rgb,
        gs_sh,
        gs_mlp_network_config,
        gs_xyz_offset_max_step,
        gs_clip_scaling,
        shape_param_dim,
        expr_param_dim,
        fix_opacity,
        fix_rotation,
        skip_decoder,
        **kwargs,
    ):
        """
        Build and return the GS renderer instance based on specified configuration.

        Args:
            human_model_path: Path to human body model (e.g., SMPL-X).
            smplx_subdivide_num: Number of subdivisions for the mesh.
            smplx_type: String specifying SMPL-X type.
            transformer_dim: Dimensionality for transformer features.
            gs_query_dim: Dimensionality of the Gaussian Splatting query.
            gs_use_rgb: Whether to use RGB color in Gaussian Splatting.
            gs_sh: Spherical harmonics degree.
            gs_mlp_network_config: Network config for MLP in GS.
            gs_xyz_offset_max_step: Maximum allowable step for xyz offset in GS.
            gs_clip_scaling: Clipping scaling value for GS.
            shape_param_dim: Dimensionality for shape parameters.
            expr_param_dim: Dimensionality for expression parameters.
            fix_opacity: Whether to fix opacity during rendering.
            fix_rotation: Whether to fix rotation during rendering.
            skip_decoder: Whether to skip the decoder module.

        Keyword Args:
            cano_pose_type: Canonical pose type (default: 0).
            dense_sample_pts: Number of dense sample points for rendering (default: 40000).
            gs_deform_scale: Deformation scale for GS (default: 0.005).
            render_features: Whether to render with features only (default: False).
            gs_rendering: Rendering type string (default: 'venilla').
            deform_head_type: Deformation head type if using deformable GS.
            decoder_mlp: Whether to use a decoder MLP (default: False).
            decode_with_extra_info: Provide extra info to the decoder if specified.

        Returns:
            An instance of the selected GS renderer class, properly configured.
        """

        # Determine renderer type and import appropriate class

        cano_pose_type = kwargs.get("cano_pose_type", 0)
        dense_sample_pts = kwargs.get("dense_sample_pts", 40000)
        gs_deform_scale = kwargs.get("gs_deform_scale", 0.005)
        render_features = kwargs.get("render_features", False)
        gs_render_type = kwargs.get("gs_rendering", "venilla")

        renderer_classes = {
            "venilla": "core.models.rendering.gs_renderer.GS3DRenderer",
            "gsplat": "core.models.rendering.gsplat_renderer.GSPlatRenderer",
            "featsplat": "core.models.rendering.gsplat_renderer.GSPlatFeatRenderer",
            "featbacksplat": "core.models.rendering.gsplat_renderer.GSPlatBackFeatRenderer",
            "posfeatsplat": "core.models.rendering.gsplat_renderer.GSPlatFeatPosEmbedRenderer",
        }

        if gs_render_type not in renderer_classes:
            raise NotImplementedError(f"Renderer type {gs_render_type} not supported")

        module, class_name = renderer_classes[gs_render_type].rsplit(".", 1)

        GS3DRenderer = getattr(importlib.import_module(module), class_name)

        # Prepare renderer-specific kwargs
        renderer_kwargs = {}
        if "deformable" in gs_render_type:
            renderer_kwargs["deform_head_type"] = kwargs.get("deform_head_type", "mlp")

        # Initialize the renderer
        return GS3DRenderer(
            human_model_path=human_model_path,
            subdivide_num=smplx_subdivide_num,
            smpl_type=smplx_type,
            feat_dim=transformer_dim,
            query_dim=gs_query_dim,
            use_rgb=gs_use_rgb,
            sh_degree=gs_sh,
            mlp_network_config=gs_mlp_network_config,
            xyz_offset_max_step=gs_xyz_offset_max_step,
            clip_scaling=gs_clip_scaling,
            shape_param_dim=shape_param_dim,
            expr_param_dim=expr_param_dim,
            cano_pose_type=cano_pose_type,
            fix_opacity=fix_opacity,
            fix_rotation=fix_rotation,
            decoder_mlp=kwargs.get("decoder_mlp", False),
            skip_decoder=skip_decoder,
            decode_with_extra_info=kwargs.get("decode_with_extra_info", None),
            gradient_checkpointing=self.gradient_checkpointing,
            apply_pose_blendshape=kwargs.get("apply_pose_blendshape", False),
            dense_sample_pts=dense_sample_pts,
            gs_deform_scale=gs_deform_scale,
            render_features=render_features,
            **renderer_kwargs,
        )

    def build_transformer(
        self,
        transformer_type,
        transformer_layers,
        transformer_heads,
        transformer_dim,
        encoder_feat_dim,
        **kwargs,
    ):
        """
        Build and return a transformer decoder module based on the specified transformer type and configuration.

        Args:
            transformer_type (str): The type of transformer block to use.
            transformer_layers (int): Number of transformer layers.
            transformer_heads (int): Number of self-attention heads.
            transformer_dim (int): Feature dimensionality for transformer.
            encoder_feat_dim (int): Feature dimensionality for the encoder output.
            **kwargs: Additional keyword arguments containing decoder configs, must include 'transformer_decoder' key.

        Returns:
            nn.Module: Instantiated transformer decoder module as per the provided configuration.

        Raises:
            NotImplementedError: If specified transformer_decoder 'type' is not recognized.
        """

        transformer_decoder_kwargs = kwargs["transformer_decoder"]
        transformer_decoder_type = transformer_decoder_kwargs["type"]

        if transformer_decoder_type == "patch_pvt_mm_encoderonly":
            from core.models.PI_transformer import (
                PITransformerA4OEncoderOnly as TransformerDecoder,  # PI-Encoder Only
            )
        elif transformer_decoder_type == "patch_pvt_mm_encoder":
            from core.models.PI_transformer import (
                PITransformerA4OEncoder as TransformerDecoder,  # PI-Encoder Only with Middle Layer, flexible for Scaling Up.
            )
        elif transformer_decoder_type == "patch_pvt_mm_encoder_decoder":
            from core.models.PI_transformer import (
                PITransformerA4OE2EEncoderDecoder as TransformerDecoder,  # PI-Encoder-Decoder
            )
        elif transformer_decoder_type == "patch_efficient_pvt_mm_encoder_decoder":
            from core.models.transformer_block.efficient_pi_transformer import (
                EfficientPITransformerA4OE2EEncoderDecoder as TransformerDecoder,  # Efficient-PI-Encoder-Decoder
            )
        elif transformer_decoder_type == "patch_efficient_pvt_mm_encoder_decoder_dense":
            from core.models.transformer_block.efficient_pi_transformer_dense import (
                EfficientPITransformerA4OE2EEncoderDecoderDense as TransformerDecoder,  # Final Model Architecture of LHM++;
            )
        else:
            raise NotImplementedError

        return TransformerDecoder(
            img_dim=encoder_feat_dim,
            dim=transformer_dim,
            num_heads=transformer_heads,
            gradient_checkpointing=self.gradient_checkpointing,
            **transformer_decoder_kwargs,
        )

    def get_last_layer(self):
        return self.renderer.gs_net.out_layers["shs"].weight

    def hyper_step(self, step):
        self.renderer.hyper_step(step)

    @staticmethod
    def _encoder_fn(encoder_type: str):
        """
        Returns the encoder constructor/class for a given encoder type.

        Args:
            encoder_type (str): The type of encoder to use. Supported types include:
                'dino', 'dinov2', 'dinov2_unet', 'resunet', 'dinov2_featup',
                'dinov2_dpt', 'dinov2_fusion', 'sapiens', 'sapiens_class'.

        Returns:
            class: The class or constructor of the specified encoder.

        Raises:
            AssertionError: If the encoder_type is not supported.
        """

        encoder_type = encoder_type.lower()
        assert encoder_type in [
            "dino",
            "dinov2",
            "dinov2_unet",
            "resunet",
            "dinov2_featup",
            "dinov2_dpt",
            "dinov2_fusion",
            "sapiens",
            "sapiens_class",
        ], "Unsupported encoder type"
        if encoder_type == "dino":
            from .encoders.dino_wrapper import DinoWrapper

            logger.info("Using DINO as the encoder")
            return DinoWrapper
        elif encoder_type == "dinov2":
            from .encoders.dinov2_wrapper import Dinov2Wrapper

            logger.info("Using DINOv2 as the encoder")
            return Dinov2Wrapper
        elif encoder_type == "dinov2_unet":
            from .encoders.dinov2_unet_wrapper import Dinov2UnetWrapper

            logger.info("Using Dinov2Unet as the encoder")
            return Dinov2UnetWrapper
        elif encoder_type == "resunet":
            from .encoders.xunet_wrapper import XnetWrapper

            logger.info("Using XnetWrapper as the encoder")
            return XnetWrapper
        elif encoder_type == "dinov2_featup":
            from .encoders.dinov2_featup_wrapper import Dinov2FeatUpWrapper

            logger.info("Using Dinov2FeatUpWrapper as the encoder")
            return Dinov2FeatUpWrapper
        elif encoder_type == "dinov2_dpt":
            from .encoders.dinov2_dpt_wrapper import Dinov2DPTWrapper

            logger.info("Using Dinov2DPTWrapper as the encoder")
            return Dinov2DPTWrapper
        elif encoder_type == "dinov2_fusion":
            from .encoders.dinov2_fusion_wrapper import Dinov2FusionWrapper

            logger.info("Using Dinov2FusionWrapper as the encoder")
            return Dinov2FusionWrapper
        elif encoder_type == "sapiens":
            from .encoders.sapiens_warpper import SapiensWrapper

            logger.info("Using Sapiens as the encoder")
            return SapiensWrapper
        elif encoder_type == "sapiens_class":
            from .encoders.sapiens_warpper import SapiensClassWrapper

            logger.info("Using Sapiens as the encoder")
            return SapiensClassWrapper

    def forward_motionembed(self, motion_tokens):
        """
        Computes motion embedding features from the given motion tokens.

        Args:
            motion_tokens (torch.Tensor): Motion embedding tokens with shape [B, N, D], where
                B is the batch size, N is number of tokens, and D the feature dimension.

        Returns:
            torch.Tensor: Motion embedding features after processing, with shape [B, D].
        """

        motion_tokens = motion_tokens.mean(dim=1, keepdim=True)
        motion_tokens = self.motion_embed_mlp(motion_tokens).squeeze(1)  # [B, D]
        return motion_tokens

    def forward_transformer(
        self, image_feats, query_points, motion_embed=None, **kwargs
    ):
        """
        Applies forward transformation to the input features.
        Args:
            image_feats (torch.Tensor): Input image features list. List[ Shape[B, C, H, W]]
            patch_start_idx: start patch index
            camera_embeddings (torch.Tensor): Camera embeddings. Shape [B, D].
            query_points (torch.Tensor): Query points. Shape [B, L, D].
        Returns:
            torch.Tensor: Transformed features. Shape [B, L, D].
        """

        B = image_feats[-1].shape[0]

        if self.latent_query_points_type == "embedding":
            range_ = torch.arange(self.num_pcl, device=image_feats.device)
            x = self.pcl_embeddings(range_).unsqueeze(0).repeat((B, 1, 1))  # [B, L, D]

        elif self.latent_query_points_type.startswith("smplx"):
            x = self.pcl_embed(self.pcl_embeddings.unsqueeze(0)).repeat(
                (B, 1, 1)
            )  # [B, L, D]

        elif self.latent_query_points_type.startswith("e2e_smplx_pt"):
            query_points = query_points["neutral_coords"]
            x = self.pcl_embed(query_points)  # [B, L, D]
        elif self.latent_query_points_type.startswith("e2e_smplx"):
            # Linear warp -> MLP + LayerNorm
            x = self.pcl_embed(query_points)  # [B, L, D]
        elif self.latent_query_points_type.startswith("e2e_points"):
            x = query_points  # embedding in decode transformer.
        else:
            raise NotImplementedError
        transformer_results = self.transformer(
            query_points=x, cond_imgs=image_feats, motion_emb=motion_embed, **kwargs
        )  # [B, L, D]
        return transformer_results

    def forward_encode_image(self, image):
        """
        Encodes the input image using the model's encoder.

        Args:
            image (torch.Tensor): Input image tensor. Shape [B, C, H, W].

        Returns:
            torch.Tensor: Image features as produced by the encoder.
        """

        if self.training and self.encoder_gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            image_feats = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.encoder),
                image,
                **ckpt_kwargs,
            )
        else:
            image_feats = self.encoder(image)

        return image_feats

    @torch.compile
    def forward_latent_points(
        self, image, camera, query_points=None, ref_imgs_bool=None
    ):
        """
        Forward pass of the latent points generation.
        Args:
            image (torch.Tensor): Input image tensor of shape [B, S, C_img, H_img, W_img].
            camera (torch.Tensor): Camera tensor of shape [B, D_cam_raw].
            query_points (torch.Tensor, optional): Query points tensor. for example, smplx surface points, Defaults to None.
        Returns:
            torch.Tensor: Generated tokens tensor.
            torch.Tensor: Encoded image features tensor.
        """

        def obtain_input_ratio(image):
            _, _, H, W = image.shape

            try:
                patch_size = self.encoder.patch_size
            except:
                down_ratio = self.encoder.downsample_ratio
                patch_size = (H // down_ratio, W // down_ratio)
            return patch_size

        # batch, the length of sequence
        B, S = image.shape[0:2]

        # encode image
        # image_feats is cond texture
        image = rearrange(image, "B S C H W -> (B S) C H W")

        image_feats = self.forward_encode_image(image)
        cls_feats, image_feats = (
            image_feats[:, :1],
            image_feats[:, 1:],
        )  # the first is the cls token

        motion_feats = self.forward_motionembed(cls_feats.mean(dim=1, keepdim=True))
        image_feats = rearrange(image_feats, "(B S) P C -> B S P C", B=B)
        motion_feats = rearrange(motion_feats, "(B S) C -> B S C", B=B)
        latent_size = obtain_input_ratio(image)

        if self.images_aggregator is not None:
            # vggt output,  [pose_tokens, register_tokens, feats tokens]
            image_tokens_list, patch_start_idx = self.images_aggregator(
                image_feats, latent_size
            )

        assert (
            image_feats.shape[-1] == self.encoder_feat_dim
        ), f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"

        # transformer generating latent points
        results = self.forward_transformer(
            image_feats,
            query_points=query_points,
            motion_embed=motion_feats,
            latent_size=latent_size,
            ref_imgs_bool=ref_imgs_bool,
        )

        return (
            results["query_feats"],
            results["img_feats"],
            results["motion_embs"],
            results["pos_embs"],
        )

    def forward(
        self,
        image,
        source_c2ws,
        source_intrs,
        render_c2ws,
        render_intrs,
        render_bg_colors,
        smplx_params,
        **kwargs,
    ):
        """
        Forward pass of the ModelHumanA4OLRM.

        Args:
            image (torch.Tensor): Reference images of shape [B, N_ref, C_img, H_img, W_img].
            source_c2ws (torch.Tensor): Source camera-to-world matrices of shape [B, N_ref, 4, 4].
            source_intrs (torch.Tensor): Source camera intrinsics of shape [B, N_ref, 4, 4].
            render_c2ws (torch.Tensor): Render camera-to-world matrices of shape [B, N_source, 4, 4].
            render_intrs (torch.Tensor): Render camera intrinsics of shape [B, N_source, 4, 4].
            render_bg_colors (torch.Tensor): Background colors for rendering, shape [B, N_source, 3].
            smplx_params (dict): Dictionary containing SMPL-X parameters, e.g., "body_pose", "betas", etc.
            **kwargs: Additional keyword arguments (e.g., query_pts_path, src_head_imgs, ref_imgs_bool).

        Returns:
            tuple: Outputs from latent point processing and rendering.
        """

        assert (
            image.shape[0] == render_c2ws.shape[0]
        ), "Batch size mismatch for image and render_c2ws"
        assert (
            image.shape[0] == render_bg_colors.shape[0]
        ), "Batch size mismatch for image and render_bg_colors"
        assert (
            image.shape[0] == smplx_params["betas"].shape[0]
        ), "Batch size mismatch for image and smplx_params"
        assert (
            image.shape[0] == smplx_params["body_pose"].shape[0]
        ), "Batch size mismatch for image and smplx_params"
        assert len(smplx_params["betas"].shape) == 2

        render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(
            render_intrs[0, 0, 0, 2] * 2
        )
        query_points = None

        if kwargs["query_pts_path"] is not None:
            if self.latent_query_points_type.startswith(
                "e2e_smplx"
            ) or self.latent_query_points_type.startswith("e2e_points"):
                query_points, smplx_params = self.renderer.get_query_points(
                    kwargs["query_pts_path"][0],
                    smplx_params,
                    device=image.device,
                )
        else:
            query_points, smplx_params = self.renderer.get_query_points(
                None,
                smplx_params,
                device=image.device,
            )

        latent_points, image_feats, motion_emb, pos_emb = self.forward_latent_points(
            image,
            camera=None,
            query_points=query_points,
            ref_imgs_bool=kwargs.get("ref_imgs_bool", None),
        )  # [B, N, C]

        # render target views

        render_results = self.renderer(
            gs_hidden_features=latent_points,
            query_points=query_points,
            smplx_data=smplx_params,
            c2w=render_c2ws,
            intrinsic=render_intrs,
            height=render_h,
            width=render_w,
            background_color=render_bg_colors,
            additional_features={"image_feats": image_feats, "image": image[:, 0]},
            df_data=kwargs["df_data"],
            is_refine=False,
            patch_size=self.neural_renderer_patch_size,
        )

        # Neural renderer for image prediction
        if self.neural_renderer is not None:
            if self.neural_renderer_input == "feats":
                predict_rgbs, predict_masks = self.neural_renderer(
                    image_feats,
                    render_results["comp_features"],
                    motion_emb,
                    render_h,
                    render_w,
                    pos_emb,
                )
            elif self.neural_renderer_input == "rgb+feats":
                renderer_inputs = torch.cat(
                    [render_results["comp_features"], render_results["comp_rgb"]], dim=2
                )
                predict_rgbs, predict_masks = self.neural_renderer(
                    image_feats,
                    renderer_inputs,
                    motion_emb,
                    render_h,
                    render_w,
                    pos_emb,
                )
            else:
                raise NotImplementedError
            # [B V 3 H W]
            neural_comp_rgb = predict_rgbs.permute(0, 1, 4, 2, 3)
            # [B V 1 H W]
            neural_comp_mask = predict_masks.permute(0, 1, 4, 2, 3)
        else:
            neural_comp_rgb = None
            neural_comp_mask = None

        comp_rgb = render_results["comp_rgb"]
        comp_mask = render_results["comp_mask"]

        # gaussian attribute.
        gs_attrs_list = render_results.pop("gs_attr")
        offset_list = []
        scaling_list = []
        opacity_list = []
        for gs_attrs in gs_attrs_list:
            offset_list.append(gs_attrs.offset_xyz)
            scaling_list.append(gs_attrs.scaling)
            opacity_list.append(gs_attrs.opacity)

        offset_output = torch.stack(offset_list)
        scaling_output = torch.stack(scaling_list)
        opacity_output = torch.stack(opacity_list)

        return dict(
            comp_rgb=comp_rgb,
            comp_mask=comp_mask,
            neural_comp_rgb=neural_comp_rgb,
            neural_comp_mask=neural_comp_mask,
            offset_output=offset_output,
            scaling_output=scaling_output,
            opacity_output=opacity_output,
            mesh_meta=render_results["mesh_meta"],
        )

    def obtain_params(self, cfg):
        """
        Obtains parameters for the optimizer based on their type and requirements.

        Args:
            cfg (dict): Configuration dictionary containing training parameters.

        Returns:
            list: A list of parameter groups for the optimizer.
        """

        # add all bias and LayerNorm params to no_decay_params
        no_decay_params, decay_params = [], []

        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                no_decay_params.extend([p for p in module.parameters()])
            elif hasattr(module, "bias") and module.bias is not None:
                no_decay_params.append(module.bias)

        # add remaining parameters to decay_params
        _no_decay_ids = set(map(id, no_decay_params))
        decay_params = [p for p in self.parameters() if id(p) not in _no_decay_ids]

        # filter out parameters with no grad
        decay_params = list(filter(lambda p: p.requires_grad, decay_params))
        no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))

        # Optimizer
        opt_groups = [
            {
                "params": decay_params,
                "weight_decay": cfg.train.optim.weight_decay,
                "lr": cfg.train.optim.lr,
                "name": "decay",
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
                "lr": cfg.train.optim.lr,
                "name": "no_decay",
            },
        ]

        logger.info("======== Weight Decay Parameters ========")
        logger.info(f"Total: {len(decay_params)}")
        logger.info("======== No Weight Decay Parameters ========")
        logger.info(f"Total: {len(no_decay_params)}")

        print(f"Total Params: {len(no_decay_params) + len(decay_params)}")

        return opt_groups

    @torch.no_grad()
    def infer_single_view(
        self,
        image,
        source_c2ws,
        source_intrs,
        render_c2ws,
        render_intrs,
        render_bg_colors,
        smplx_params,
        query_pts_path=None,
        **kwargs,
    ):
        """
        Annotates the function for single-view inference in the ModelHumanA4OLRM class.

        Args:
            image (torch.Tensor): Shape [B, N_ref, C_img, H_img, W_img]. Input reference images.
            source_c2ws (torch.Tensor): [B, N_ref, 4, 4]. Camera-to-world matrices for references.
            source_intrs (torch.Tensor): [B, N_ref, 4, 4]. Intrinsic matrices for references.
            render_c2ws (torch.Tensor): [B, N_source, 4, 4]. Cam-to-world for rendering cameras.
            render_intrs (torch.Tensor): [B, N_source, 4, 4]. Intrinsics for rendering cameras.
            render_bg_colors (torch.Tensor): [B, N_source, 3]. BG color for rendering views.
            smplx_params (dict): Parametric human body params (pose, betas, etc).
            query_pts_path (str, optional): Path to query points file (if using predefined queries).
            **kwargs:
                ref_imgs_bool (optional): Boolean mask for reference images.

        Returns:
            gs_model_list: List of renderer outputs for the generated renders.
            query_points: Used query points, possibly e.g. SMPL-X surface or embedding.
            smplx_params: Possibly modified parametric inputs, after query point processing.
        """

        assert (
            image.shape[0] == render_c2ws.shape[0]
        ), "Batch size mismatch for image and render_c2ws"
        assert (
            image.shape[0] == render_bg_colors.shape[0]
        ), "Batch size mismatch for image and render_bg_colors"
        assert (
            image.shape[0] == smplx_params["betas"].shape[0]
        ), "Batch size mismatch for image and smplx_params"
        assert len(smplx_params["betas"].shape) == 2

        render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(
            render_intrs[0, 0, 0, 2] * 2
        )

        torch.cuda.synchronize()
        print("staring testing")
        query_points = None

        if self.latent_query_points_type.startswith(
            "e2e_smplx"
        ) or self.latent_query_points_type.startswith("e2e_points"):
            query_points, smplx_params = self.renderer.get_query_points(
                query_pts_path, smplx_params, device=image.device
            )

        time_calculator = AverageTimeCalculator()

        # ********************** Compute Reconstruction Time ********************
        torch.cuda.synchronize()
        start_time = time.time()
        latent_points, image_feats, motion_emb, pos_emb = self.forward_latent_points(
            image,
            camera=None,
            query_points=query_points,
            ref_imgs_bool=kwargs.get("ref_imgs_bool", None),
        )  # [B, N, C]

        self.renderer.hyper_step(10000000)  # set to max step
        gs_model_list, query_points, smplx_params = self.renderer.forward_gs(
            gs_hidden_features=latent_points,
            query_points=query_points,
            smplx_data=smplx_params,
            additional_features={"image_feats": image_feats, "image": image[:, 0]},
        )
        torch.cuda.synchronize()
        time_calculator.add_time(time.time() - start_time)
        print(time_calculator.average_time())

        # ********************** Compute Reconstruction Time ********************
        return (
            gs_model_list,
            query_points,
            smplx_params["transform_mat_neutral_pose"],
            latent_points,
            image_feats,
            motion_emb,
            pos_emb,
        )

    @torch.no_grad()
    def animation_infer(
        self,
        gs_model_list,
        query_points,
        smplx_params,
        render_c2ws,
        render_intrs,
        render_bg_colors,
        gs_hidden_features,
        image_latents,
        motion_emb,
        pos_emb=None,
        pad_forward=False,
        offset_list=None,
        mask_seqs=None,
        output_rgb=None,
        return_gs_alpha=False,
    ):
        """
        Perform animation inference for Gaussian Splatting-based human rendering.

        Args:
            gs_model_list: List of GS model instances to be rendered.
            query_points: Query points used for rendering (e.g., point cloud or mesh).
            smplx_params: SMPL-X parameters for pose and shape.
            render_c2ws: Camera-to-world transformation matrices, shape [batch_size, num_views, ...].
            render_intrs: Camera intrinsics for rendered views, shape [batch_size, num_views, ...].
            render_bg_colors: Background colors for rendered views.
            gs_hidden_features: Latent GS features (e.g., per point/vertex/patch features).
            image_latents: Latent image features for DPT decoder.
            motion_emb: Motion embedding or features.
            pos_emb (optional): Positional encoding, if used; defaults to None.
            pad_forward (bool, optional): If True, pad the input before forward pass. Defaults to False.
            offset_list (optional): List of offsets for animation, if any. Defaults to None.
            mask_seqs (optional): List or tensor of foreground masks for each view. Defaults to None.
            output_rgb (optional): If provided, stores the output RGBs. Defaults to None.

        Returns:
            Output(s) from DPT or renderer depending on configuration, e.g.:
                - List of rendered RGB images per view.
                - List of foreground/background masks per view.
                - Optionally, additional per-view meta-data or outputs for animation inference.

        Note:
            This function supports multi-view animation inference and optionally uses
            masks and DPT decoding, depending on the model configuration.
        """

        render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(
            render_intrs[0, 0, 0, 2] * 2
        )

        # render target views
        num_views = render_c2ws.shape[1]
        output = []
        output_mask = []
        # When return_gs_alpha is set, we also collect the raw gsplat
        # comp_mask (the per-pixel accumulated opacity from the Gaussian
        # rasterizer). This is the "true" alpha channel -- the
        # output_mask above carries the *neural renderer's* mask, which is
        # learned post-processing and does not necessarily resemble alpha.
        output_gs_alpha = []

        # DEBUG

        for view_idx in range(num_views):
            if mask_seqs is not None:
                mask = mask_seqs[view_idx]
                _render_h, _render_w = mask.shape

                render_intr_view = render_intrs[:, view_idx : view_idx + 1]
                intrs = render_intr_view

                render_results = self.renderer.forward_animate_gs(
                    gs_model_list,
                    query_points,
                    self.renderer.get_single_view_smpl_data(smplx_params, view_idx),
                    render_c2ws[:, view_idx : view_idx + 1],
                    intrs,
                    _render_h,
                    _render_w,
                    render_bg_colors[:, view_idx : view_idx + 1],
                    patch_size=self.neural_renderer_patch_size,
                    features=gs_hidden_features,
                )

                if self.neural_renderer_input == "feats":
                    # backup to ori space.
                    predict_rgbs, predict_masks = self.neural_renderer(
                        image_latents,
                        render_results["comp_features"],
                        motion_emb,
                        int(_render_h),
                        int(_render_w),
                        pos_emb_list=pos_emb,
                    )
                elif self.neural_renderer_input == "rgb+feats":
                    renderer_inputs = torch.cat(
                        [render_results["comp_features"], render_results["comp_rgb"]],
                        dim=2,
                    )
                    predict_rgbs, predict_masks = self.neural_renderer(
                        image_latents,
                        renderer_inputs,
                        motion_emb,
                        _render_h,
                        _render_w,
                        pos_emb_list=pos_emb,
                    )
                else:
                    raise NotImplementedError

                # predict_rgbs = predict_rgbs * predict_masks + (1-predict_masks)

                scale_x, scale_y, offset_x, offset_y = offset_list[view_idx]
                predict_rgb = (predict_rgbs[0, 0].detach().cpu().numpy() * 255).astype(
                    np.uint8
                )
                predict_rgb = cv2.resize(
                    predict_rgb,
                    (int(_render_w / scale_x), int(_render_h / scale_y)),
                    interpolation=cv2.INTER_AREA,
                )

                predict_mask = (
                    predict_masks[0, 0].detach().cpu().numpy() * 255
                ).astype(np.uint8)[..., 0]
                predict_mask = cv2.resize(
                    predict_mask,
                    (int(_render_w / scale_x), int(_render_h / scale_y)),
                    interpolation=cv2.INTER_NEAREST,
                )

                _h, _w = predict_rgb.shape[:2]
                boards = (output_rgb.clone().detach().cpu().numpy() * 255).astype(
                    np.uint8
                )
                boards[offset_y : offset_y + _h, offset_x : offset_x + _w] = predict_rgb

                mask_boards = np.zeros_like(boards[..., 0])
                mask_boards[offset_y : offset_y + _h, offset_x : offset_x + _w] = (
                    predict_mask
                )

                output.append(torch.from_numpy(boards / 255.0).float().unsqueeze(0))
                output_mask.append(
                    torch.from_numpy(mask_boards / 255.0).float().unsqueeze(0)
                )

                if return_gs_alpha:
                    # Raw gsplat alpha for this view (continuous, not the
                    # neural mask). Same crop+place pipeline as the rgb
                    # so it stays pixel-aligned to the rendered person.
                    gs_alpha = _gs_alpha_to_2d(render_results["comp_mask"])
                    gs_alpha = cv2.resize(
                        gs_alpha,
                        (int(_render_w / scale_x), int(_render_h / scale_y)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    gs_alpha_board = np.zeros(boards.shape[:2], dtype=np.float32)
                    gs_alpha_board[
                        offset_y : offset_y + _h, offset_x : offset_x + _w
                    ] = gs_alpha
                    output_gs_alpha.append(
                        torch.from_numpy(gs_alpha_board).float().unsqueeze(0)
                    )

            else:
                render_intr_view = render_intrs[:, view_idx : view_idx + 1]
                intrs = render_intr_view
                _render_h, _render_w = render_h, render_w

                render_results = self.renderer.forward_animate_gs(
                    gs_model_list,
                    query_points,
                    self.renderer.get_single_view_smpl_data(smplx_params, view_idx),
                    render_c2ws[:, view_idx : view_idx + 1],
                    intrs,
                    _render_h,
                    _render_w,
                    render_bg_colors[:, view_idx : view_idx + 1],
                    patch_size=self.neural_renderer_patch_size,
                    features=gs_hidden_features,
                )

                if self.neural_renderer_input == "feats":
                    # backup to ori space.
                    predict_rgbs, predict_masks = self.neural_renderer(
                        image_latents,
                        render_results["comp_features"],
                        motion_emb,
                        int(_render_h),
                        int(_render_w),
                        pos_emb_list=pos_emb,
                    )
                elif self.neural_renderer_input == "rgb+feats":
                    renderer_inputs = torch.cat(
                        [render_results["comp_features"], render_results["comp_rgb"]],
                        dim=2,
                    )
                    predict_rgbs, predict_masks = self.neural_renderer(
                        image_latents,
                        renderer_inputs,
                        motion_emb,
                        _render_h,
                        _render_w,
                        pos_emb_list=pos_emb,
                    )
                else:
                    raise NotImplementedError

                predict_rgb = (predict_rgbs[0, 0].detach().cpu().numpy() * 255).astype(
                    np.uint8
                )
                predict_mask = (
                    predict_masks[0, 0].detach().cpu().numpy() * 255
                ).astype(np.uint8)[..., 0]

                output.append(
                    torch.from_numpy(predict_rgb / 255.0).float().unsqueeze(0)
                )
                output_mask.append(
                    torch.from_numpy(predict_mask / 255.0).float().unsqueeze(0)
                )

                if return_gs_alpha:
                    gs_alpha = _gs_alpha_to_2d(render_results["comp_mask"])
                    output_gs_alpha.append(
                        torch.from_numpy(gs_alpha).float().unsqueeze(0)
                    )

            del render_results
            torch.cuda.empty_cache()

        pred_rgbs = torch.cat(output, dim=0)
        pred_masks = torch.cat(output_mask, dim=0)

        if return_gs_alpha:
            pred_gs_alphas = torch.cat(output_gs_alpha, dim=0)
            return pred_rgbs, pred_masks, pred_gs_alphas
        return pred_rgbs, pred_masks

    @torch.no_grad()
    def inference_gs(
        self,
        gs_model_list,
        query_points,
        smplx_params,
        render_c2ws,
        render_intrs,
        render_bg_colors,
        gs_hidden_features,
        pad_forward=False,
    ):
        """
        Perform inference on the Gaussian Splatting (GS) renderer for a given set of parameters.

        Args:
            gs_model_list (list): List of GS models for rendering.
            query_points (torch.Tensor): Query points used for rendering.
            smplx_params (dict): Dictionary containing SMPL-X parameters.
            render_c2ws (torch.Tensor): Render camera-to-world matrices of shape [B, N_views, 4, 4].
            render_intrs (torch.Tensor): Render camera intrinsics of shape [B, N_views, 4, 4].
            render_bg_colors (torch.Tensor): Background colors for rendering, shape [B, N_views, 3].
            gs_hidden_features (torch.Tensor): Latent features for GS models.
            pad_forward (bool, optional): Whether to pad the input prior to inference. Defaults to False.

        Returns:
            torch.Tensor: Output from the canonical GS inference for the first view.
        """

        render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(
            render_intrs[0, 0, 0, 2] * 2
        )
        # render target views
        view_idx = 0

        cano_gs = self.renderer.inference_cano_gs(
            gs_model_list,
            query_points,
            self.renderer.get_single_view_smpl_data(smplx_params, view_idx),
            render_c2ws[:, view_idx : view_idx + 1],
            render_intrs[:, view_idx : view_idx + 1],
            render_h,
            render_w,
            render_bg_colors[:, view_idx : view_idx + 1],
            features=gs_hidden_features,
        )

        return cano_gs[0]

    def _prepare_smplx_data(self, smplx_data):
        cano_keys = [
            "root_pose",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
            "expr",
            "trans",
        ]

        merge_data = {k: torch.zeros_like(smplx_data[k][0, 0:1]) for k in cano_keys}

        # Special handling for body pose
        if "body_pose" in merge_data:
            # leg
            merge_data["body_pose"][-1, 0, -1] = math.pi / 12
            merge_data["body_pose"][-1, 1, -1] = -math.pi / 12

            # hands
            merge_data["body_pose"][-1, 15, -1] = -math.pi / 6
            merge_data["body_pose"][-1, 16, -1] = math.pi / 6

        merge_data["betas"] = smplx_data["betas"]
        merge_data["transform_mat_neutral_pose"] = smplx_data[
            "transform_mat_neutral_pose"
        ]

        return merge_data
