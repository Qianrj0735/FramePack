import os

os.environ["HF_HOME"] = os.path.abspath(
    os.path.realpath(os.path.join(os.path.dirname(__file__), "./hf_download"))
)
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers_helper.hf_login import login
from transformers import (
    LlamaTokenizerFast,
    LlamaModel,
    CLIPTokenizer,
    CLIPTextModel,
    SiglipImageProcessor,
    SiglipVisionModel,
)
from diffusers import AutoencoderKLHunyuanVideo, DDPMScheduler
from diffusers_helper.models.hunyuan_video_packed import (
    HunyuanVideoTransformer3DModelPacked,
)
from diffusers_helper.hunyuan import vae_encode, encode_prompt_conds, vae_decode
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.utils import resize_and_center_crop, crop_or_pad_yield_mask
from datasets import load_dataset
import numpy as np
import wandb
from datetime import datetime
import argparse
from PIL import Image
import skvideo.io
from tqdm import tqdm


class FramePackDataset(Dataset):
    """
    原始帧迭代：对于每个 idx，取 idx-seq_len+1 … idx 这 seq_len 帧，
    跨 episode 边界或越界时，用本 episode 的第一帧补齐。
    """

    def __init__(
        self,
        hf_repo: str,
        split: str = "train",
        seq_len: int = 19 + 1 + 9,
        prompt: str = "a robotic arm performing a task",
    ):
        self.data = load_dataset(hf_repo, split=split)
        self.seq_len = seq_len
        self.prompt = prompt

        # 记录每条数据的 episode 起点索引
        self.episode_idx = self.data["episode_index"]
        self.episode_start = [0] * len(self.data)
        current_start = 0
        for i in range(len(self.data)):
            if i == 0 or self.episode_idx[i] != self.episode_idx[i - 1]:
                current_start = i
            self.episode_start[i] = current_start

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ep_start = self.episode_start[idx]
        frames = []
        for offset in range(idx - self.seq_len + 1, idx + 1):
            j = offset if offset >= ep_start else ep_start
            img: Image.Image = self.data[j]["observation.images.cam_high"]
            img = img.resize((640, 608))
            frames.append(np.array(img))
        seq_np = np.stack(frames, axis=0)  # (T, H, W, 3)
        return seq_np[:19], seq_np[19:20], seq_np[20:], self.prompt


def collate_fn(batch):
    his, start, future, prompts = zip(*batch)
    return np.array(his), np.array(start), np.array(future), list(prompts)


def train(
    hf_repo: str,
    epochs: int = 10,
    batch_size: int = 1,
    lr: float = 2e-5,
    seq_len: int = 19,
    log_root: str = "./logs",
    hf_token: str = None,
    device: str = None,
):
    # 1) 登录 Hugging Face
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("请提供 HF_TOKEN 或 --hf_token")
    login(token)

    # 2) 设备选择
    # device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cuda"
    # 3) 目录 & W&B 初始化
    run_name = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_dir = os.path.join(log_root, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    wandb.init(
        project="FramePackTraining",
        name=run_name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seq_len": seq_len,
        },
        dir=run_dir,
    )

    # 4) 加载模型与组件
    repo = "hunyuanvideo-community/HunyuanVideo"
    llama_tokenizer = LlamaTokenizerFast.from_pretrained(repo, subfolder="tokenizer")
    llama_encoder = LlamaModel.from_pretrained(
        repo, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer_2")
    clip_encoder = CLIPTextModel.from_pretrained(
        repo, subfolder="text_encoder_2", torch_dtype=torch.float16
    ).to(device)
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        repo, subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    feature_extractor = SiglipImageProcessor.from_pretrained(
        "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
    )
    image_encoder = SiglipVisionModel.from_pretrained(
        "lllyasviel/flux_redux_bfl",
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    ).to(device)

    transformer = (
        HunyuanVideoTransformer3DModelPacked.from_pretrained(
            "lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16
        )
        .to(device)
        .train()
    )

    # 5) 冻结非训练模块
    for m in (llama_encoder, clip_encoder, vae, image_encoder):
        for p in m.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr)
    scheduler = DDPMScheduler(beta_schedule="scaled_linear", num_train_timesteps=1000)

    # 6) 数据集 & DataLoader
    dataset = FramePackDataset(hf_repo, split="train", seq_len=19 + 1 + 9)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # 7) 保存部分训练样本以供可视化
    vis_dir = os.path.join(run_dir, "train_samples")
    os.makedirs(vis_dir, exist_ok=True)
    num_vis = min(5, len(dataset))
    # for i in range(num_vis):
    #     seq_np, _ = dataset[i]
    # for t, frame in enumerate(seq_np):
    #     img = Image.fromarray(frame)
    #     img.save(os.path.join(vis_dir, f"sample_{i}_frame{t}.png"))

    total_steps = 0
    for epoch in range(1, epochs + 1):
        with tqdm(loader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            for step, (his, start, future, prompts) in enumerate(pbar, start=1):
                total_steps += 1

                # 文本条件编码
                llama_vec, clip_pooler = encode_prompt_conds(
                    prompts[0],
                    llama_encoder,
                    clip_encoder,
                    llama_tokenizer,
                    clip_tokenizer,
                )
                llama_vec = llama_vec.to(device)
                llama_vec, llama_mask = crop_or_pad_yield_mask(llama_vec, length=512)
                llama_mask = llama_mask.to(device)
                clip_pooler = clip_pooler.to(device)

                # 视频帧 → VAE 潜空间
                vids = []
                his_vids = []
                future_vids = []
                img_feats = []
                s0 = start[0]
                H, W = s0.shape[1], s0.shape[2]
                H2, W2 = find_nearest_bucket(H, W, resolution=640)
                for s, h, ft in zip(start, his, future):
                    resized = [resize_and_center_crop(f, W2, H2) for f in s]
                    vid = torch.stack(
                        [torch.from_numpy(f).permute(2, 0, 1) for f in resized], dim=1
                    )
                    vids.append(vid)

                    his_resized = [resize_and_center_crop(f, W2, H2) for f in h]
                    his_vid = torch.stack(
                        [torch.from_numpy(f).permute(2, 0, 1) for f in his_resized],
                        dim=1,
                    )
                    his_vids.append(his_vid)

                    future_resized = [resize_and_center_crop(f, W2, H2) for f in ft]
                    future_vid = torch.stack(
                        [torch.from_numpy(f).permute(2, 0, 1) for f in future_resized],
                        dim=1,
                    )
                    future_vids.append(future_vid)
                    # resizeds.append(resized)
                    img_feat = hf_clip_vision_encode(
                        resized[0], feature_extractor, image_encoder
                    ).last_hidden_state.to(device)
                    img_feats.append(img_feat)
                llama_vec = llama_vec.repeat_interleave(len(vids), dim=0)
                llama_mask = llama_mask.repeat_interleave(len(vids), dim=0)

                img_feats = torch.cat(img_feats, dim=0)

                vid = torch.stack(vids)
                vid = (vid.to(device).float() / 127.5) - 1
                his_vid = torch.stack(his_vids)
                his_vid = (his_vid.to(device).float() / 127.5) - 1
                his_vid = his_vid.flip(2)
                future_vid = torch.stack(future_vids)
                future_vid = (future_vid.to(device).float() / 127.5) - 1
                start_latent = vae_encode(vid, vae)
                history_latents = torch.zeros_like(start_latent).repeat_interleave(
                    his_vid.size(2), dim=2
                )
                future_latents = torch.zeros_like(start_latent).repeat_interleave(
                    future_vid.size(2), dim=2
                )

                horizon = random.randint(0, 19)
                for hrz in range(horizon):
                    history_latents[:, :, hrz : hrz + 1, :, :] = vae_encode(
                        his_vid[:, :, hrz : hrz + 1, :, :], vae
                    )
                for ft_hrz in range(9):
                    future_latents[:, :, ft_hrz : ft_hrz + 1, :, :] = vae_encode(
                        future_vid[:, :, ft_hrz : ft_hrz + 1, :, :], vae
                    )
                # history_latents = vae_encode(his_vid, vae)  # (B,C,T',h,w)

                B, C, T, h, w = history_latents.shape
                latent_window_size = 9
                latent_padding = random.randint(0, 3)
                latent_padding_size = latent_padding * latent_window_size
                indices = torch.arange(
                    0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])
                ).unsqueeze(0)
                (
                    clean_latent_indices_pre,
                    blank_indices,
                    latent_indices,
                    clean_latent_indices_post,
                    clean_latent_2x_indices,
                    clean_latent_4x_indices,
                ) = indices.split(
                    [1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1
                )
                clean_latent_indices = torch.cat(
                    [clean_latent_indices_pre, clean_latent_indices_post], dim=1
                )

                clean_latents_pre = start_latent.to(history_latents)
                clean_latents_post, clean_latents_2x, clean_latents_4x = (
                    history_latents[:, :, : 1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                )
                clean_latents = torch.cat(
                    [clean_latents_pre, clean_latents_post], dim=2
                )

                # CLIP 视觉特征

                # DDPM 加噪
                rnd = torch.Generator("cuda").manual_seed(3407)
                num_frames = latent_window_size * 4 - 3
                t = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device)
                noise = torch.randn(
                    (
                        B,
                        16,
                        (num_frames + 3) // 4,
                        H2 // 8,
                        W2 // 8,
                    ),
                    generator=rnd,
                    device=rnd.device,
                ).to(device=device, dtype=torch.float32)

                noisy_latents = scheduler.add_noise(future_latents, noise, t)

                # 前向 + 损失
                gs = 10.0
                distilled_guidance = torch.full((B,), gs * 1000.0, device=device)
                out = transformer(
                    hidden_states=noisy_latents.to(torch.bfloat16),
                    timestep=t,
                    encoder_hidden_states=llama_vec.to(torch.bfloat16),
                    encoder_attention_mask=llama_mask,
                    pooled_projections=clip_pooler.to(torch.bfloat16),
                    guidance=distilled_guidance.to(torch.bfloat16),
                    latent_indices=latent_indices,
                    clean_latents=clean_latents.to(torch.bfloat16),
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x.to(torch.bfloat16),
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x.to(torch.bfloat16),
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    image_embeddings=img_feats.to(torch.bfloat16),
                )
                pred_noise = out.sample
                loss = F.mse_loss(pred_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if total_steps % 10 == 0:
                    print(
                        f"[{epoch}/{epochs}] step {step}/{len(loader)}  loss={loss.item():.4f}"
                    )
                wandb.log({"train/loss": loss.item()}, step=total_steps)

        # 保存 checkpoint
        ckpt = os.path.join(ckpt_dir, f"epoch{epoch}.pt")
        torch.save(transformer.state_dict(), ckpt)
        wandb.save(ckpt)
        print(f"Saved checkpoint: {ckpt}")

        # 在每个 epoch 末尾做 inference
        transformer.eval()
        with torch.no_grad():
            seq_np, prompt = dataset[0]
            llama_vec_inf, llama_mask_inf, clip_pooler_inf, _ = encode_prompt_conds(
                prompt[0], llama_encoder, clip_encoder, llama_tokenizer, clip_tokenizer
            )
            llama_vec_inf = llama_vec_inf.to(device)
            llama_mask_inf = llama_mask_inf.to(device)
            clip_pooler_inf = clip_pooler_inf.to(device)

            # 用首帧启动
            input_img = seq_np[0]
            H, W = input_img.shape[0], input_img.shape[1]
            H2, W2 = find_nearest_bucket(H, W, resolution=640)
            img_resized = resize_and_center_crop(input_img, W2, H2)
            img_tensor = (
                torch.from_numpy(img_resized).permute(2, 0, 1)[None].float() / 127.5 - 1
            ).to(device)

            init_latent = vae_encode(img_tensor, vae)
            img_feats_inf = hf_clip_vision_encode(
                img_resized, feature_extractor, image_encoder
            ).last_hidden_state.to(device)

            gen_latents = sample_hunyuan(
                transformer=transformer,
                sampler="unipc",
                initial_latent=init_latent,
                concat_latent=None,
                strength=1.0,
                width=W2,
                height=H2,
                frames=seq_len,
                real_guidance_scale=1.0,
                distilled_guidance_scale=gs,
                guidance_rescale=0.0,
                num_inference_steps=25,
                generator=torch.Generator(device).manual_seed(42),
                prompt_embeds=llama_vec_inf,
                prompt_embeds_mask=llama_mask_inf,
                prompt_poolers=clip_pooler_inf,
                negative_prompt_embeds=torch.zeros_like(llama_vec_inf),
                negative_prompt_embeds_mask=torch.zeros_like(llama_mask_inf),
                negative_prompt_poolers=torch.zeros_like(clip_pooler_inf),
                device=device,
                image_embeddings=img_feats_inf,
            )

            pixels = vae_decode(gen_latents, vae).cpu()
            pixels = (
                (pixels * 127.5 + 127.5).permute(0, 2, 3, 4, 1).numpy().astype(np.uint8)
            )
            out_path = os.path.join(run_dir, f"epoch{epoch}_sample.mp4")
            skvideo.io.vwrite(out_path, pixels, outputdict={"-r": "30"})
            print(f"Saved inference video: {out_path}")

        transformer.train()

    print("Training complete.")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="RUnia/IO_teleop_one_episode_piper_example",
        help="Hugging Face 数据集仓库名",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seq_len", type=int, default=19, help="1+2+16 多分辨率帧数")
    parser.add_argument("--log_root", type=str, default="./logs")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        hf_repo=args.hf_repo,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_len=args.seq_len,
        log_root=args.log_root,
        hf_token=args.hf_token,
        device=args.device,
    )
