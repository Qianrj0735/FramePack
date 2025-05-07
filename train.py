import os
os.environ['HF_TOKEN'] = "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"
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers_helper.hf_login import login
from transformers import (
    LlamaTokenizerFast, LlamaModel,
    CLIPTokenizer, CLIPTextModel,
    SiglipImageProcessor, SiglipVisionModel
)
from diffusers import AutoencoderKLHunyuanVideo
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.hunyuan import vae_encode
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.utils import resize_and_center_crop
from datasets import load_dataset
import numpy as np
import wandb
import skvideo.io
from datetime import datetime
import argparse
from PIL import Image

class AlohaPenDataset(Dataset):
    """
    Loads video sequences from the "physical-intelligence/aloha_pen_uncap_diverse" dataset
    and pairs them with a fixed text prompt.
    """
    def __init__(self, split='train', prompt='uncap the pen'):
        self.data = load_dataset('physical-intelligence/aloha_pen_uncap_diverse', split=split)
        self.prompt = prompt

    def __len__(self):
        return len(self.data['timestamp'])

    def __getitem__(self, idx):
        frames_np = self.data[idx]['observation.images.cam_high']# list of HxWx3 numpy arrays
        frames_np = np.asarray(frames_np)
        return frames_np, self.prompt


def collate_fn(batch):
    videos, prompts = zip(*batch)
    return list(videos), list(prompts)


def train(
    epochs=10,
    batch_size=1,
    lr=2e-5,
    log_root='./logs',
    hf_token=None,
    device=None
):
    # Authenticate with HF for model downloads if required
    # hf_token should be provided or set in the HF_TOKEN env var
    token = hf_token or os.getenv('HF_TOKEN')
    if not token:
        raise ValueError("HuggingFace token not provided. Set HF_TOKEN environment variable or pass --hf_token.")
    login(token)

    # Set device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Create run-specific directories
    run_name = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    run_dir = os.path.join(log_root, run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize W&B logging
    wandb.init(
        project='FramePackTraining', name=run_name,
        config={'epochs': epochs, 'batch_size': batch_size, 'lr': lr},
        dir=run_dir
    )

    # Load models
    llama_tokenizer = LlamaTokenizerFast.from_pretrained('hunyuanvideo-community/HunyuanVideo', subfolder='tokenizer')
    llama_encoder = LlamaModel.from_pretrained('hunyuanvideo-community/HunyuanVideo', subfolder='text_encoder', torch_dtype=torch.float16).to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained('hunyuanvideo-community/HunyuanVideo', subfolder='tokenizer_2')
    clip_encoder = CLIPTextModel.from_pretrained('hunyuanvideo-community/HunyuanVideo', subfolder='text_encoder_2', torch_dtype=torch.float16).to(device)
    vae = AutoencoderKLHunyuanVideo.from_pretrained('hunyuanvideo-community/HunyuanVideo', subfolder='vae', torch_dtype=torch.float16).to(device)
    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
    transformer.train()

    # Freeze encoders
    for model in (llama_encoder, clip_encoder, vae, image_encoder):
        for p in model.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr)

    # Prepare dataset & loader
    dataset = AlohaPenDataset(split='train')
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )

    # Save a few raw training examples for visualization
    num_vis = 3
    for i in range(min(num_vis, len(dataset))):
        frames_np, prompt = dataset[i]
        img = Image.fromarray(frames_np)  # first frame
        img.save(os.path.join(run_dir, f'train_sample_{i}_frame0.png'))

    total_steps = 0
    for epoch in range(1, epochs+1):
        for step, (videos_np, prompts) in enumerate(loader, start=1):
            total_steps += 1
            frames_np, prompt = videos_np[0], prompts[0]

            # Text embeddings
            ll = llama_tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
            cl = clip_tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
            with torch.no_grad():
                llama_emb = llama_encoder(ll.input_ids.to(device))[0]
                clip_emb = clip_encoder(cl.input_ids.to(device))[0]

            # Preprocess frames
            H, W, _ = frames_np[0].shape
            H2, W2 = find_nearest_bucket(H, W, resolution=640)
            resized = [resize_and_center_crop(f, W2, H2) for f in frames_np]
            vid = torch.from_numpy(np.stack(resized)).permute(0,3,1,2).float().to(device)/127.5 - 1

            # Latent & vision features
            start_lat = vae_encode(vid[0:1], vae)
            img_feats = hf_clip_vision_encode(resized[0], feature_extractor, image_encoder).last_hidden_state.to(device)
            target_lat = vae_encode(vid[1:], vae)

            # Diffusion step
            ts = torch.randint(0, transformer.config.num_train_timesteps, (1,), device=device)
            noise = torch.randn_like(target_lat)
            noisy = transformer.scheduler.add_noise(target_lat, noise, ts)
            pred = transformer(noisy, ts, prompt_embeds=llama_emb, prompt_poolers=clip_emb, image_embeddings=img_feats)

            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_steps % 10 == 0:
                print(f"Epoch{epoch} Step{step}/{len(loader)} Loss{loss.item():.4f}")
            wandb.log({'train/loss': loss.item()}, step=total_steps)

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"epoch{epoch}.pt")
        torch.save(transformer.state_dict(), ckpt_path)
        wandb.save(ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # Inference & save sample video (MP4)
        transformer.eval()
        with torch.no_grad():
            frames_np, prompt = dataset[0]
            H, W, _ = frames_np[0].shape
            H2, W2 = find_nearest_bucket(H, W, resolution=640)
            resized = [resize_and_center_crop(f, W2, H2) for f in frames_np]
            vid = torch.from_numpy(np.stack(resized)).permute(0,3,1,2).float().to(device)/127.5 - 1
            start_lat = vae_encode(vid[0:1], vae)
            img_feats = hf_clip_vision_encode(resized[0], feature_extractor, image_encoder).last_hidden_state.to(device)

            ts = torch.arange(0, transformer.config.num_train_timesteps, transformer.config.num_train_timesteps//25, device=device)
            latents = start_lat.unsqueeze(2)
            for t in ts:
                noise_pred = transformer(latents, t.unsqueeze(0), prompt_embeds=llama_emb, prompt_poolers=clip_emb, image_embeddings=img_feats)
                latents = transformer.scheduler.step(noise_pred, t, latents)

            decoded = vae.decode(latents.squeeze(2)).cpu().permute(0,2,3,1).numpy()*127.5+127.5
            decoded = decoded.astype(np.uint8)
            out_path = os.path.join(run_dir, f"sample_epoch{epoch}.mp4")
            skvideo.io.vwrite(out_path, decoded, outputdict={'-r':'30'})
            print(f"Saved sample video: {out_path}")
        transformer.train()

    print("Training complete.")
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--log_root', type=str, default='./logs')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace token for model authentication')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_root=args.log_root,
        hf_token=args.hf_token,
        device=args.device
    )
