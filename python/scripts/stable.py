#!/usr/bin/env python
"""
LoRA fine-tuning for Stable-Diffusion with Tora logging.

Adapted from Hugging Face `train_text_to_image_lora.py` (Apache-2.0)
and your original HF-Trainer template.
"""

import os
import time
import math
import argparse
import random
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from tora import Tora

from datasets import load_dataset
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    AutoencoderKL,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

from accelerate import Accelerator, DistributedDataParallelKwargs


# ========= shared helpers unchanged =========
def safe_value(value):
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    elif isinstance(value, bool):
        return int(value)
    elif isinstance(value, str):
        return None
    else:
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


def log_metric(client, name, value, step):
    value = safe_value(value)
    if value is not None:
        client.log(name=name, value=value, step=step)


# ========= dataset handling =========
class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, resolution):
        self.dataset = hf_dataset
        self.resolution = resolution
        self.image_tfm = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        image = self.image_tfm(image)
        caption = example.get("caption", example.get("text", ""))
        return {"pixel_values": image, "captions": caption}


# ========= LoRA model loader =========
def load_sd_lora_model(config, torch_dtype):
    pipe = StableDiffusionPipeline.from_pretrained(
        config["model_name"],
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()

    lora_config = LoraConfig(
        r=config["rank"],
        lora_alpha=config["rank"] * 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=config["lora_dropout"],
        bias="none",
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.train()

    for p in pipe.unet.base_model.parameters():
        p.requires_grad = False

    return pipe


# ========= training loop =========
def train(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with=None, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    raw_ds = load_dataset(
        config["dataset_name"], split="train", cache_dir=config.get("cache_dir")
    )
    train_ds = ImageTextDataset(raw_ds, config["resolution"])
    train_dl = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    dtype = torch.float16 if config["mixed_precision"] == "fp16" else torch.float32
    pipe = load_sd_lora_model(config, dtype).to(device)
    text_tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    trainable_params = (p for p in pipe.unet.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(trainable_params, lr=config["lr"])
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=opt,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["epochs"]
        * math.ceil(len(train_dl) / config["grad_accum"]),
    )

    tora = Tora.create_experiment(
        name=config["experiment_name"],
        description=config["experiment_description"],
        hyperparams=config,
        tags=config.get("tags", []),
    )
    tora_step = 0

    pipe.unet, opt, train_dl, lr_scheduler = accelerator.prepare(
        pipe.unet, opt, train_dl, lr_scheduler
    )

    global_step = 0
    loss_fn = nn.MSELoss()
    for epoch in range(config["epochs"]):
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(pipe.unet):
                input_ids = text_tokenizer(
                    batch["captions"],
                    padding="max_length",
                    truncation=True,
                    max_length=text_tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.to(device)
                encoder_hidden_states = text_encoder(input_ids)[0]

                noise = torch.randn_like(batch["pixel_values"]).to(device, dtype)
                timesteps = torch.randint(
                    0,
                    pipe.scheduler.num_timesteps,
                    (batch["pixel_values"].shape[0],),
                    device=device,
                ).long()

                latents = pipe.vae.encode(
                    batch["pixel_values"].to(device, dtype)
                ).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                model_pred = pipe.unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                loss = loss_fn(model_pred.float(), noise.float()) / config["grad_accum"]
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    opt.step()
                    lr_scheduler.step()
                    opt.zero_grad()
                    global_step += 1

                    log_metric(tora, "train_loss", loss.item(), global_step)
                    log_metric(tora, "lr", lr_scheduler.get_last_lr()[0], global_step)

            if global_step % config["save_steps"] == 0:
                if accelerator.is_main_process:
                    adapter_path = os.path.join(
                        config["output_dir"], f"checkpoint-{global_step}"
                    )
                    accelerator.unwrap_model(pipe.unet).save_pretrained(adapter_path)
                accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            with torch.no_grad():
                prompt = config["eval_prompt"]
                image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[
                    0
                ]
                img_path = os.path.join(
                    config["output_dir"], f"sample_epoch_{epoch}.png"
                )
                image.save(img_path)
                log_metric(tora, "eval_sample_saved", epoch, global_step)

    if accelerator.is_main_process:
        final_dir = os.path.join(config["output_dir"], "final_adapter")
        accelerator.unwrap_model(pipe.unet).save_pretrained(final_dir)
        print(f"[✓] LoRA adapter saved to {final_dir}")

    tora.shutdown()
    accelerator.print("Training complete!")


# ========= CLI =========
def parse_args():
    parser = argparse.ArgumentParser(
        description="Stable-Diffusion LoRA training with Tora logging"
    )
    parser.add_argument("--dataset_name", type=str, default="conceptual_captions")
    parser.add_argument(
        "--model_name", type=str, default="runwayml/stable-diffusion-v1-5"
    )
    parser.add_argument("--output_dir", type=str, default="sd_lora_output")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mixed_precision", type=str, choices=["no", "fp16"], default="fp16"
    )
    parser.add_argument(
        "--eval_prompt",
        type=str,
        default="a scenic landscape with mountains and lake, realistic style",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = vars(args)
    cfg.update(
        {
            "experiment_name": f"LORA_SD_{cfg['model_name'].split('/')[-1]}",
            "experiment_description": f"LoRA (r={cfg['rank']}) fine-tuning of {cfg['model_name']} on {cfg['dataset_name']}",
            "tags": ["stable-diffusion", "lora", "diffusers"],
            "save_steps": 100,
        }
    )
    os.makedirs(cfg["output_dir"], exist_ok=True)
    train(cfg)
