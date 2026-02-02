import torch
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from diffusers.pipelines.qwenimage import QwenImagePipeline, QwenImagePipelineOutput
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import re, sys, os, argparse
import pandas as pd

def compute_current_jvp(log_dict, u_all_1step, steps: int = 2, dtype=torch.float32):
    L = len(log_dict["v"])
    if L == 0:
        return None
    actual_steps = min(steps, L)

    i1 = L - 1
    i0 = L - actual_steps

    x_start = log_dict["latents_pre"][i0].to(dtype)
    x_end = log_dict["latents"][i1].to(dtype)
    s_start = log_dict["sigmas_pre"][i0].to(dtype)
    s_end = log_dict["sigmas"][i1].to(dtype)
    denom = s_end - s_start
    denom_b = denom.view(-1, *([1] * (x_start.ndim - denom.dim())))
    avg_u = (x_end - x_start) / denom_b
    inst_u = u_all_1step[i0].to(dtype)
    jvp = (inst_u - avg_u) / denom_b

    return jvp

def make_safe_stem(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s)

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@torch.no_grad()
def meancache_inference(
    self,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    true_cfg_scale: float = 4.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 1.0,
    num_images_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
):
  
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
    )
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    prompt_embeds, prompt_embeds_mask = self.encode_prompt(
        prompt=prompt,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    if do_true_cfg:
        negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
            prompt=negative_prompt,
            prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=negative_prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4

    
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    img_shapes = [(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)] * batch_size

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # handle guidance
    if self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    if self.attention_kwargs is None:
        self._attention_kwargs = {}

    log_dict = {
        "timesteps": [],
        "sigmas_pre": [],
        "sigmas": [],
        "noise_pred": [],
        "neg_noise_pred": [],
        "v": [],
        "latents_pre":[],
        "latents": []
    }
        
    # 6. Denoising loop
    self.scheduler.set_begin_index(0)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            
            if self.interrupt:
                continue

            self._current_timestep = t
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            
            # ------------------cache-------------------------
            should_calc = self.should_calc_list[i]
            if not should_calc:
                
                # jvp cache
                if self.cache_jvp:                   
                    jvp_cur_step = compute_current_jvp(log_dict, log_dict["v"], steps=self.edge_order[i-1]).squeeze(0)
                    jvp_pred = jvp_cur_step.to(latents.dtype)
                    
                    s_start = self.scheduler.sigmas[i]
                    s_end = self.scheduler.sigmas[i+1]
                    
                    denom   = s_end - s_start
                    if denom.numel() == 1:
                        denom_b = denom
                    else:
                        denom_b = denom
                        while denom_b.dim() < x_start.dim():
                            denom_b = denom_b.unsqueeze(-1)
                    
                    v_mean_from_jvp = log_dict["v"][-1].to(latents.device, latents.dtype) -  jvp_pred * denom_b 
                    noise_pred = v_mean_from_jvp
                
                log_dict["latents_pre"].append(latents.detach())
                log_dict["timesteps"].append(t.item()) 
                log_dict["sigmas_pre"].append(self.scheduler.sigmas[i].detach())
                log_dict["sigmas"].append(self.scheduler.sigmas[i+1].detach())
                log_dict["v"].append(noise_pred.detach())                

                
            else:
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_prompt_embeds_mask.sum(dim=1).tolist(),
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]

                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)
                    
                    log_dict["latents_pre"].append(latents.detach())
                    log_dict["timesteps"].append(t.item()) 
                    log_dict["sigmas_pre"].append(self.scheduler.sigmas[i].detach())
                    log_dict["sigmas"].append(self.scheduler.sigmas[i+1].detach())
                    log_dict["v"].append(noise_pred.detach())                


            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            log_dict["latents"].append(latents.to(torch.float32).detach()) 
            
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
    

    self._current_timestep = None
    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return QwenImagePipelineOutput(images=image)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # If --cache-jvp is provided without a value, it defaults to 20.
    # If not provided at all, it is None.
    parser.add_argument("--cache-jvp", type=int, nargs='?', const=20, default=None)
    parser.add_argument("--step", type=int, default=50)   
    parser.add_argument("--seed", type=int, default=0)       
    
    return parser.parse_args()


def main():

    args = get_args()

    mapping_rules = {
        'v_diff_mean': 1,
        'v_diff_mean_jvp1_s2': 2,
        'v_diff_mean_jvp1_s3': 3,
        'v_diff_mean_jvp1_s4': 4,
        'v_diff_mean_jvp1_s5': 5,
        'v_diff_mean_jvp1_s6': 6,
        'chain': 0,
    }
    
    calc_dict = {
        25: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 23, 31, 37, 44, 45, 46, 47, 48, 49],
        17: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 19, 27, 35, 43, 47, 49],
        10: [0, 2, 4, 7, 14, 21, 28, 35, 42, 49],
    }
    
    edge_source = {
        25: ['chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'v_diff_mean_jvp1_s2', 'v_diff_mean_jvp1_s2', 'v_diff_mean_jvp1_s2', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean_jvp1_s5', 'v_diff_mean_jvp1_s5', 'chain'],
        17: ['chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'v_diff_mean_jvp1_s2', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean_jvp1_s5', 'v_diff_mean_jvp1_s5'],
        10: ['v_diff_mean', 'v_diff_mean_jvp1_s3', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean_jvp1_s4'],
    }
    
    bool_list = [False] * args.step
    cache_status_str = "nocache"
    is_cache_enabled = False

    if args.cache_jvp is not None:
        is_cache_enabled = True
        # Default to 20 if the input step is not in calc_dict
        target_step = args.cache_jvp if args.cache_jvp in calc_dict else 20
        cache_status_str = f"JVP_cache_{target_step}"

        for i in calc_dict[target_step]:
            if i < args.step:
                bool_list[i] = True
                
        result_edge_order = [0] * 50 
        edge_rule = edge_source[target_step]
        edge_order = [mapping_rules.get(rule) for rule in edge_rule]        
        
        for i in range(len(calc_dict[target_step]) - 1):
            start = calc_dict[target_step][i]
            end = calc_dict[target_step][i + 1]

            for pos in range(start, end):
                result_edge_order[pos] = edge_order[i]

        assert len(result_edge_order) == len(bool_list), f"Edge rules ({len(result_edge_order)}) != bool_list steps ({bool_list})"

    model_name = "Qwen/Qwen-Image-2512"

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    pipe = QwenImagePipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    if is_cache_enabled:
        QwenImagePipeline.__call__ = meancache_inference
        QwenImagePipeline.should_calc_list = bool_list
        QwenImagePipeline.edge_order = result_edge_order
        QwenImagePipeline.cache_jvp = True
        print(f"[INFO] Running with {cache_status_str}")
    else:
        print("[INFO] Running with nocache")

    # Pipeline inputs
    positive_magic = {
        "en": "Ultra HD, 4K, cinematic composition.",
        "zh": "超清，4K，电影级构图"
    }

    prompt = '''A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming with vivid, fresh, youthful charm.'''

    negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"

    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }
    width, height = aspect_ratios["16:9"]
        
    start_time = time.time()

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=args.step,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cuda").manual_seed(42)
    ).images[0]
    
    end_time = time.time()

    print(cache_status_str)
    print(f"Time: {end_time - start_time:.3f} s")

    image.save(f"image_{cache_status_str}_2512.png")


if __name__ == "__main__":
    main()
