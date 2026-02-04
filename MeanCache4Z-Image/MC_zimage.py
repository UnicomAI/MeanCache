import torch
from diffusers import ZImagePipeline
from diffusers.pipelines.z_image.pipeline_z_image import *
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import re, sys, os, argparse, time

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


@torch.no_grad()
def meancache_inference(
    self,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 5.0,
    cfg_normalization: bool = False,
    cfg_truncation: float = 1.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[List[torch.FloatTensor]] = None,
    negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
):
    height = height or 1024
    width = width or 1024

    vae_scale = self.vae_scale_factor * 2
    if height % vae_scale != 0:
        raise ValueError(
            f"Height must be divisible by {vae_scale} (got {height}). "
            f"Please adjust the height to a multiple of {vae_scale}."
        )
    if width % vae_scale != 0:
        raise ValueError(
            f"Width must be divisible by {vae_scale} (got {width}). "
            f"Please adjust the width to a multiple of {vae_scale}."
        )

    device = self._execution_device

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False
    self._cfg_normalization = cfg_normalization
    self._cfg_truncation = cfg_truncation
    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = len(prompt_embeds)

    # If prompt_embeds is provided and prompt is None, skip encoding
    if prompt_embeds is not None and prompt is None:
        if self.do_classifier_free_guidance and negative_prompt_embeds is None:
            raise ValueError(
                "When `prompt_embeds` is provided without `prompt`, "
                "`negative_prompt_embeds` must also be provided for classifier-free guidance."
            )
    else:
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.in_channels

    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        torch.float32,
        device,
        generator,
        latents,
    )

    # Repeat prompt_embeds for num_images_per_prompt
    if num_images_per_prompt > 1:
        prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
        if self.do_classifier_free_guidance and negative_prompt_embeds:
            negative_prompt_embeds = [npe for npe in negative_prompt_embeds for _ in range(num_images_per_prompt)]

    actual_batch_size = batch_size * num_images_per_prompt
    image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)

    # 5. Prepare timesteps
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    self.scheduler.sigma_min = 0.0
    scheduler_kwargs = {"mu": mu}
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)


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
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000
            # Normalized time for time-aware config (0 at start, 1 at end)
            t_norm = timestep[0].item()

            # Handle cfg truncation
            current_guidance_scale = self.guidance_scale
            if (
                self.do_classifier_free_guidance
                and self._cfg_truncation is not None
                and float(self._cfg_truncation) <= 1
            ):
                if t_norm > self._cfg_truncation:
                    current_guidance_scale = 0.0

            # Run CFG only if configured AND scale is non-zero
            apply_cfg = self.do_classifier_free_guidance and current_guidance_scale > 0
            

            if apply_cfg:
                latents_typed = latents.to(self.transformer.dtype)
                latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
                timestep_model_input = timestep.repeat(2)
            else:
                latent_model_input = latents.to(self.transformer.dtype)
                prompt_embeds_model_input = prompt_embeds
                timestep_model_input = timestep

            latent_model_input = latent_model_input.unsqueeze(2)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))
            
            # ------------------cache-------------------------
            should_calc = self.should_calc_list[i]
            if not should_calc:
                if self.cache_jvp:                   
                    jvp_cur_step = compute_current_jvp(log_dict, log_dict["v"], steps=self.edge_order[i-1]).squeeze(0)
                    # jvp_cur_step = compute_current_jvp(log_dict, log_dict["v"], steps=3).squeeze(0)

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
                model_out_list = self.transformer(
                    latent_model_input_list, timestep_model_input, prompt_embeds_model_input, return_dict=False
                )[0]

                if apply_cfg:
                    # Perform CFG
                    pos_out = model_out_list[:actual_batch_size]
                    neg_out = model_out_list[actual_batch_size:]

                    noise_pred = []
                    for j in range(actual_batch_size):
                        pos = pos_out[j].float()
                        neg = neg_out[j].float()

                        pred = pos + current_guidance_scale * (pos - neg)

                        # Renormalization
                        if self._cfg_normalization and float(self._cfg_normalization) > 0.0:
                            ori_pos_norm = torch.linalg.vector_norm(pos)
                            new_pos_norm = torch.linalg.vector_norm(pred)
                            max_new_norm = ori_pos_norm * float(self._cfg_normalization)
                            if new_pos_norm > max_new_norm:
                                pred = pred * (max_new_norm / new_pos_norm)

                        noise_pred.append(pred)

                    noise_pred = torch.stack(noise_pred, dim=0)
                else:
                    noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

                noise_pred = noise_pred.squeeze(2)
                noise_pred = -noise_pred

                log_dict["latents_pre"].append(latents.detach())
                log_dict["timesteps"].append(t.item()) 
                log_dict["sigmas_pre"].append(self.scheduler.sigmas[i].detach())
                log_dict["sigmas"].append(self.scheduler.sigmas[i+1].detach())
                log_dict["v"].append(noise_pred.detach())              

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
            assert latents.dtype == torch.float32
            log_dict["latents"].append(latents.to(torch.float32).detach()) 

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    if output_type == "latent":
        image = latents

    else:
        latents = latents.to(self.vae.dtype)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return ZImagePipelineOutput(images=image)


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
        25: [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 22, 29, 37, 42, 45, 47, 48, 49],
        20: [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 30, 38, 43, 46, 48, 49],
        17: [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 19, 27, 35, 43, 46, 48, 49],
        15: [0, 1, 2, 3, 5, 7, 10, 12, 17, 25, 33, 41, 45, 48, 49],
        13: [0, 1, 2, 3, 5, 9, 15, 22, 29, 36, 43, 47, 49]
    }
    
    edge_source = {
        25: ['chain', 'chain', 'chain', 'chain', 'chain', 'chain', 'v_diff_mean_jvp1_s2', 'v_diff_mean_jvp1_s2', 'chain', 'chain', 'chain', 'v_diff_mean_jvp1_s2', 'chain', 'chain', 'chain', 'v_diff_mean_jvp1_s5', 'v_diff_mean_jvp1_s5', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean_jvp1_s5', 'v_diff_mean_jvp1_s5', 'chain', 'chain'],
        20: ['chain', 'chain', 'chain', 'chain', 'v_diff_mean_jvp1_s2', 'v_diff_mean_jvp1_s2', 'v_diff_mean_jvp1_s2', 'v_diff_mean_jvp1_s2', 'v_diff_mean_jvp1_s2', 'v_diff_mean_jvp1_s5', 'v_diff_mean_jvp1_s4', 'v_diff_mean_jvp1_s5', 'v_diff_mean_jvp1_s5', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean_jvp1_s5', 'v_diff_mean_jvp1_s2', 'chain'],
        17: ['chain', 'chain', 'chain', 'chain', 'v_diff_mean_jvp1_s2', 'v_diff_mean_jvp1_s2', 'v_diff_mean_jvp1_s2', 'v_diff_mean_jvp1_s2', 'v_diff_mean_jvp1_s2', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean_jvp1_s5', 'v_diff_mean_jvp1_s2', 'chain'],
        15: ['chain', 'chain', 'chain', 'v_diff_mean_jvp1_s2', 'v_diff_mean_jvp1_s2', 'v_diff_mean_jvp1_s2', 'v_diff_mean_jvp1_s2', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean_jvp1_s5', 'v_diff_mean_jvp1_s4', 'chain'],
        13: ['chain', 'chain', 'chain', 'v_diff_mean_jvp1_s2', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean', 'v_diff_mean_jvp1_s5', 'v_diff_mean_jvp1_s2']
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

    
    # Load the pipeline
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")    


    if is_cache_enabled:
        ZImagePipeline.__call__ = meancache_inference
        ZImagePipeline.should_calc_list = bool_list
        ZImagePipeline.edge_order = result_edge_order
        ZImagePipeline.cache_jvp = True
        print(f"[INFO] Running with {cache_status_str}")
    else:
        print("[INFO] Running with nocache")

    # Generate image
    prompt = "两名年轻亚裔女性紧密站在一起，背景为朴素的灰色纹理墙面，可能是室内地毯地面。左侧女性留着长卷发，身穿藏青色毛衣，左袖有奶油色褶皱装饰，内搭白色立领衬衫，下身白色裤子；佩戴小巧金色耳钉，双臂交叉于背后。右侧女性留直肩长发，身穿奶油色卫衣，胸前印有“Tun the tables”字样，下方为“New ideas”，搭配白色裤子；佩戴银色小环耳环，双臂交叉于胸前。两人均面带微笑直视镜头。照片，自然光照明，柔和阴影，以藏青、奶油白为主的中性色调，休闲时尚摄影，中等景深，面部和上半身对焦清晰，姿态放松，表情友好，室内环境，地毯地面，纯色背景。"
    negative_prompt = ""

    start_time = time.time()
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=1280,
        width=720,
        cfg_normalization=False,
        num_inference_steps=50,
        guidance_scale=4,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).images[0]  
    
    end_time = time.time()
    print(f"Time: {end_time - start_time:.3f} s")

    print(cache_status_str)
    image.save(f"image_{cache_status_str}.png")    

if __name__ == "__main__":
    main()





