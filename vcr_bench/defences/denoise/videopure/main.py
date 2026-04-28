import os
import time
import torch

from .pipe_defense import Video_Diffpure
from .schedule.scheduling_ddim import DDIMScheduler
from .schedule.scheduling_ddpm import DDPMScheduler
from .core.raft_arch import RAFT_SR
from .flow_net import Flow_models


class Denoiser(torch.nn.Module):
    def __init__(self, model, classifier, t, flow_models=None, denoise_type='ddim', classifier_name='i3d_resnet50'):
        super().__init__()
        self.model = model
        self.classifier = classifier
        self.t = t
        self.device = 'cuda'
        self.diffattack = False
        self.batch_size = 1
        self.flow_models = flow_models
        self.denoise_method = getattr(self, denoise_type)
        self.classifier_name = classifier_name

    def videopure(self, x):
        pipeline = self.model
        verbose = os.getenv("VIDEOPURE_VERBOSE", "1").strip().lower() not in {"0", "false", "no", "off"}
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.enable_vae_slicing()
        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=False)
        if verbose:
            print("[VideoPure] Stage 1/3: computing optical flow...", flush=True)
        t0 = time.time()
        flows = self.flow_models.compute_flows(x.clone())
        if verbose:
            print(f"[VideoPure] Stage 1/3 done in {time.time() - t0:.1f}s", flush=True)
            print("[VideoPure] Stage 2/3: temporal inversion...", flush=True)
        t1 = time.time()
        inv_latents = pipeline.invert_temporal(prompt='', videos=x, inversion_t=self.t)
        if verbose:
            print(f"[VideoPure] Stage 2/3 done in {time.time() - t1:.1f}s", flush=True)
            print("[VideoPure] Stage 3/3: denoising/sampling...", flush=True)
        t2 = time.time()
        output_image = pipeline.videopure(
            prompt='',
            video_latents=inv_latents,
            inversion_t=self.t,
            flows=flows,
            flow_model=self.flow_models,
        )
        if verbose:
            print(f"[VideoPure] Stage 3/3 done in {time.time() - t2:.1f}s", flush=True)
            print(f"[VideoPure] Total denoise time: {time.time() - t0:.1f}s", flush=True)
        return output_image.detach()

    def ddim_inversion(self, x):
        pipeline = self.model
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.enable_vae_slicing()
        inv_latents = pipeline.invert(prompt='', videos=x, inversion_t=self.t)
        return pipeline.ddim_inversion(prompt='', video_latents=inv_latents, inversion_t=self.t).detach()

    def ddim(self, x):
        pipeline = self.model
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.enable_vae_slicing()
        return pipeline.ddim(prompt='', video_latents=x, inversion_t=self.t).detach()

    def ddpm(self, x):
        pipeline = self.model
        pipeline.enable_vae_slicing()
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        return pipeline.ddpm(prompt='', video_latents=x, inversion_t=self.t * 20).detach()

    def forward(self, x):
        x_pure = self.denoise_method(x)
        output = self.classifier(x_pure)
        return output, x_pure

    def classify(self, x):
        return self.classifier(x)
