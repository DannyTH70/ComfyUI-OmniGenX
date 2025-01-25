import sys
import folder_paths
import os.path as osp
import os
import torch
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
import tempfile
import shutil

# 定义路径
now_dir = osp.dirname(__file__)
models_dir = folder_paths.models_dir
omnigenx_dir = osp.join(models_dir, "OmniGenX")
omnigen_dir = osp.join(omnigenx_dir, "OmniGen-v1")
tmp_dir = osp.join(now_dir, "tmp")

sys.path.append(now_dir)
from OmniGen import OmniGenPipeline 

class LoadOmniGen:
    def __init__(self):
        self.pipe = None  # 添加成员变量存储 pipeline
        if not osp.exists(osp.join(omnigen_dir, "model.safetensors")):
            snapshot_download("Shitao/OmniGen-v1", local_dir=omnigen_dir)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Input latent tensor / 输入潜空间张量"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "tooltip": 
                    """提示词格式说明 / Prompt Format Guide:
                    1. 引用图片 / Reference Image: image_1
                    2. 图像编辑 / Image Editing: image_1 change the color to blue
                    3. 风格迁移 / Style Transfer: image_1 in the style of Van Gogh
                    4. 图像混合 / Image Mixing: image_1 mixed with image_2
                    5. 图像变体 / Image Variation: image_1 create variations
                    6. 纯文本生成 / Text-to-Image: a beautiful landscape painting"""
                    
                }),
                "num_inference_steps": ("INT", {
                    "default": 50, 
                    "min": 1, 
                    "max": 100, 
                    "step": 1,
                    "tooltip": "Number of denoising steps / 去噪步数"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 2.5, 
                    "min": 1.0, 
                    "max": 5.0, 
                    "step": 0.1,
                    "tooltip": "Text guidance scale / 文本引导程度"
                }),
                "img_guidance_scale": ("FLOAT", {
                    "default": 1.6, 
                    "min": 1.0, 
                    "max": 2.0, 
                    "step": 0.1,
                    "tooltip": "Image guidance scale / 图像引导程度"
                }),
                "max_input_image_size": ("INT", {
                    "default": 1024, 
                    "min": 128, 
                    "max": 2048, 
                    "step": 8,
                    "tooltip": "Maximum size for input images / 输入图像的最大尺寸"
                }),
                "match_input_size": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Automatically adjust output image size to match input image / 将输出图像大小自动调整为与输入图像相同"
                }),
                "seed": ("INT", {
                    "default": 666,
                    "tooltip": "Random seed for generation / 生成用的随机种子"
                }),
                "store_in_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep model in VRAM between generations. Faster but uses more VRAM / 在生成之间将模型保留在显存中。速度更快但使用更多显存"
                }),
                "memory_optimize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable memory optimization mode to reduce VRAM usage / 启用内存优化模式，分开处理不同的引导过程，可以减少显存占用"
                }),
                "cpu_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Load model to CPU to significantly reduce VRAM usage but slower / 将模型加载到CPU中，显著减少显存占用但会降低生成速度"
                }),
            
            },
            "optional": {
                "image_1": ("IMAGE", {
                    "tooltip": "First input reference image / 第一张输入参考图像"
                }),
                "image_2": ("IMAGE", {
                    "tooltip": "Second input reference image / 第二张输入参考图像"
                }),
                "image_3": ("IMAGE", {
                    "tooltip": "Third input reference image / 第三张输入参考图像"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "OmniGenX"

    def save_temp_image(self, image):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=tmp_dir) as f:
            img_np = image.numpy()[0] * 255
            img_pil = Image.fromarray(img_np.astype(np.uint8))
            img_pil.save(f.name)
            return f.name

    def generate(self, latent, prompt, num_inference_steps, guidance_scale, 
                img_guidance_scale, max_input_image_size, store_in_vram,
                memory_optimize,  
                cpu_offload,     
                match_input_size,  
                seed, image_1=None, image_2=None, image_3=None):
        try:
            # 使用 latent 获取尺寸
            height = latent["samples"].shape[2] * 8
            width = latent["samples"].shape[3] * 8
            
            # 获取或创建 pipeline
            if self.pipe is None or not store_in_vram:
                self.pipe = OmniGenPipeline.from_pretrained(omnigen_dir)
            
            # 根据 cpu_offload 设置正确配置
            if cpu_offload:
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.disable_model_cpu_offload()

            # 处理输入图像
            input_images = []
            os.makedirs(tmp_dir, exist_ok=True)

            if image_1 is not None:
                input_images.append(self.save_temp_image(image_1))
                prompt = prompt.replace("image_1", "<img><|image_1|></img>")
            
            if image_2 is not None:
                input_images.append(self.save_temp_image(image_2))
                prompt = prompt.replace("image_2", "<img><|image_2|></img>")
            
            if image_3 is not None:
                input_images.append(self.save_temp_image(image_3))
                prompt = prompt.replace("image_3", "<img><|image_3|></img>")

            if len(input_images) == 0:
                input_images = None
            print(prompt)
            # 移除回调函数，直接调用 pipe
            output = self.pipe(
                prompt=prompt,
                input_images=input_images,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                img_guidance_scale=img_guidance_scale,
                num_inference_steps=num_inference_steps,
                separate_cfg_infer=memory_optimize,
                offload_model=cpu_offload,
                use_input_image_size_as_output=match_input_size,
                seed=seed,
                max_input_image_size=max_input_image_size,
                use_kv_cache=True,          # 用 key-value 缓存
                offload_kv_cache=True,      # 允许将 key-value 缓存卸载到 CPU
            )

            # 转换输出格式
            img = np.array(output[0]) / 255.0
            img = torch.from_numpy(img).unsqueeze(0)
            
            # 清理资源
            if not store_in_vram:
                del self.pipe
                self.pipe = None
                torch.cuda.empty_cache()
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
                os.makedirs(tmp_dir, exist_ok=True)

            return (img,)
            
        except Exception as e:
            print(f"Error in generate: {str(e)}")
            raise e

NODE_CLASS_MAPPINGS = {
    "LoadOmniGen": LoadOmniGen
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadOmniGen": "Load OmniGen Model"
}
