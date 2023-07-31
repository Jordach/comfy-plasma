from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from io import BytesIO
from xml.dom import minidom
import math
import copy
import random
import torch
import numpy as np
import comfy
import math
import os
import hashlib
import requests
import json
# import piexif
# import piexif.helper

# Useful utilities

def remap(val, min_val, max_val, min_map, max_map):
	return (val-min_val)/(max_val-min_val) * (max_map-min_map) + min_map

def conv_pil_tensor(img):
	return (torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0),)

def conv_tensor_pil(tsr):
	return Image.fromarray(np.clip(255. * tsr.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def clamp(val, min, max):
	if val < min:
		return min
	elif val > max:
		return max
	else:
		return val

def get_pil_resampler(resampler):
	if resampler == "nearest":
		return Image.Resampling.NEAREST
	elif resampler == "box":
		return Image.Resampling.BOX
	elif resampler == "bilinear":
		return Image.Resampling.BILINEAR
	elif resampler == "bicubic":
		return Image.Resampling.BICUBIC
	elif resampler == "hamming":
		return Image.Resampling.HAMMING
	elif resampler == "lanczos":
		return Image.Resampling.LANCZOS
	else:
		return Image.Resampling.NEAREST

# borrowed from https://github.com/receyuki/stable-diffusion-prompt-reader/blob/master/sd_prompt_reader/image_data_reader.py
EASYDIFFUSION_MAPPING_A = {
	"prompt": "Prompt",
	"negative_prompt": "Negative Prompt",
	"seed": "Seed",
	"use_stable_diffusion_model": "Stable Diffusion model",
	"clip_skip": "Clip Skip",
	"use_vae_model": "VAE model",
	"sampler_name": "Sampler",
	"width": "Width",
	"height": "Height",
	"num_inference_steps": "Steps",
	"guidance_scale": "Guidance Scale",
}

EASYDIFFUSION_MAPPING_B = {
	"prompt": "prompt",
	"negative_prompt": "negative_prompt",
	"seed": "seed",
	"use_stable_diffusion_model": "use_stable_diffusion_model",
	"clip_skip": "clip_skip",
	"use_vae_model": "use_vae_model",
	"sampler_name": "sampler_name",
	"width": "width",
	"height": "height",
	"num_inference_steps": "num_inference_steps",
	"guidance_scale": "guidance_scale",
}

def handle_auto1111(params):
	if params and "\nSteps:" in params:
		# has a negative:
		if "Negative prompt:" in params:
			prompt_index = [params.index("\nNegative prompt:"), params.index("\nSteps:")]
			neg = params[prompt_index[0] + 1 + len("Negative prompt: "):prompt_index[-1]]
		else:
			index = [params.index("\nSteps:")]
			neg = ""

		pos = params[:prompt_index[0]]
		return pos, neg
	elif params:
		# has a negative:
		if "Negative prompt:" in params:
			prompt_index = [params.index("\nNegative prompt:")]
			neg = params[prompt_index[0] + 1 + len("Negative prompt: "):]
		else:
			index = [len(params)]
			neg = ""
		
		pos = params[:prompt_index[0]]
		return pos, neg
	else:
		return "", ""

def handle_ezdiff(params):
	data = json.loads(params)
	if data.get("prompt"):
		ed = EASYDIFFUSION_MAPPING_B
	else:
		ed = EASYDIFFUSION_MAPPING_A

	pos = data.get(ed["prompt"])
	data.pop(ed["prompt"])
	neg = data.get(ed["negative_prompt"])
	return pos, neg

def handle_invoke_modern(params):
	meta = json.loads(params.get("sd-metadata"))
	img = meta.get("image")
	prompt = img.get("prompt")
	index = [prompt.rfind("["), prompt.rfind("]")]

	# negative
	if -1 not in index:
		pos = prompt[:index[0]]
		neg = prompt[index[0] + 1:index[1]]
		return pos, neg
	else:
		return prompt, ""

def handle_invoke_legacy(params):
	dream = params.get("Dream")
	pi = dream.rfind('"')
	ni = [dream.rfind("["), dream.rfind("]")]

	# has neg
	if -1 not in ni:
		pos = dream[1:ni[0]]
		neg = dream[ni[0] + 1:ni[1]]
		return pos, neg
	else:
		pos = dream[1:pi]
		return pos, ""

def handle_novelai(params):
	pos = params.get("Description")
	comment = params.get("Comment") or {}
	comment_json = json.loads(comment)
	neg = comment_json.get("uc")
	return pos, neg

def handle_qdiffusion(params):
	pass

def handle_drawthings(params):
	try:
		data = minidom.parseString(params.get("XML:com.adobe.xmp"))
		data_json = json.loads(data.getElementByTagName("exif:UserComment")[0].childNodes[1].childNodes[1].childNodes[0].data)
	except:
		return "", ""
	else:
		pos = data_json.get("c")
		neg = data_json.get("uc")
		return pos, neg


class GreyScale:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"IMAGE": ("IMAGE",),
			}
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "process_image"

	CATEGORY = "image/postprocessing"

	def process_image(self, IMAGE):
		cimg = conv_tensor_pil(IMAGE)
		gimg = ImageOps.grayscale(cimg)
		rgbimg = Image.new("RGB", (gimg.width, gimg.height))
		rgbimg.paste(gimg)
		return conv_pil_tensor(rgbimg)

class Equalize:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"IMAGE": ("IMAGE",),
			}
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "process_image"

	CATEGORY = "image/postprocessing"

	def process_image(self, IMAGE):
		cimg = conv_tensor_pil(IMAGE)
		return conv_pil_tensor(ImageOps.equalize(cimg))

class AutoContrast:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"IMAGE": ("IMAGE",),
				"cutoff": ("FLOAT", {"default": 2, "min": 0, "max": 100, "step": 0.01}),
				"min_value": ("INT", {"default": -1, "min": -1, "max": 255, "step": 1})
			}
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "process_image"

	CATEGORY = "image/postprocessing"

	def process_image(self, IMAGE, cutoff, min_value):
		cimg = conv_tensor_pil(IMAGE)
		if min_value >= 0:
			return conv_pil_tensor(ImageOps.autocontrast(cimg, cutoff=cutoff, ignore=min_value))
		else:
			return conv_pil_tensor(ImageOps.autocontrast(cimg, cutoff=cutoff))

class ResizeFactor:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"IMAGE": ("IMAGE",),
				"factor": ("FLOAT", {"default": 2, "min": 0.01, "max": 10, "step": 0.01}),
				"resampler": (["nearest", "box", "bilinear", "bicubic", "hamming", "lanczos"],)
			}
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "process_image"

	CATEGORY = "image/postprocessing"

	def process_image(self, IMAGE, factor, resampler):
		cimg = conv_tensor_pil(IMAGE)
		(w, h) = (int(cimg.width * factor), int(cimg.height * factor))
		sampler = get_pil_resampler(resampler)
		return conv_pil_tensor(cimg.resize((w, h), resample=sampler))

class BlendImages:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"IMAGE_A": ("IMAGE",),
				"IMAGE_B": ("IMAGE",),
				"blend": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.001})
			}
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "process_image"

	CATEGORY = "image/postprocessing"

	def process_image(self, IMAGE_A, IMAGE_B, blend):
		# Convert from tensors
		source_a = conv_tensor_pil(IMAGE_A)
		source_b = conv_tensor_pil(IMAGE_B)
		aw, ah = source_a.width, source_a.height
		bw, bh = source_b.width, source_b.height

		# Convert image to RGB space
		img_a = Image.new("RGB", (aw, ah))
		img_a.paste(source_a)
		img_b = Image.new("RGB", (bw, bh))
		img_b.paste(source_b)
		
		# If img_b is not the same size as img_a - scale img_b to img_a dimensions.
		if ((aw != bw) or (ah != bh)):
			img_b.resize((aw, ah), resample=get_pil_resampler("lanczos"))

		# Finally, blend the two
		return conv_pil_tensor(Image.blend(img_a, img_b, blend))

class GaussianBlur:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"IMAGE": ("IMAGE",),
				"blur_radius": ("FLOAT", {"default": 1, "min": 1, "max": 1024, "step": 0.01})
			}
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "process_image"

	CATEGORY = "image/postprocessing"

	def process_image(self, IMAGE, blur_radius):
		img = conv_tensor_pil(IMAGE)

		# Finally, blend the two
		return conv_pil_tensor(img.filter(ImageFilter.GaussianBlur(blur_radius)))

class PowerImage:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"IMAGE": ("IMAGE",),
				"power_of": ("FLOAT", {"default": 1, "min": 1, "max": 65536, "step": 0.01}),
				"mode": (["darken", "lighten", "emphasize both"],),
			}
		}
	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "process_image"

	CATEGORY = "image/postprocessing"

	def process_image(self, IMAGE, power_of, mode):
		cimg = conv_tensor_pil(IMAGE)
		w, h = cimg.size

		pbar = comfy.utils.ProgressBar(h)
		step = 0
		for y in range(h):
			for x in range(w):
				r, g, b = cimg.getpixel((x, y))

				if mode == "lighten":
					r *= 1 + pow(r/255, power_of)
					g *= 1 + pow(g/255, power_of)
					b *= 1 + pow(b/255, power_of)
				elif mode == "emphasize both":
					r *= 0.5 + pow(r/255, power_of)
					g *= 0.5 + pow(g/255, power_of)
					b *= 0.5 + pow(b/255, power_of)
				else:
					r *= pow(r/255, power_of)
					g *= pow(g/255, power_of)
					b *= pow(b/255, power_of)

				r = clamp(r, 0, 255)
				g = clamp(g, 0, 255)
				b = clamp(b, 0, 255)

				cimg.putpixel((x, y), (int(r), int(g), int(b)))
			step += 1
			pbar.update_absolute(step, h)

		return conv_pil_tensor(cimg)

class ImageContrast:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"IMAGE": ("IMAGE",),
				"contrast": ("FLOAT", {"default": 1, "min": 0, "max": 10, "step": 0.01}),
				"brightness": ("FLOAT", {"default": 1, "min": 0, "max": 10, "step": 0.01}),
			}
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "process_image"

	CATEGORY = "image/postprocessing"

	def process_image(self, IMAGE, contrast, brightness):
		cimg = conv_tensor_pil(IMAGE)
		w, h = cimg.size

		pbar = comfy.utils.ProgressBar(2)
		step = 0
		cnt = ImageEnhance.Contrast(cimg)
		cimg = cnt.enhance(contrast)
		step += 1
		pbar.update_absolute(step, h)
		brt = ImageEnhance.Brightness(cimg)
		cimg = brt.enhance(brightness)
		step += 1
		pbar.update_absolute(step, h)

		return conv_pil_tensor(cimg)

class PlasmaNoise:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"width": ("INT", {
					"default": 512,
					"min": 128,
					"max": 8192,
					"step": 8
				}),
				"height": ("INT", {
					"default": 512,
					"min": 128,
					"max": 8192,
					"step": 8
				}),

				"turbulence": ("FLOAT", {
					"default": 2.75,
					"min": 0.5,
					"max": 32,
					"step": 0.01
				}),
				
				"value_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"value_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),

				"red_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"red_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"green_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"green_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"blue_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"blue_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				# Does nothing because ComfyUI doesn't understand "static" output nodes
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
			}
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "generate_plasma"
	CATEGORY = "image/noise"

	def generate_plasma(self, width, height, turbulence, value_min, value_max, red_min, red_max, green_min, green_max, blue_min, blue_max, seed):
		# Image size
		w = width
		h = height
		aw = copy.deepcopy(w)
		ah = copy.deepcopy(h)

		outimage = Image.new("RGB", (aw, ah))
		if w >= h:
			h = w
		else:
			w = h

		# Clamp per channel and globally
		clamp_v_min = value_min
		clamp_v_max = value_max
		clamp_r_min = red_min
		clamp_r_max = red_max
		clamp_g_min = green_min
		clamp_g_max = green_max
		clamp_b_min = blue_min
		clamp_b_max = blue_max

		roughness = turbulence
		pixmap = []

		random.seed(seed)
		def adjust(xa, ya, x, y, xb, yb):
			if(pixmap[x][y] == 0):
				d=math.fabs(xa-xb) + math.fabs(ya-yb)
				v=(pixmap[xa][ya] + pixmap[xb][yb])/2.0 + (random.random()-0.555) * d * roughness
				c=int(math.fabs(v + random.randint(-48, 48)))
				if c < 0:
					c = 0
				elif c > 255:
					c = 255
				pixmap[x][y] = c

		def subdivide(x1, y1, x2, y2):
			if(not((x2-x1 < 2.0) and (y2-y1 < 2.0))):
				x=int((x1 + x2)/2.0)
				y=int((y1 + y2)/2.0)
				adjust(x1,y1,x,y1,x2,y1)
				adjust(x2,y1,x2,y,x2,y2)
				adjust(x1,y2,x,y2,x2,y2)
				adjust(x1,y1,x1,y,x1,y2)
				if(pixmap[x][y] == 0):
					v=int((pixmap[x1][y1] + pixmap[x2][y1] + pixmap[x2][y2] + pixmap[x1][y2]) / 4.0)
					pixmap[x][y] = v

				subdivide(x1,y1,x,y)
				subdivide(x,y1,x2,y)
				subdivide(x,y,x2,y2)
				subdivide(x1,y,x,y2)
		pbar = comfy.utils.ProgressBar(4)
		step = 0

		pixmap = [[0 for i in range(h)] for j in range(w)]
		pixmap[0][0] = random.randint(0, 255)
		pixmap[w-1][0] = random.randint(0, 255)
		pixmap[w-1][h-1] = random.randint(0, 255)
		pixmap[0][h-1] = random.randint(0, 255)
		subdivide(0,0,w-1,h-1)
		r = copy.deepcopy(pixmap)

		step += 1
		pbar.update_absolute(step, 4)

		pixmap = [[0 for i in range(h)] for j in range(w)]
		pixmap[0][0] = random.randint(0, 255)
		pixmap[w-1][0] = random.randint(0, 255)
		pixmap[w-1][h-1] = random.randint(0, 255)
		pixmap[0][h-1] = random.randint(0, 255)
		subdivide(0,0,w-1,h-1)
		g = copy.deepcopy(pixmap)

		step += 1
		pbar.update_absolute(step, 4)

		pixmap = [[0 for i in range(h)] for j in range(w)]
		pixmap[0][0] = random.randint(0, 255)
		pixmap[w-1][0] = random.randint(0, 255)
		pixmap[w-1][h-1] = random.randint(0, 255)
		pixmap[0][h-1] = random.randint(0, 255)
		subdivide(0,0,w-1,h-1)
		b = copy.deepcopy(pixmap)

		step += 1
		pbar.update_absolute(step, 4)

		# Handle value clamps
		lv = 0
		mv = 0
		if clamp_v_min == -1:
			lv = 0
		else:
			lv = clamp_v_min

		if clamp_v_max == -1:
			mv = 255
		else:
			mv = clamp_v_max

		lr = 0
		mr = 0
		if clamp_r_min == -1:
			lr = lv
		else:
			lr = clamp_r_min

		if clamp_r_max == -1:
			mr = mv
		else:
			mr = clamp_r_max

		lg = 0
		mg = 0
		if clamp_g_min == -1:
			lg = lv
		else:
			lg = clamp_g_min

		if clamp_g_max == -1:
			mg = mv
		else:
			mg = clamp_g_max

		lb = 0
		mb = 0
		if clamp_b_min == -1:
			lb = lv
		else:
			lb = clamp_b_min

		if clamp_b_max == -1:
			mb = mv
		else:
			mb = clamp_b_max

		#print(f"V:{lv}/{mv}, R:{lr}/{mr}, G:{lg}/{mg}, B:{lb}/{mb}")
		for y in range(ah):
			for x in range(aw):
				nr = int(remap(r[x][y], 0, 255, lr, mr))
				ng = int(remap(g[x][y], 0, 255, lg, mg))
				nb = int(remap(b[x][y], 0, 255, lb, mb))
				outimage.putpixel((x,y), (nr, ng, nb))

		step += 1
		pbar.update_absolute(step, 4)
		return conv_pil_tensor(outimage)

class RandNoise:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"width": ("INT", {
					"default": 512,
					"min": 128,
					"max": 8192,
					"step": 8
				}),
				"height": ("INT", {
					"default": 512,
					"min": 128,
					"max": 8192,
					"step": 8
				}),

				"value_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"value_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),

				"red_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"red_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"green_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"green_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"blue_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"blue_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				# Does nothing because ComfyUI doesn't understand "static" output nodes
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
			}
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "generate_noise"
	CATEGORY = "image/noise"

	def generate_noise(self, width, height, value_min, value_max, red_min, red_max, green_min, green_max, blue_min, blue_max, seed):
		# Image size
		w = width
		h = height
		aw = copy.deepcopy(w)
		ah = copy.deepcopy(h)

		outimage = Image.new("RGB", (aw, ah))
		random.seed(seed)

		# Clamp per channel and globally
		clamp_v_min = value_min
		clamp_v_max = value_max
		clamp_r_min = red_min
		clamp_r_max = red_max
		clamp_g_min = green_min
		clamp_g_max = green_max
		clamp_b_min = blue_min
		clamp_b_max = blue_max

		# Handle value clamps
		lv = 0
		mv = 0
		if clamp_v_min == -1:
			lv = 0
		else:
			lv = clamp_v_min

		if clamp_v_max == -1:
			mv = 255
		else:
			mv = clamp_v_max

		lr = 0
		mr = 0
		if clamp_r_min == -1:
			lr = lv
		else:
			lr = clamp_r_min

		if clamp_r_max == -1:
			mr = mv
		else:
			mr = clamp_r_max

		lg = 0
		mg = 0
		if clamp_g_min == -1:
			lg = lv
		else:
			lg = clamp_g_min

		if clamp_g_max == -1:
			mg = mv
		else:
			mg = clamp_g_max

		lb = 0
		mb = 0
		if clamp_b_min == -1:
			lb = lv
		else:
			lb = clamp_b_min

		if clamp_b_max == -1:
			mb = mv
		else:
			mb = clamp_b_max

		pbar = comfy.utils.ProgressBar(ah)
		step = 0
		for y in range(ah):
			for x in range(aw):
				nr = random.randint(lr, mr)
				ng = random.randint(lg, mg)
				nb = random.randint(lb, mb)
				outimage.putpixel((x,y), (nr, ng, nb))
			step += 1
			pbar.update_absolute(step, ah)

		return conv_pil_tensor(outimage)

class GreyNoise:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"width": ("INT", {
					"default": 512,
					"min": 128,
					"max": 8192,
					"step": 8
				}),
				"height": ("INT", {
					"default": 512,
					"min": 128,
					"max": 8192,
					"step": 8
				}),
				
				"value_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"value_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),

				"red_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"red_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"green_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"green_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"blue_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"blue_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				# Does nothing because ComfyUI doesn't understand "static" output nodes
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
			}
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "generate_noise"
	CATEGORY = "image/noise"

	def generate_noise(self, width, height, value_min, value_max, red_min, red_max, green_min, green_max, blue_min, blue_max, seed):
		# Image size
		w = width
		h = height
		aw = copy.deepcopy(w)
		ah = copy.deepcopy(h)

		outimage = Image.new("RGB", (aw, ah))
		random.seed(seed)

		# Clamp per channel and globally
		clamp_v_min = value_min
		clamp_v_max = value_max
		clamp_r_min = red_min
		clamp_r_max = red_max
		clamp_g_min = green_min
		clamp_g_max = green_max
		clamp_b_min = blue_min
		clamp_b_max = blue_max

		# Handle value clamps
		lv = 0
		mv = 0
		if clamp_v_min == -1:
			lv = 0
		else:
			lv = clamp_v_min

		if clamp_v_max == -1:
			mv = 255
		else:
			mv = clamp_v_max

		lr = 0
		mr = 0
		if clamp_r_min == -1:
			lr = lv
		else:
			lr = clamp_r_min

		if clamp_r_max == -1:
			mr = mv
		else:
			mr = clamp_r_max

		lg = 0
		mg = 0
		if clamp_g_min == -1:
			lg = lv
		else:
			lg = clamp_g_min

		if clamp_g_max == -1:
			mg = mv
		else:
			mg = clamp_g_max

		lb = 0
		mb = 0
		if clamp_b_min == -1:
			lb = lv
		else:
			lb = clamp_b_min

		if clamp_b_max == -1:
			mb = mv
		else:
			mb = clamp_b_max

		pbar = comfy.utils.ProgressBar(ah)
		step = 0
		for y in range(ah):
			for x in range(aw):
				nv = random.randint(lv, mv)
				nr = int(remap(nv, lv, mv, lr, mr))
				ng = int(remap(nv, lv, mv, lg, mg))
				nb = int(remap(nv, lv, mv, lb, mb))
				outimage.putpixel((x,y), (nr, ng, nb))
			step += 1
			pbar.update_absolute(step, ah)

		return conv_pil_tensor(outimage)

class PinkNoise:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"width": ("INT", {
					"default": 512,
					"min": 128,
					"max": 8192,
					"step": 8
				}),
				"height": ("INT", {
					"default": 512,
					"min": 128,
					"max": 8192,
					"step": 8
				}),

				"value_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"value_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),

				"red_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"red_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"green_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"green_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"blue_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"blue_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				# Does nothing because ComfyUI doesn't understand "static" output nodes
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
			}
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "generate_noise"
	CATEGORY = "image/noise"

	def generate_noise(self, width, height, value_min, value_max, red_min, red_max, green_min, green_max, blue_min, blue_max, seed):
		# Image size
		w = width
		h = height
		aw = copy.deepcopy(w)
		ah = copy.deepcopy(h)

		outimage = Image.new("RGB", (aw, ah))
		random.seed(seed)

		# Clamp per channel and globally
		clamp_v_min = value_min
		clamp_v_max = value_max
		clamp_r_min = red_min
		clamp_r_max = red_max
		clamp_g_min = green_min
		clamp_g_max = green_max
		clamp_b_min = blue_min
		clamp_b_max = blue_max

		# Handle value clamps
		lv = 0
		mv = 0
		if clamp_v_min == -1:
			lv = 0
		else:
			lv = clamp_v_min

		if clamp_v_max == -1:
			mv = 255
		else:
			mv = clamp_v_max

		lr = 0
		mr = 0
		if clamp_r_min == -1:
			lr = lv
		else:
			lr = clamp_r_min

		if clamp_r_max == -1:
			mr = mv
		else:
			mr = clamp_r_max

		lg = 0
		mg = 0
		if clamp_g_min == -1:
			lg = lv
		else:
			lg = clamp_g_min

		if clamp_g_max == -1:
			mg = mv
		else:
			mg = clamp_g_max

		lb = 0
		mb = 0
		if clamp_b_min == -1:
			lb = lv
		else:
			lb = clamp_b_min

		if clamp_b_max == -1:
			mb = mv
		else:
			mb = clamp_b_max

		pbar = comfy.utils.ProgressBar(ah)
		step = 0
		for y in range(ah):
			for x in range(aw):
				nr = clamp(int(np.power(random.randint(lr, mr)/255, 1/3) * 255), 0, 255)
				ng = clamp(int(np.power(random.randint(lg, mg)/255, 1/3) * 255), 0, 255)
				nb = clamp(int(np.power(random.randint(lb, mb)/255, 1/3) * 255), 0, 255)
				outimage.putpixel((x,y), (nr, ng, nb))
			step += 1
			pbar.update_absolute(step, ah)

		return conv_pil_tensor(outimage)

class BrownNoise:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"width": ("INT", {
					"default": 512,
					"min": 128,
					"max": 8192,
					"step": 8
				}),
				"height": ("INT", {
					"default": 512,
					"min": 128,
					"max": 8192,
					"step": 8
				}),

				"value_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"value_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),

				"red_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"red_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"green_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"green_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"blue_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"blue_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				# Does nothing because ComfyUI doesn't understand "static" output nodes
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
			}
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "generate_noise"
	CATEGORY = "image/noise"

	def generate_noise(self, width, height, value_min, value_max, red_min, red_max, green_min, green_max, blue_min, blue_max, seed):
		# Image size
		w = width
		h = height
		aw = copy.deepcopy(w)
		ah = copy.deepcopy(h)

		outimage = Image.new("RGB", (aw, ah))
		random.seed(seed)

		# Clamp per channel and globally
		clamp_v_min = value_min
		clamp_v_max = value_max
		clamp_r_min = red_min
		clamp_r_max = red_max
		clamp_g_min = green_min
		clamp_g_max = green_max
		clamp_b_min = blue_min
		clamp_b_max = blue_max

		# Handle value clamps
		lv = 0
		mv = 0
		if clamp_v_min == -1:
			lv = 0
		else:
			lv = clamp_v_min

		if clamp_v_max == -1:
			mv = 255
		else:
			mv = clamp_v_max

		lr = 0
		mr = 0
		if clamp_r_min == -1:
			lr = lv
		else:
			lr = clamp_r_min

		if clamp_r_max == -1:
			mr = mv
		else:
			mr = clamp_r_max

		lg = 0
		mg = 0
		if clamp_g_min == -1:
			lg = lv
		else:
			lg = clamp_g_min

		if clamp_g_max == -1:
			mg = mv
		else:
			mg = clamp_g_max

		lb = 0
		mb = 0
		if clamp_b_min == -1:
			lb = lv
		else:
			lb = clamp_b_min

		if clamp_b_max == -1:
			mb = mv
		else:
			mb = clamp_b_max

		pbar = comfy.utils.ProgressBar(ah)
		step = 0
		for y in range(ah):
			for x in range(aw):
				nr = clamp(int(np.power(np.power(random.randint(lr, mr)/255, 1/3), 1/3) * 255), 0, 255)
				ng = clamp(int(np.power(np.power(random.randint(lg, mg)/255, 1/3), 1/3) * 255), 0, 255)
				nb = clamp(int(np.power(np.power(random.randint(lb, mb)/255, 1/3), 1/3) * 255), 0, 255)
				outimage.putpixel((x,y), (nr, ng, nb))
			step += 1
			pbar.update_absolute(step, ah)

		return conv_pil_tensor(outimage)

# Torch rand noise
def prepare_rand_noise(latent_image, seed, noise_inds=None):
	"""
	creates random noise given a latent image and a seed.
	optional arg skip can be used to skip and discard x number of noise generations for a given seed
	"""
	generator = torch.manual_seed(seed)
	if noise_inds is None:
		return (torch.rand(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu") - 0.5) * 2 * 1.73
	
	unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
	noises = []
	for i in range(unique_inds[-1]+1):
		noise = (torch.rand(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu") - 0.5) * 2 * 1.73
		if i in unique_inds:
			noises.append(noise)
	noises = [noises[i] for i in inverse]
	noises = torch.cat(noises, axis=0)
	return noises

# Modified ComfyUI sampler
def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, latent_noise, use_rand=False, start_step=None, last_step=None):
	device = comfy.model_management.get_torch_device()
	latent_image = latent["samples"]

	noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")

	if latent_noise > 0:
		batch_inds = latent["batch_index"] if "batch_index" in latent else None
		if use_rand:
			noise = noise + (prepare_rand_noise(latent_image, seed, batch_inds) * latent_noise)
		else:
			noise = noise + (comfy.sample.prepare_noise(latent_image, seed, batch_inds) * latent_noise)

	noise_mask = None
	if "noise_mask" in latent:
		noise_mask = latent["noise_mask"]

	pbar = comfy.utils.ProgressBar(steps)
	def callback(step, x0, x, total_steps):
		pbar.update_absolute(step + 1, total_steps)

	samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
								  denoise=denoise, disable_noise=True, start_step=start_step, last_step=last_step,
								  force_full_denoise=False, noise_mask=noise_mask, callback=callback)
	out = latent.copy()
	out["samples"] = samples
	return (out, )

class PlasmaSampler:
	@classmethod
	def INPUT_TYPES(s):
		return {
				"required":
					{
						"model": ("MODEL",),
						"noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
						"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
						"cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
						"denoise": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
						"latent_noise": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
						"distribution_type": (["default", "rand"],),
						"sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
						"scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
						"positive": ("CONDITIONING", ),
						"negative": ("CONDITIONING", ),
						"latent_image": ("LATENT", ),
					}
				}

	RETURN_TYPES = ("LATENT",)
	FUNCTION = "sample"
	CATEGORY = "sampling"

	def sample(self, model, noise_seed, steps, cfg, denoise, sampler_name, scheduler, positive, negative, latent_image, latent_noise, distribution_type):
		rand = False
		if distribution_type == "rand":
			rand = True
		return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, latent_noise, use_rand=rand)

class LoadImagePath:
	@classmethod
	def INPUT_TYPES(s):
		return {
				"required": 
					{
						"image": ("STRING", {"default": ""})
					}
				}

	CATEGORY = "image"

	RETURN_TYPES = ("IMAGE", "MASK")
	FUNCTION = "load_image"
	def load_image(self, image):
		# Removes any quotes from Explorer
		image_path = str(image)
		image_path = image_path.replace('"', "")
		i = None
		if image_path.startswith("http"):
			response = requests.get(image_path)
			i = Image.open(BytesIO(response.content)).convert("RGB")
		else:
			i = Image.open(image_path)
		i = ImageOps.exif_transpose(i)
		image = i.convert("RGB")
		image = np.array(image).astype(np.float32) / 255.0
		image = torch.from_numpy(image)[None,]
		if 'A' in i.getbands():
			mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
			mask = 1. - torch.from_numpy(mask)
		else:
			mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
		return (image, mask)

	@classmethod
	def IS_CHANGED(s, image):
		image_path = str(image)
		image_path = image_path.replace('"', "")
		m = hashlib.sha256()
		if not image_path.startswith("http"):
			with open(image_path, 'rb') as f:
				m.update(f.read())
			return m.digest().hex()
		else:
			m.update(image.encode("utf-8"))
			return m.digest().hex()

	@classmethod
	def VALIDATE_INPUTS(s, image):
		image_path = str(image)
		image_path = image_path.replace('"', "")
		if image_path.startswith("http"):
			return True
		if not os.path.isfile(image_path):
			return "No file found: {}".format(image_path)

		return True

class LoadImagePathWithMetadata:
	@classmethod
	def INPUT_TYPES(s):
		return {
				"required": 
					{
						"image": ("STRING", {"default": ""})
					}
				}

	CATEGORY = "image"

	RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "INT", "INT")
	RETURN_NAMES = ("image", "mask", "PROMPT", "NEGATIVE", "WIDTH", "HEIGHT")
	FUNCTION = "load_image"

	def load_image(self, image):
		# Removes any quotes from Explorer
		image_path = str(image)
		image_path = image_path.replace('"', "")
		i = None
		if image_path.startswith("http"):
			response = requests.get(image_path)
			i = Image.open(BytesIO(response.content)).convert("RGB")
		else:
			i = Image.open(image_path)
		prompt = ""
		negative = ""
		width = i.width
		height = i.height
		
		if i.format == "PNG":
			# auto1111
			if "parameters" in i.info:
				params = i.info.get("parameters")
				prompt, negative = handle_auto1111(params)

			# easy diffusion
			elif "negative_prompt" in i.info or "Negative Prompt" in i.info:
				params = str(i.info).replace("'", '"')
				prompt, negative = handle_ezdiff(params)
			# invokeai modern
			elif "sd-metadata" in i.info:
				prompt, negative = handle_invoke_modern(i.info)
			# legacy invokeai
			elif "Dream" in i.info:
				prompt, negative = handle_invoke_legacy(i.info)
			# novelai
			elif i.info.get("Software") == "NovelAI":
				prompt, negative = handle_novelai(i.info)
			# qdiffusion
			# elif ????:
			# drawthings (iPhone, iPad, macOS)
			elif "XML:com.adobe.xmp" in i.info:
				prompt, negative = handle_drawthings(i.info)
		
		# Removes EXIF rotation and other nonsense
		i = ImageOps.exif_transpose(i)
		image = i.convert("RGB")
		image = np.array(image).astype(np.float32) / 255.0
		image = torch.from_numpy(image)[None,]
		if 'A' in i.getbands():
			mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
			mask = 1. - torch.from_numpy(mask)
		else:
			mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
		return (image, mask, prompt, negative, width, height)

	@classmethod
	def IS_CHANGED(s, image):
		image_path = str(image)
		image_path = image_path.replace('"', "")
		m = hashlib.sha256()
		if not image_path.startswith("http"):
			with open(image_path, 'rb') as f:
				m.update(f.read())
			return m.digest().hex()
		else:
			m.update(image.encode("utf-8"))
			return m.digest().hex()

	@classmethod
	def VALIDATE_INPUTS(s, image):
		image_path = str(image)
		image_path = image_path.replace('"', "")
		if image_path.startswith("http"):
			return True
		if not os.path.isfile(image_path):
			return "No file found: {}".format(image_path)

		return True

NODE_CLASS_MAPPINGS = {
	"JDC_Plasma": PlasmaNoise,
	"JDC_RandNoise": RandNoise,
	"JDC_GreyNoise": GreyNoise,
	"JDC_PinkNoise": PinkNoise,
	"JDC_BrownNoise": BrownNoise,
	"JDC_PlasmaSampler": PlasmaSampler,
	"JDC_PowerImage": PowerImage,
	"JDC_Contrast": ImageContrast,
	"JDC_Greyscale": GreyScale,
	"JDC_EqualizeGrey": Equalize,
	"JDC_AutoContrast": AutoContrast,
	"JDC_ResizeFactor": ResizeFactor,
	"JDC_BlendImages": BlendImages,
	"JDC_GaussianBlur": GaussianBlur,
	"JDC_ImageLoader": LoadImagePath,
	"JDC_ImageLoaderMeta": LoadImagePathWithMetadata
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"JDC_Plasma": "Plasma Noise",
	"JDC_RandNoise": "Random Noise",
	"JDC_GreyNoise": "Greyscale Noise",
	"JDC_PinkNoise": "Pink Noise",
	"JDC_BrownNoise": "Brown Noise",
	"JDC_PlasmaSampler": "Plasma KSampler",
	"JDC_PowerImage": "Image To The Power Of",
	"JDC_Contrast": "Brightness & Contrast",
	"JDC_Greyscale": "RGB to Greyscale",
	"JDC_EqualizeGrey": "Equalize Histogram",
	"JDC_AutoContrast": "AutoContrast",
	"JDC_ResizeFactor": "Resize Image by Factor",
	"JDC_BlendImages": "Blend Images",
	"JDC_GaussianBlur": "Gaussian Blur",
	"JDC_ImageLoader": "Load Image From Path",
	"JDC_ImageLoaderMeta": "Load Image From Path With Meta"
}