from PIL import Image
import math
import copy
import random
import torch
import numpy as np
import comfy

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
	CATEGORY = "image"

	def generate_plasma(self, width, height, turbulence, value_min, value_max, red_min, red_max, green_min, green_max, blue_min, blue_max, seed=None):
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

		def remap(val, min_val, max_val, min_map, max_map):
			return (val-min_val)/(max_val-min_val) * (max_map-min_map) + min_map

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
		return (torch.from_numpy(np.array(outimage).astype(np.float32) / 255.0).unsqueeze(0),)

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

NODE_CLASS_MAPPINGS = {
	"JDC_Plasma": PlasmaNoise,
	"JDC_PlasmaSampler": PlasmaSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"JDC_Plasma": "Plasma Noise",
	"JDC_PlasmaSampler": "Plasma KSampler"
}