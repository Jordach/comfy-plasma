# comfy-plasma
A simple plasma noise generator for ComfyUI. Other noise generators may appear over time.

# Usage:
## Plasma Noise:
![Example](images/plasma_node.png)

* Width, Height: Sets the generated image size as desired - steps in increments of 8.
* Turbulence: Scales the noise clouds, lower values result in smoother, larger clouds, while higher values result in more static like noise.
* Value Min/Max: Limits how dark/bright the noise can reach. -1 implies 0 and 255 for Min and Max respectively.
* Red Min/Max: Limits how strong the red channel can be. -1 Will use the settings of Value Min/Max instead of it's own setting.
* Green Min/Max: Limits how strong the green channel can be. -1 Will use the settings of Value Min/Max instead of it's own setting.
* Blue Min/Max: Limits how strong the blue channel can be. -1 Will use the settings of Value Min/Max instead of it's own setting.
* Seed: Only there so ComfyUI will use the node to generate a new image, instead of locking up and refusing to work. (Should you want to reuse a previous noise - use a save image node.)

____
## KSampler Plasma:
## ***WARNING: This node does not add latent noise to the image and is entirely dependant on the noise from the Plasma Noise node. It otherwise works exactly like any other KSampler in ComfyUI.***

![Example](images/ksampler_node.png)
____
# Example Workflow:
An example to use Plasma Noise as a replacement noise for txt2img is as follows:

![How to use it](images/example.png)