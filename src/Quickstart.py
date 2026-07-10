from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")

input_image = load_image("/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/assets/example_input.png")
prompt = "remove degradation"

output_image = pipe(prompt, image=input_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
output_image.save("/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/assets/example_output.png")
print("ok")