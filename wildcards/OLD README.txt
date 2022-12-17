Welcome! This is v0.2 of the first public release of the Umi AI Character Creator/Generator!

Calling this an AI technically isn't correct. It's actually a Wildcard system filled with hundreds, and eventually thousands, of possible options that can all combine in an infinite number of ways. In much the same way as there are mods for Skyrim, Umi AI is a mod for the WebUI AI system.

To start using this engine, you need only put a few things into your txt2img prompt box.

__CORE__
__SFW__
__NSFW__

Pick one of those and paste it in your prompts. SFW and NSFW will load only scenes and characters of those types respectively. CORE will load either one without bias.

Next, we need to load our negative prompts. There are a lot you can choose, but I like to use:

Negative prompt: lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, out of focus, censorship, Missing vagina, Blurry faces, Blank faces, bad face, Ugly, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, extra ear, amputee, missing hands, missing arms, missing legs, SFW, Extra fingers, 6 fingers, Extra feet, Missing nipples, ghost, multiple boys, multiple girls

These negative prompts will NOT appear, or will at least have a high likelihood not to appear. For example, since the engine hates it when multiple characters are onscreen, we try to minimize the chances of them appearing.

Finally, we pick our generation settings. These will vary DRASTICALLY. I will give you a baseline setting range that I like to use, with the addendum that if you don't have at least a GTX 1080 and 32GB's of ram like me, you might not even be able to generate anything. It will turn to a black image, or WebUI might crash. Post in the Umi AI Discord if you have those issues: https://discord.gg/9K7j7DTfG2

Steps: 30, Sampler: Euler, CFG scale: 10, Size: 768x768, Denoising strength: 0.6, Clip skip: 2, ENSD: 31337, First pass size: 384x384

Let's break down what is going on with these settings.

Steps: 30 - The more steps you use to create an image, the better it's supposed to come out. But also, the more steps you use, the longer it takes to generate something., You can think of 1-step = 1-second of generation time. 30 steps is usually better than 15, but not as good as 60, or 90, etc. However...

Sampler: Euler - Euler is an extremely efficient low-step scaler. It outputs excellent quality images at low step counts. This means you usually want to just run Euler at 30 steps because the image gets worse the more steps you use above that. It also means you can output images very quickly. If you want to try other samplers, you can use DDIM which is best at higher stepcounts, or Euler A, which is different from Euler, but not functionally better. Swap between them and see what changes.

CFG Scale: 10 - CFG affects how closely the engine follows your prompts. A setting of 0 indicates maximum looseness and it will basically ignore your prompts, while 30 will follow it as strictly as possible. You never want either of these. I recommend 7-14 as a safe zone.

Next we need to talk about high-res stuff. You want to go into settings and enable "Upscale latent space image when doing hires. fix". Use ctrl+f to find it. Then, enable the hi-res fix whenever you're generating artwork, and set the first pass to some value, and the full resolution to an identical ratio as the first pass, but bigger. I usually just use a 1:1 ratio.

Ex: First Pass is 256x256, Full Size is 768x768. These are both 1:1 ratios, so we're good! This will load an initial low quality image, then scale it upward.

Finally, we set denoising to between 0.5 and 0.7. When you scale up to high res from low res, denoising affects how much the image changes. Lower settings are good for when you have a great initial image but it keeps getting changed at the last second and that bothers you. High denoising is the opposite. Play around with them.




So, in total, we will paste something like:
__NSFW__
Negative prompt: lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, out of focus, censorship, Missing vagina, Blurry faces, Blank faces, bad face, Ugly, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, extra ear, amputee, missing hands, missing arms, missing legs, SFW, Extra fingers, 6 fingers, Extra feet, Missing nipples, ghost, multiple boys, multiple girls
Steps: 30, Sampler: Euler, CFG scale: 10, Size: 768x768, Denoising strength: 0.6, Clip skip: 2, ENSD: 31337, First pass size: 384x384

Example: https://i.imgur.com/0NtG39I.png



This will load from the NSFW prompts and start generating random characters and scenes. These can be male or female, but male characters usually have issues. Umi AI will add more support for them in the coming days, weeks, and months.

Once you paste that wall of text, but before you hit generate, you need to do one more thing.
- Press the blue arrow next to Generate. https://i.imgur.com/Tvtnb6c.png It will 'pop down' the negative prompts and the sampler/steps info where it all belongs.

Aaaand then you can start generating characters and scenes! For further questions, please join the Umi AI discord where I will help you guys get all set up!

https://discord.gg/9K7j7DTfG2