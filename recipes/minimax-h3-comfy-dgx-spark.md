# MiniMax H3 on DGX Spark with ComfyUI

Make a talking 15-second clip on spark-56bc, with a style LoRA, a locked cast, and a path to a longer piece that does not invent a new face at every cut.

This recipe was measured on 2026-10-11. A 1280×704, 20-step, 362-frame job took about 97 to 100 minutes and returned a real MP4. That clock is the pin. Do not interrupt a job because an hour has passed.

## What this is

| Item | Value |
|---|---|
| Host | `spark-56bc` (SSH host in `~/.ssh/config`, user `mikesai3`) |
| Engine | ComfyUI `0.35.0`, service name `comfyui-minimax-h3` |
| Listen | `127.0.0.1:8188` only. No auth. Not on the tailnet. |
| Reach it | SSH, then `curl http://127.0.0.1:8188`. Do not publish the port. |
| Comfy root | `/home/mikesai3/comfyui-minimax-h3-dgx-spark/ComfyUI` |
| Diffusion | `models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| Text encoder | `models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Video VAE | `models/vae/minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE | `models/vae/minimax_h3_audio_vae_fp32.safetensors` |
| Style LoRA | `models/loras/80s_horror_wk.safetensors` (Wasserman, strength `1.0`) |
| Output | `ComfyUI/output/` |

Weights load on the first job, not when the process starts. An idle GPU at process start is normal.

## What this is not

- Do not start container `minimax-h3-fl2va` next to this Comfy process. That API is a different serve and must stay down while Comfy is up.
- Do not stand H3 on spark-d369. That box is for another model.
- Do not send a style-LoRA trigger word to OpenRouter or any hosted Hailuo call. The weights are not loaded there. The token can render as text.
- The style LoRA knows tape, neon, grain, and period wardrobe. It does not know a person.
- A prompt lock does not hold a face across clips. More adjectives will not fix a drift.
- A group lineup still is a casting sheet, not an identity lock. The model can pick the wrong body.
- A still holds who someone is. It does not hold how the last clip was moving.
- The motion chain below is the process. It has not been proven on a finished Midnight Munchies cut. Check stills before calling a chain locked.

## Before you generate

1. SSH in. Confirm the service is up:

```bash
ssh spark-56bc 'curl -sf --max-time 8 -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8188/system_stats'
```

Expect `200`.

2. Confirm the queue is empty and the card is cool:

```bash
ssh spark-56bc 'curl -sf http://127.0.0.1:8188/queue; nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader'
```

Need zero running jobs, temperature under 70°C, utilization near idle. Abort a new job if temperature is at or above 80°C.

3. Confirm the LoRA is in the model list, not only on disk:

```bash
ssh spark-56bc 'curl -sf http://127.0.0.1:8188/models/loras'
```

`80s_horror_wk.safetensors` must appear. If it is missing, download the v1.0.0 release asset into `models/loras/` and restart is not required if Comfy rescans; if the name is still absent, restart the service before spending a denoise.

License: the LoRA weights are a MiniMax H3 Community License derivative. Read `WEIGHTS_LICENSE.md` in [wassermanproductions/wasserman-80s-horror-lora](https://github.com/wassermanproductions/wasserman-80s-horror-lora) before you ship a file made with them. Credit Sam Wasserman. Do not publish generated MP4s under the MiniMax Community License without checking the territory terms.

## One clip

### Graph

API-format prompt. Node ids are local names. Class types must match.

| Node | class_type | Inputs that matter |
|---|---|---|
| 6 | `UNETLoader` | `unet_name=minimax_h3_fl2va_pruned_int8_convrot.safetensors`, `weight_dtype=default` |
| 200 | `LoraLoaderModelOnly` | model from 6, `lora_name=80s_horror_wk.safetensors`, `strength_model=1.0` |
| 13 | `CLIPLoader` | `clip_name=qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, `type=minimax`, `device=default` |
| 11 | `VAELoader` | video VAE |
| 24 | `VAELoader` | audio VAE |
| 104 | `MiniMaxH3ImageToVideo` | clip from 13, vae from 11, prompt, `width=1280`, `height=704`, `length=362` |
| 9 | `BasicScheduler` | model from 200, `scheduler=simple`, `steps=20`, `denoise=1.0` |
| 15 | `RandomNoise` | one integer seed per clip. Do not randomize on a rerun you want to compare. |
| 16 | `BasicGuider` | model from 200, conditioning from 104 |
| 17 | `KSamplerSelect` | `sampler_name=res_multistep` |
| 14 | `SamplerCustomAdvanced` | noise, guider, sampler, sigmas from 9, latent from 104 |
| 10 | `VAEDecode` | samples from 14, video VAE |
| 23 | `VAEDecodeAudio` | samples from 14, audio VAE |
| 91 | `CreateVideo` | images from 10, audio from 23, `fps=24` |
| 92 | `SaveVideo` | video from 91, `filename_prefix` unique per clip, `format=auto`, `codec=auto` |

`length` snaps to the `17k+5` grid. `362` is about 15 seconds and is the top of the trained range. `124` is about 5 seconds. Do not invent a longer length and call it tested.

Width and height step by 32. `1280×704` is the Wasserman VHS size used in the measured jobs. Do not send `1080p` or `2560×1440` to this graph.

### Prompt

Start every prompt with:

```text
80s_horror_wk, shot on VHS.
```

Then the scene. Name the action, the camera, and the sound. Do not write "film grain" as a substitute for the trigger. The trigger is what loads the look.

Dialogue goes in quotes with a delivery note. Keep lines short. A 15-second clip cannot hold a speech.

### Submit and wait

POST the API graph to `http://127.0.0.1:8188/prompt` from the Spark, not from another machine. A `prompt_id` and empty `node_errors` means the job was accepted. It does not mean the file exists.

Poll `/history/{prompt_id}`. While the id is absent, the job is still queued or running. Check `/queue` if you need to see that it is the only running job.

Measured wall for this graph: about 97 to 100 minutes. GPU stays near 93%. Temperature in the measured runs stayed in the high 70s to mid 80s. Abort if temperature reaches 85°C.

A Hermes or other remote terminal that dies at a timeout does not kill the Comfy job. Before you report "still running" or submit again, check three things:

1. `/queue` running count.
2. `/history/{prompt_id}` for `success` or `error`.
3. The file under `ComfyUI/output/` matching `filename_prefix`.

A paused poller keeps a stale "running" status after the MP4 has landed. A resumed serial submitter will queue a second copy of the same scene. Kill that process before you submit the next clip yourself.

### Verify the file

```bash
ffprobe -v error -show_entries format=duration,size \
  -show_entries stream=codec_name,width,height,avg_frame_rate,sample_rate \
  -of default=nw=1 /path/to/clip.mp4
```

Expect h264, 1280×704, 24 fps, aac, 32000 Hz, duration about 15.08 seconds. A JSON file is a failed request. Pull stills at the beats you care about before you call the clip done. A successful file is not a finished cast.

## Identity lock

Do this before the first denoise of any piece with a recurring cast. Do not start clip 1 until the stills are approved.

1. One still per character. That person alone. Face readable. Full costume. No other people. No logos. No readable brand names. A real chip brand in a still will come back on the bag.
2. Approve the still by looking at it. Reject a child, a flipped gender, a missing costume piece, or a second person.
3. Copy each still into `ComfyUI/input/`.
4. Replace `MiniMaxH3ImageToVideo` with `MiniMaxH3ReferenceToVideo` when a still must hold.
5. Connect the video VAE. Without it, the still only nudges the text encoder. That is the weak path, and it will drift.
6. Wire each still as `ref_images.ref_image_0`, `ref_images.ref_image_1`, and so on, up to nine. The autogrow prefix is `ref_images.` and the index starts at 0.
7. A flat key `ref_image_0` is passed through as an unexpected keyword. The job errors in seconds with `got an unexpected keyword argument 'ref_image_0'`. That failure was measured.
8. In every prompt, tag the person with the 1-based label that matches that order: `<Picture 1>`, `<Picture 2>`. The tag is 1-based even though the input name ends in `_0`. Never reorder the stills. Never swap the numbers.
9. Start at `ref_image_size=match`. If a face still drifts, rerun that clip at `max`. `max` holds identity harder and the node says it can take several times longer. Do not add adjectives. Do not replace the still mid-piece.
10. `first_frame` and `last_frame` only pin those endpoints. To hold a pose in the middle, chain `MiniMaxH3AddGuide` at that frame index. A negative index counts from the end. The guide needs the video VAE when an image is connected, and it needs a MiniMax H3 AV latent, not a generic latent.

Measured: a woman vampire drifted to a pale face and black hair when scene 3 was prompt plus style LoRA only. The rerun used one solo still, `<Picture 1>`, the video VAE, `ref_image_size=match`, and guides at frame 240 and frame -1. The wide shot and the 10-second close-up held the teased hair, heavy makeup, leather jacket, short skirt, and cape. The last frame cropped her face out, so a freeze still does not prove the point. Pull the beat, not only the last frame.

## One job, then cool down

The card stays near 93% for the whole denoise. Do not stack a second generate, a chat probe, or the old FL2VA container on top of it.

After each finished clip:

1. Confirm the MP4 exists and `/history` says success.
2. Leave the GPU idle for 10 minutes.
3. Do not submit the next job until temperature is below 70°C.
4. If it is still at or above 70°C at 10 minutes, keep waiting until it drops, or until 20 minutes have passed.
5. Do not start the next clip if temperature is still at or above 80°C.

## Longer than one clip

Do not ask this box for one long denoise. A 15-second clip is already about 97 minutes. A longer piece is a chain of clips.

A still holds identity. It does not hold motion across a cut. Separate text jobs plus an ffmpeg concat are a hard cut. Use that only when a hard cut is what you want.

### Motion chain

These nodes are installed on the Spark. The chain is the process. It is not yet a measured Midnight Munchies deliverable.

1. Pass the same approved stills, in the same `<Picture N>` order, on every segment.
2. Clip 1 has no motion context. Save the sampler output latent with `MiniMaxH3MotionContextSaveLatent`, `clip_index` 1. Wire the sampler latent, the same one you decode. Do not save a decoded frame and call it the chain.
3. Cooldown.
4. Clip 2: `MiniMaxH3MotionContextLoadLatent` at `clip_index` 1 into `MiniMaxH3MotionContext` `context_latent`. Set `context_length` to `22` and `audio_context_length` to `24`. Save this clip as index 2.
5. `22` frames is the seamless pin. `5` is barely fluid. `56` spends 2.3 seconds of the render on frames you throw away.
6. After decode, run `MiniMaxH3MotionContextTrim` with `trim_frames` equal to the context length, `fps=24`, and `match_tail` true. Those frames are the overlap. Leaving them in plays the join twice. H3's audio grid is about 8 ms off per clip. That error stacks. `match_tail` locks sound to picture.
7. Repeat. Load N, save N+1. One job, then cooldown.
8. If a face drifts, rerun that segment at `ref_image_size=max`. Do not swap the still.

`MiniMaxH3MotionContextChain` is the canvas button loop (Approve, Run, Reset). An agent driving the API should set `clip_index` explicitly. Do not click Run and also POST `/prompt` for the same slot.

### Hard-cut stitch

Use this when you want three finished clips in order and you accept a cut.

```bash
ffmpeg -i scene-1.mp4 -i scene-2.mp4 -i scene-3.mp4 \
  -filter_complex "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" \
  -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -r 24 \
  -c:a aac -b:a 192k -ar 48000 \
  -movflags +faststart \
  out.mp4
```

Re-encode. A stream-copy concat of these files warned non-monotonic audio timestamps at the joins. The concat filter plus an audio re-encode is the clean stitch.

Do not grade a 2K file with high-amplitude temporal noise. That pass washed the picture and produced an unusable bitrate. If a grade fails, ship the clean stitch.

Verify:

```bash
ffprobe -v error -show_entries format=duration,size \
  -show_entries stream=codec_name,width,height,avg_frame_rate,sample_rate \
  -of default=nw=1 out.mp4
```

Three 15.08-second clips stitch to about 45.25 seconds.

## Agent checklist

Copy this into the run log. Do not skip a line because the last run worked.

- [ ] Comfy `system_stats` returned 200 on spark-56bc via SSH.
- [ ] FL2VA container is not running beside it.
- [ ] Queue running count is 0 before submit.
- [ ] GPU temperature is under 70°C.
- [ ] LoRA name is in `/models/loras`.
- [ ] One approved solo still per recurring character, numbered, unchanged from the last clip.
- [ ] Reference inputs use `ref_images.ref_image_0` and up. Video VAE is connected.
- [ ] Prompt tags are `<Picture 1>` and up, same order as the stills.
- [ ] Prompt starts with `80s_horror_wk, shot on VHS.`
- [ ] `length` is 124, 362, or another `17k+5` value you meant.
- [ ] One `/prompt` POST. Saved `prompt_id`. No second submitter alive.
- [ ] Waited for `/history` success or a real output file. Did not treat a dead poller as a dead job.
- [ ] ffprobe shows h264, 1280×704, 24 fps, aac.
- [ ] Stills at the story beats were opened. Misses were named.
- [ ] Cooldown finished before the next clip.
- [ ] If this clip continues another, the sampler latent was saved and the overlap was trimmed.

## Failures already paid for

| What happened | What to do |
|---|---|
| Job "still running" after the MP4 exists | Read `/history/{id}` and the output file. Do not submit again. |
| Resumed a paused serial script after you queued the next clip | Kill that process first. It will submit a duplicate. |
| `ref_image_0` on `MiniMaxH3ReferenceToVideo` | Use `ref_images.ref_image_0`. |
| Face changes in a later scene | Solo still, VAE connected, same `<Picture N>`. Then `max` if `match` drifts. |
| "Pale man" plus teased hair and a cape | The costume cues won. Name the body in the still, not in a weaker noun. |
| Trademark on a prop | Ask for a plain unlabeled prop. The model draws the real brand. |
| Stream-copy concat clicks | Re-stitch with the concat filter and re-encode audio. |
| Temperature at or above 85°C | Stop. Cool. Do not start the next clip at or above 80°C. |
| Hosted API asked to apply this LoRA | It cannot. This recipe is the Spark Comfy graph only. |
