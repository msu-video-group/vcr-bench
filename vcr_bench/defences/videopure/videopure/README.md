# VideoPure

## Start
```
git clone https://github.com/deep-kaixun/VideoPure.git
cd VideoPure
pip install -r requirements.txt
```


## Pretrained Model(NL_res50)
Download [here](https://drive.google.com/file/d/19Xci1TRWBkBv7A7AiAw7tqTOnly-IQNG/view?usp=drive_link).

```bash
mv i3d_resnet50.pth ckpt/NL_res50.pth
```

## Run
```bash
python3 main.py --noise_type videopure

```

## Predownload Diffusers Weights
To prefetch `damo-vilab/text-to-video-ms-1.7b` before running defence:

```bash
python3 Defences/VideoPure/predownload_diffuser.py
```

Optional custom cache path:

```bash
python3 Defences/VideoPure/predownload_diffuser.py --cache-dir /path/to/hf-cache
```

Default behavior at first defence init:
- checks local cache in `Defences/VideoPure/hf-cache`
- auto-downloads if missing
- uses a lock file so parallel first-time inits do not start duplicate downloads

Environment flags:

```bash
export VIDEOPURE_LOCAL_FILES_ONLY=1   # default: use local cache for loading
export VIDEOPURE_AUTO_DOWNLOAD=1      # default: download on cache miss
export VIDEOPURE_CACHE_DIR=/path/to/hf-cache
```

Disable automatic first-run download (strict offline mode):

```bash
export VIDEOPURE_AUTO_DOWNLOAD=0
```

## Acknowledgements
[diffusers](https://github.com/huggingface/diffusers)





