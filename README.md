# SadTalker-Video-Lip-Sync

This project is based on SadTalker to implement Wav2lip for video lip synthesis. Using video files to generate voice-controlled lip shapes and setting a custom enhancement method for the face region, image enhancement is performed on the synthetic lip shape (face) region to improve the clarity of the generated lip shapes. Use the DAIN frame interpolation DL algorithm to add frames to the generated video and complement the transition action of the synthetic lip shapes between frames, making the synthesized lip shapes smoother, more realistic and natural.

Colab: [![Open on colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuroDonu/SadLIpFIX/blob/master/notebok.ipynb)

## 1. Environment setup. (Environment)

```
git clone https://github.com/NeuroDonu/SadLIpFIX
cd /SadLIpFIX/
python -m venv venv 
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
apt install ffmpeg -y
pip install -r requirements.txt
python -m pip install paddlepaddle-gpu==2.3.2 --extra-index-url https://paddle-wheel.bj.bcebos.com/2.5.1/linux/linux-gpu-cuda12.0-cudnn8.9-mkl-gcc12.2-avx/paddlepaddle_gpu-2.5.1.post120-cp39-cp39-linux_x86_64.whl
```

## 2. Repository structure

```
SadTalker-Video-Lip-Sync
├──checkpoints
|   ├──BFM_Fitting
|   ├──DAIN_weight
|   ├──hub
|   ├── ...
├──dian_output
|   ├── ...
├──examples
|   ├── audio
|   ├── video
├──results
|   ├── ...
├──src
|   ├── ...
├──sync_show
├──third_part
|   ├── ...
├──...
├──inference.py
├──README.md
```

## 3.ВЗаимодействие

```python
python inference.py --driven_audio <audio.wav> \
                    --source_video <video.mp4> \
                    --enhancer <none,lip,face> \  #(lip по дефолту)
                    --use_DAIN \ #(Использование этой функции займет большой объем видеопамяти и отнимет много времени.)
             		--time_step 0.5 #(Частота вставки кадров, по умолчанию 0,5, то есть 25 кадров в секунду —> 50 кадров в секунду; 0,25, то есть 25 кадров в секунду —> 100 кадров в секунду.)
```

## 4.Модели

Весь список моделей：

```python
├──checkpoints
|   ├──BFM_Fitting
|   ├──DAIN_weight
|   ├──hub
|   ├──auido2exp_00300-model.pth
|   ├──auido2pose_00140-model.pth
|   ├──epoch_20.pth
|   ├──facevid2vid_00189-model.pth.tar
|   ├──GFPGANv1.3.pth
|   ├──GPEN-BFR-512.pth
|   ├──mapping_00109-model.pth.tar
|   ├──ParseNet-latest.pth
|   ├──RetinaFace-R50.pth
|   ├──shape_predictor_68_face_landmarks.dat
|   ├──wav2lip.pth
```
You can download it here: https://mega.nz/file/cW833LJY#ZCaLy3_5SUntsb_wOPztzVAwwI2rbmva8sf4bIWxPTw
```python
#Installing checkpoints
apt install megatools
cd SadTalker-Video-Lip-Sync
megadl https://mega.nz/file/cW833LJY#ZCaLy3_5SUntsb_wOPztzVAwwI2rbmva8sf4bIWxPTw
tar -zxvf checkpoints.tar.gz
```

All the latest news comes out on my <a href=https://t.me/derkarta>channel</a>

## Based on
- SadTalker: https://github.com/Winfredy/SadTalker
- VideoReTalking：https://github.com/vinthony/video-retalking
- DAIN: https://arxiv.org/abs/1904.00830
- PaddleGAN: https://github.com/PaddlePaddle/PaddleGAN
