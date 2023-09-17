# SadTalker-Video-Lip-Sync


Этот проект основан на SadTalker для реализации Wav2lip для синтеза видео губ. Используя видеофайлы для создания форм губ, управляемых голосом, и устанавливая настраиваемый метод улучшения для области лица, выполняется улучшение изображения области синтетической формы губ (лица), чтобы улучшить четкость сгенерированных форм губ. Используйте алгоритм DL интерполяции кадров DAIN, чтобы добавлять кадры в сгенерированное видео и дополнить действие перехода синтетических форм губ между кадрами, делая синтезированные формы губ более плавными, реалистичными и естественными.
Колаб: [![Открыть на колаб](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuroDonu/SadLIpFIX/blob/master/notebook.ipynb) 

## 1.Установка среды. (Environment)

```
git clone https://github.com/NeuroDonu/SadLIpFIX
cd /SadLIpFIX/
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
apt install ffmpeg -y
pip install face_alignment facexlib yacs basicsr
wget https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-linux.zip
unzip ninja-linux.zip -d /usr/local/bin/
update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
pip install -r requirements.txt
python -m pip install paddlepaddle-gpu==2.3.2 --extra-index-url https://paddle-wheel.bj.bcebos.com/2.5.1/linux/linux-gpu-cuda12.0-cudnn8.9-mkl-gcc12.2-avx/paddlepaddle_gpu-2.5.1.post120-cp39-cp39-linux_x86_64.whl
```

## 2.Структура репозитория

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

Скачать можете вот тут: https://mega.nz/file/cW833LJY#ZCaLy3_5SUntsb_wOPztzVAwwI2rbmva8sf4bIWxPTw
```python
#Установка чекпоинтов
apt install megatools
cd SadTalker-Video-Lip-Sync
megadl https://mega.nz/file/cW833LJY#ZCaLy3_5SUntsb_wOPztzVAwwI2rbmva8sf4bIWxPTw
tar -zxvf checkpoints.tar.gz
```


## Взято за основу

- SadTalker: https://github.com/Winfredy/SadTalker
- VideoReTalking：https://github.com/vinthony/video-retalking
- DAIN: https://arxiv.org/abs/1904.00830
- PaddleGAN: https://github.com/PaddlePaddle/PaddleGAN
