# CAB420-Assignment-3

AI client designed to identify the instrument playing in an audio file

https://docs.google.com/document/d/1eHoOQRKqMwNzufmLqakUrGKqjmJeSzuPhyg24kqIl_k/edit?usp=sharing

## Dependencies

- Install [ffmpeg](https://ffmpeg.org/download.html) version 2.8.15 or newer
  - Test install is functioning by typing `ffmpeg -version`
- `pip install numpy soundfile librosa json datetime sklearn tqdm pytorch fastai pathlib shutil matplotlib seaborn jupyter`

To Run CNN

- Create data folder
- Extract NSynth Dataset to data folder maintaining the following structure
  - ./data/
    - nsynth-test/
      - audio/
      - examples.json
    - nsynth-train/
      - audio/
      - examples.json
    - nsynth-valid/
      - audio/
      - examples.json
- `python cnn_preproccessing.py`
- `python generate_filelist.py`
- `python cnn_classify.py`
- `python svm_classify.py`
- `python knn_classify.py`
