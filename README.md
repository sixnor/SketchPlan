# SketchPlan

**TO RECREATE THE TRAINING DATA**
1. Download test and shell training data from: WILL BE LINK HERE
2. Extract data into SketchPlan folder
3. Download Zip-NeRF undistorted data from: https://smerf-3d.github.io/#data
4. Place the images, images_2, images_4, images_8, sparse directories into data/env_here/ for nyc, london, alameda and berlin
5. Run `python generate_dataset.py`


**TO TRAIN THE MODEL**
1. Set hyperparameters in `train_model.py`
2. Run `python train_model.py`

**TO PLAY AROUND WITH THE MODEL**
1. Set test scene in `visualizer.py`
2. Run `python visualizer.py`



## 🔧 Environment Setup

We recommend using [conda](https://docs.conda.io/en/latest/) to manage dependencies.

```bash
conda create -n sketchplan python=3.12
conda activate sketchplan
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install nerfstudio
pip install -r requirements.txt
```





