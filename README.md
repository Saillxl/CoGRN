# CoGRN



---

## 📖 Introduction



---

## 📂 Clone Repository

```bash
git clone https://github.com/Saillxl/CoGRN.git
cd CoGRN/
```

---

## 📑 Dataset Preparation

The dataset should follow the format below. 


## ⚙️ Requirements
We recommend creating a clean environment:

```
conda create -n CoGRN python=3.10
conda activate CoGRN
pip install -r requirements.txt
```

## 🚀 data_preprocess

```
python train.py --hiera_path './checkpoints/sam2_hiera_large.pt' --train_image_path 'data/BUSI/train/img.yaml' --train_mask_path 'data/BUSI/train/ann.yaml' --save_path 'output/BUSI' 
```

## 🧪 Testing
Run test.py
```
python test.py --checkpoint 'output/BUSI/SAM2-UNet-70.pth' --test_image_path 'data/BUSI/test/img.yaml' --test_gt_path 'data/BUSI/test/ann.yaml' --save_path 'output/'
```

## 📌 Citation
If you find this repository useful, please cite our paper(bibtex):
```

```

## 🙏 Acknowledgement

##
