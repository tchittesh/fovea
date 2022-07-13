# FOVEA: Foveated Image Magnification for Autonomous Navigation

![Demo of FOVEA](./demo.gif)

Official repository for the ICCV 2021 paper _FOVEA: Foveated Image Magnification for Autonomous Navigation_ [[paper]](https://arxiv.org/abs/2108.12102) [[website]](http://www.cs.cmu.edu/~mengtial/proj/fovea/) [[talk]](https://youtu.be/PWe6CDeXJ7k).

## Setup (Data + Code + Models)

We use the Argoverse-HD dataset for evaluation. You can download that from its official website [here](http://www.cs.cmu.edu/~mengtial/proj/streaming/).

Our code implementation uses Python 3.8.5, PyTorch 1.6.0, and [mmdetection](https://github.com/open-mmlab/mmdetection) 2.7.0. To set
up the conda environment used to run our experiments, please follow these steps from some initial directory `/path/to/repos/`:

1. Create the conda virtual environment and install packaged dependencies. You should install [miniconda](https://docs.conda.io/en/latest/miniconda.html) if not already installed.
   ```
   conda create -n fovea python=3.8.5 && conda activate fovea
   conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
   pip install tqdm html4vision scipy
   ```
2. Install mmdetection 2.7.0 from source. This will first require installing mmcv 1.1.5.
   ```
   pip install mmcv-full==1.1.5 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html
   git clone https://github.com/open-mmlab/mmdetection.git && cd mmdetection
   git checkout 3e902c3afc62693a71d672edab9b22e35f7d4776  # checkout v2.7.0
   pip install . && cd ..
   ```
3. Install fovea from source.
   ```
   git clone https://github.com/tchittesh/FOVEA.git && cd fovea
   pip install . && cd ..
   ```

We have finetuned checkpoints available to download on [Google Drive](https://drive.google.com/file/d/1MPZM0OZThZ8SLdUO-uy6c7giq6J8R0Xu/view?usp=sharing).

Your final directory structure should look something like this:
```
/path/to/data/
├── Argoverse-1.1/
└── Argoverse-HD/

/path/to/repos/
├── mmdetection/
└── fovea/

/path/to/checkpoints/
├── faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth
├── KDE_SC.pth
├── KDE_SD.pth
├── KDE_SI.pth
└── LKDE_SI.pth
```

## Scripts

This should be super easy! Fill in the `experiments/path.sh` file with your local paths to the data/repos/checkpoints as well as where you want output files to go. Then, simply run 
```
sh experiments/KDE_SI.sh
```
(or any of the others) to perform inference on the Argoverse-HD dataset using the FOVEA models. For a deeper look at what each flag in these bash scripts is accomplishing, refer to `fovea/utils/config.py`. The Python script used for online inference is `tools/test.py`.

## Citation

If you use this code, please cite:
```
@inproceedings{thavamani2021fovea,
  title={FOVEA: Foveated Image Magnification for Autonomous Navigation},
  author={Thavamani, Chittesh and Li, Mengtian and Cebron, Nicolas and Ramanan, Deva},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15539--15548},
  year={2021}
}
```
