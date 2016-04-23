## [Context Encoders: Feature Learning by Inpainting](http://cs.berkeley.edu/~pathak/context_encoder/)
Accepted at [CVPR 2016](http://cvpr2016.thecvf.com)<br/>
[Project Website](http://cs.berkeley.edu/~pathak/context_encoder/)

If you find Context-Encoder useful in your research, please cite:

    @inproceedings{pathakCVPR16context,
        Author = {Pathak, Deepak and Kr\"ahenb\"uhl, Philipp and Donahue, Jeff and Darrell, Trevor and Efros, Alexei},
        Title = {Context Encoders: Feature Learning by Inpainting},
        Booktitle = {Computer Vision and Pattern Recognition ({CVPR})},
        Year = {2016}
    }

### Contents
1. [Semantic Inpainting Demo](#1-semantic-inpainting-demo)
2. [Features Caffemodel](#2-features-caffemodel)

### 1) Semantic Inpainting Demo

Inpainting using context encoder trained jointly with reconstruction and adversarial loss. Currently, I have only released the demo for the center region inpainting only and will release the arbitrary region semantic inpainting models soon.

0. Install Torch:  http://torch.ch/docs/getting-started.html#_

1. Clone the repository
  ```Shell
  git clone https://github.com/pathak22/context-encoder.git
  ```
  
2. Demo
  ```Shell
  cd context-encoder
  bash ./models/scripts/download_inpaintCenter_models.sh
  # This will populate the `./models/` folder with trained models.

  net=models/inpaintCenter/paris_inpaintCenter.t7 name=paris_result imDir=images/paris overlapPred=4 manualSeed=222 batchSize=21 gpu=1 th demo.lua
  net=models/inpaintCenter/imagenet_inpaintCenter.t7 name=imagenet_result imDir=images/imagenet overlapPred=0 manualSeed=222 batchSize=21 gpu=1 th demo.lua
  net=models/inpaintCenter/paris_inpaintCenter.t7 name=ucberkeley_result imDir=images/ucberkeley overlapPred=4 manualSeed=222 batchSize=4 gpu=1 th demo.lua
  # Note: If you are running on cpu, use gpu=0
  # Note: samples given in ./images/* are held-out images
  ```
  
Sample results on held-out images: 

![teaser](images/teaser.jpg "Sample inpainting results on held-out images")

### 2) Features Caffemodel

Features for context encoder trained with reconstruction loss.

- [Prototxt](http://www.cs.berkeley.edu/~pathak/context_encoder/resources/ce_features.prototxt)
- [Caffemodel](http://www.cs.berkeley.edu/~pathak/context_encoder/resources/ce_features.caffemodel)
