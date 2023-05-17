# Partially Supervised Segmentation via CAM-Guided Class-Agnostic Segmentation

Code adapted from: https://github.com/jbeomlee93/AdvCAM  
Used classifier:        https://drive.google.com/file/d/1G0UkgjA4bndGBw2YFCrBpv71M5bj86qf/view?usp=sharing

## Requirements
To install requirements run: `pip install -r requirements.txt`   
Install the Pascal VOC 2012 dataset: https://drive.google.com/file/d/1e-yprFZzOYDAehjyMVyC5en5mNq6Mjh4/view
Used python version: 3.10.9  
The code was tested on NVIDIA GeForce GTX 1080 Ti GPU  
The used libraries are:

| Library                     | Version |
|-----------------------------|---------|
| pytorch                     | 2.1.0   |
| torchvision                 | 0.15.0  |
| numpy                       | 1.23.5  |
| matplotlib                  | 3.7.0   |
| opencv                      | 4.7.0   |
| pillow                      | 9.4.0   |
| segmentation-models-pytorch | 0.3.1   |
| tqdm                        | 4.64.1  |
| wandb                       | 0.13.10 |
| chainercv                   | 0.13.1  |
| timm                        | 0.4.12  |


## Project Structure
The project structure is as follows:
- ckpts
  - res50_cam.pth: Pre-trained classifier
- class_agnostic_seg: Directory for the class-agnostic segmentation model training and evaluation
  - ca_eval.py: 
  - ca_train.py
  - ca_utils.py
- sem_seg_with_PGT: Directory for the semantic segmentation model training and evaluation
    - seg_eval.py
    - seg_train.py
    - seg_utils.py
- cam_generator.py: Generates the CAMs for the images in the dataset
- classification_model.py: Classifier model
- dataset_seg.py: Loads the datasets from voc.py 
- requirements.txt
- run.py: Main file for the training and evaluation of the models
- transforms.py: Transformations used for training and evaluation
- voc.py: Pascal VOC dataset classes

## Usage
To run the code just run the `run.py` file.
### Arguments
| Argument                  | Description                                                                                                                                            | Default                    |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| dataset_root              | Path to the Pascal VOC 2012 dataset                                                                                                                    | "./data/VOC"               |
| data_cls_sub              | Classes to use for training/evaluation. You can use "A", "B", "all" or you can define your own subset by joyning the categories with "&" e.g "cat&dog" | "A"                        |
| other_cams                | Weather to use the CAMs from the other classes                                                                                                         | True                       |
| other_cams_max            | If True use CAM<sup>max</sup> else CAM<sup>Σ</sup>                                                                                                     | True                       |
| other_cams_presnt         | If True use CAM<sub>P</sub> else CAM<sub>O</sub>                                                                                                       | True                       |
| input_size                | Spatial size for training images e.g. H x W                                                                                                            | 320                        |
| focused                   | Class-agnostic focused instance segmentation                                                                                                           | False                      |
| mode                      | What input to use {"cam", "img", "both"} - just for class-agnostic focused instance segmentation                                                       | "both"                     |
| bbox_trans_range          | Range of the bbox transform for class-agnostic focused instance segmentation                                                                           | (-0.1, 1.1)                |
| learning_rate             | Learning rate for the training                                                                                                                         | 0.0001                     |
| batch_size                | Batch size for the training                                                                                                                            | 16                         |
| num_epochs                | Number of epochs for the training                                                                                                                      | 10                         |
| num_workers               | Number of workers                                                                                                                                      | 4                          |
| model_architecture        | Used model architecture                                                                                                                                | "DeepLabV3Plus"            |
| encoder_name              | Used encoder name for the model architecture                                                                                                           | "tu-xception71"            |
| cam_weights_name          | Path to the classifier weights file                                                                                                                    | "./ckpts/res50_cam.pth"    |
| cam_network               | Used network for the CAM generation                                                                                                                    | "classification_model"     |
| target_layer              | Target layer for the CAM generation                                                                                                                    | "stage4"                   |                                                         | None                       |
| save_model                | Weather to save the model                                                                                                                              | True                       |
| save_model_path           | Path to save the model                                                                                                                                 | "./ckpts"                  |
| save_model_name           | Name to save the model                                                                                                                                 | "model.pth"                |
| class_agnostic_model_name | Name of the class-agnostic model - used to load the model for evaluation or pseudo-mask generation                                                     | "class_agnostic_model.pth" |
| pseudo_model_name         | Name of the semantic model - used to load the model for evaluation                                                                                     | "semantic_model.pth"       |
| log_wandb                 | Weather to log to wandb                                                                                                                                | True                       |
|task                       | Task to run {"class_agnostic_train", "class_agnostic_eval", "pseudo_train", "pseudo_eval"}                                                             | "class_agnostic_train"     |

