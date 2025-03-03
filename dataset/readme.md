# InfAnFace: Bridging the Infant--Adult Domain Gap in Facial Landmark Estimation in the Wild


## Introduction 
This is the dataset for:

Wan, M., Zhu, S., Luan, L., Prateek, G., Huang, X., Schwartz-Mette, R., Hayes, M., Zimmerman, E., & Ostadabbas, S.  "InfAnFace: Bridging the infant-adult domain gap in facial landmark estimation in the wild." *26th International Conference on Pattern Recognition* (ICPR 2022). [[arXiv link](https://arxiv.org/abs/2110.08935)]

In this paper, we introduce:

* **InfAnFace**, the first **Inf**ant **An**notated **Face**s dataset, consisting of 410 images of infant faces with labels for 68 facial landmark locations and various pose attributes. 

* State-of-the-art facial landmark estimation models designed specifically for infant faces, based on the [HRNet](https://github.com/HRNet/HRNet-Facial-Landmark-Detection) architecture. 

See our [GitHub repository](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking) for more info.


## Licence

By downloading or using any of the datasets provided by the ACLab, you are agreeing to the “Non-commercial Purposes” condition. “Non-commercial Purposes” means research, teaching, scientific publication and personal experimentation. Non-commercial Purposes include use of the Dataset to perform benchmarking for purposes of academic or applied research publication. Non-commercial Purposes does not include purposes primarily intended for or directed towards commercial advantage or monetary compensation, or purposes intended for or directed towards litigation, licensing, or enforcement, even in part. These datasets are provided as-is, are experimental in nature, and not intended for use by, with, or for the diagnosis of human subjects for incorporation into a product.


## The InfAnFace dataset

**InfAnFace**, the **Inf**ant **An**notated **Face**s dataset, is the first public dataset of infant faces with facial landmark annotations. It consists of 410 images of infant faces with labels for 68 facial landmark coordinates and various pose attributes. The file structure is as follows:

````
infanface-hrnet-public
-- readme.md
-- images
   |-- ads
   |-- google
   |-- google2
   |-- youtube
   |-- youtube2
-- labels.csv
````

The images, in `/images`, are organized into five batches, which were collected and annotated by our team separately: 

* `/ads` (104 images) contains infant formula advertisement video stills sourced from YouTube,
* `/google` (100 images) and `/google2` (51 images) contain photos sourced from Google Images, and
* `/youtube` (100 images) and `/youtube2` (55 images) contain video stills sourced from YouTube.

Annotated labels can be found in `labels.csv`. These include:

* **68 facial landmark annotations** following the Multi-PIE layout (like [300-W](https://ibug.doc.ic.ac.uk/resources/300-W/)),
* binary **attribute labels** (*turned*, *occluded*, *expressive*, and *tilted*, as defined in our paper), and
* a division of the images into Train and Test sets, and a division of the Test set into Common and Challenging sets, as described below.


## Citation
Here is a BibTeX entry for our ICPR 2022 paper:
````
@inproceedings{WanICPR2022,
  title={{InfAnFace}: {Bridging} the infant-adult domain gap in facial landmark estimation in the wild},
  author={Michael Wan and Shaotong Zhu and Lingfei Luan and Prateek Gulati and Xiaofei Huang and Rebecca Schwartz-Mette and Marie Hayes and Emily Zimmerman and Sarah Ostadabbas},
  booktitle = {2022 {International} {Conference} on {Pattern} {Recognition} ({ICPR})},
  year={2022}
}
````

