# TextureBasedImageRetriever

TextureBasedImageRetriever a Content Based Image Retriever that focuses on **texture**. It implements the **offline phase** which is the calulation of descriptors of all images in the datasetn, and the **online phase** that return the n-similar images from dataset given *an input image*.

### Prerequisites

* [python3](https://www.python.org/) should be installed.

* Installing needed python modules:

```shell
pip install -r requirements.txt

```

### Running
To test things, run:

```shell
jupyter notebook

```
Choose then [testing.ipynb](https://github.com/ahmed-BH/TextureBasedImageRetriever/blob/master/testing.ipynb)

OR, you can use it in your script:

* Calculating all descriptors of our dataset:
```python
import cbir
import settings

# TBIR class takes 2 argument:
# * PATH_TO_DATASET
# * PATH_WHERE_TO_SAVE_DESCRPTORS
my_cbir = cbir.TBIR(dataset_dir=settings.DATASET_DIR, descriptors_dir=settings.DESCRIPTORS_DIR))

# calculating the descriptors
my_cbir.offline_phase()

```

* Retrieving top *'settings.ELITE_NUMBER'* similar image for a *'tes_image.jpg'*: 
```python
similar = my_cbir.online_phase("tes_image.jpg")

# showing results
for s in similar:
    print("image path: {}, distance: {}".format(s["image_path"], s["distance"]))

```

* In *settings.py* file, you can change some entries to fit your needs such as:
1. *DATASET_DIR* 
2. *DESCRIPTORS_DIR*
3. *ELITE_NUMBER* : number of best fit images to return in the online_phase
4. *DEBUG*        : show extras output

