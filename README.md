### Deep Learning Model

#### Input Satellite Images

#### Mask R-CNN

I use the state-of-the-art instance segmentation model, [Mask R-CNN](https://arxiv.org/abs/1703.06870), to perform the building footprint segmentation task. Instance segmentation is a common computer vision task where the algorithm identifies the boundaries of all the instances of a class of interest (in this paper, buildings) on input images. Mask R-CNN is a model that first generates potential proposals of bounding boxes that contain the objects of interest; and then predicts what class the objects belong to and outlines the boundaries of the identified objects. Here, I provide details on the training schedule, the hyperparameter settings of the model, and the post-processing steps to allow for replication, and leave out descriptions of the intuition and the design of the Mask R-CNN architecture, as that is beyond the scope of this paper.

##### Model Set-up

The model in this paper is based on the official PyTorch implementation of Mask R-CNN. I use ResNet50 (with the Feature Pyramid Networks) as the backbone of the model. The inputs and outputs of the model are

$$Y = f_{\theta}(X)$$

In this formulation, $X$ is a $H \times W times C$ ($H$: height; $W$: width; $C$: color) matrix that represents the raw pixel levels of the RGB (Red, Green, Blue) bands in the image (ranging from 0 to 255). Assuming that there are $N$ instances detected in the image, then $Y$ is a 4-tuple of $(B, L, S, M)$, where $B$ is a matrix of size $N \times 4$ representing $N$ bounding boxes; $L$ is a vector of length $N$ representing the predicted labels for each instance (with 1 denoting buildings and 0 denoting background); $S$ is a vector of length $N$ representing the scores of each prediction (ranging from 0 to 1); and $M$ is a matrix of size $H \times W times N$ representing predicted pixel masks for all the instances (with values ranging from 0 to 1). $f_\theta$ is the deep learning model with parameters $\theta$. For simplicity, I present the single image case; in practice, images and annotations are fed into the model in small batches.

In order to obtain a pixel mask for each instance, I follow the convention and threshold $M$ at 0.5 to obtain a boolean mask for every instance. I then convert boolean pixel masks to polygons.

##### Training Data

I utilize multiple training datasets to improve the predictive performance of the model.

1. The model is first pretrained with the [COCO (Common Objects in Contexts) dataset](http://cocodataset.org), which is a large-scale natural image dataset containing 80 object categories and around 1.5 million object instances. Despite the fact that input images and object categories in COCO are different from target satellite images, pretraining the model with a large-scale dataset often provides meaningful performance gains, even when the model is later transferred across domains. I load the COCO-pretrained model provided in PyTorch.

2. The model is then fine-tuned on the [OpenAITanzania](https://competitions.codalab.org/competitions/20100) building footprint segmentation dataset, a collection of high-resolution aerial imagery collected by consumer drones in Zanzibar, Tanzania. All the buildings in the input images are identified, outlined and classified into three categories (completed building, unfinished building, and foundation) by human annotators. For compatibility with other building footprint datasets, I collapse the first two categories into buildings and drop the third category. Most input satellite images in this paper do not contain as many unfinished buildings as in Zanzibar, so I do not preserve the unfinished building category.

The OpenAITanzania dataset is chosen over other widely used geospatial machine learning datasets for several reasons: (1) The drone images are taken in Zanzibar, Tanzania, in a developing country context. This is crucial for matching satellite images that the model will make predictions on, in terms of the rural or urban landscape, and the distribution of the density, sizes and heights of the buildings. Other datasets, such as Inria Aerial and SpaceNet, are mostly in metropolitan cities in developed countries (e.g., Chicago, Las Vegas, Paris and Shanghai). (2) In the OpenAITanzania dataset, each instance of buildings is labelled separately, allowing me to effectively train an instance segmentation model; in both Inria Aerial and SpaceNet, buildings that border each other are annotated as one polygon.

The OpenAITanzania training data can be obtained [here](https://docs.google.com/spreadsheets/d/1tP133OqpwvkzHnkmS_3nezpTJMq06tpPM6P2qU6kaZ4/edit?usp=sharing). As the native resolution of this dataset is 7cm, I down-resolution the images 4 times, in order to match the resolution of the target satellite images. The images are provided as large tiles, which cannot be fed to the deep learning model directly, so I randomly sample 4034 chips (each with 1000x1000 pixels) from the large tiles (with an oversampling ratio of 3, meaning that any area in the sample is sampled 3 times in expectation). This serves as a form of data augmentation and takes advantage of the fact that the raw input images and annotations cover large contiguous areas. 90% of the chips are used for training, and the remaining 10% for validation.

## To Replicate

1. `preprocess_openaitanzania.py`: Pre-process OpenAITanzania training data.







For every image, the Mask R-CNN model produces predictions of up to 100 instances (TODO: double check this hyper param). Each instance is associated with a bounding box, a pixel mask, a label (in Mexico, building or background; in Kenya, thatched roof, metal roof, colored roof or background) and a confidence score between 0 to 1. I drop all the instances with a confidence score lower than 0.9, and convert the pixel mask to a polygon\footnote{I also simplify the polygon with the Douglas-Peucker algorithm with a pixel tolerance of 3.}. The average precision is 
XXX. TODO: add eval metrics. As a reference, the average precision of the Mask R-CNN model on the COCO data set is XXX. This is expected since remote sensing tasks are typically easier and remote sensing objects are more homogeneous in appearance.

I use extensive data augmentation techniques to improve performance, including random flipping and cropping, randomly changing brightness, contrast, saturation, hue, randomly blurring. The model is trained with a learning rate of 3e-4, batch size of 12 and for 70 epochs. In Kenya, all the buildings are additionally annotated with roof type (in three categories: thatched roof, metal roof or colored roof) whereas the roof type classification in Mexico are done automatically (as detailed below). All the machine learning codes are implemented with Python and PyTorch.

### Training Data






The spatial coverage of the data set is global. I use contemporary satellite images from Google Static Map as input images for the deep learning model. All of these images have been pre-processed and geo-referenced and they come from a variety of sources such as Maxar (formerly DigitalGlobe) and Airbus. I use images at zoom level 19, which translates to a spatial resolution of about 30 cm per pixel (on equator). These images contain only RGB (red, green, blue) bands, and no information in other wavelengths. As all the subsequent applications are conducted in relatively low-latitude countries, area distortion induced by Google Map's pseudo-Mercator system is minimal. However, researchers should take care to correct for area distortion when applying this method to high-latitude countries or regions.

The temporal coverage of the data set is limited to a contemporary cross section. All the images are retrieved from Google Statis Map API (https://developers.google.com/maps/documentation/maps-static/intro) in 2019, and are usually taken in the preceding few years. Google Static Map mosaics images from different time periods and sources together and thus cannot provide an exact timestamp for each image, but most images are updated frequently. Note that this is not a limitation of the proposed method, but the remote sensing data that is publicly available. With access to proprietary high-resolution images, this method can be applied to input images with more complete temporal coverage to study changes over time.





From the predicted polygons, I can extract a collection of observable characteristics on housing. This includes

\paragraph{Roof color/type.} I overlay the predicted polygon with the input image, and take an average color of the roof by separately averaging over all the pixels in the RGB channels. To obtain more semantically meaningful descriptions of roof types, I conduct a principal component analysis on the color vectors. In some cases where there are ground truth labels on roof types, those information can be used for training and model predictions will come with roof type classifications.

\paragraph{Building size.} I compute the area of the predicted polygon and convert it to square meters.

\paragraph{Spatial clustering.} For each polygon, I compute the number of neighboring polygons within a given radius. In subsequent applications, the radius is set to be 100 meters.

\paragraph{Dominant angles.} For each polygon, I compute the minimum-area rectangle that encompasses it, and take both axes of the rectangles as the dominant angle. In Figure \ref{fig:schematic_urban} and \ref{fig:schematic_rural}, I plot the histogram of these dominant angles within an image. As shown in the schematics, the dominant angles of buildings tend to be more aligned (concentrated in a few bars) in urban areas than in rural areas, representing more civic planning and coordinated development.