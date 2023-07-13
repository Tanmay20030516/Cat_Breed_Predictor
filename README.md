# Cat Breed Predictor

## About the project
+ This project uses **VGG16** model to predict breed of cat. The model was trained to predict a cat's breed among 4 classes *(Bengal, Bombay, Himalayan and Persian breeds)*.
+ **VGG16** is a very powerful CNN architecture which is trained on **ImageNet** dataset (contains more than a **million images** and **1000 classes**).

## Dataset
Dataset was collected from **Kaggle** and **Google Images** (via chrome extension). Some manual cleaning was also done.

## Tools
+ Numpy
+ Matplotlib
+ Tensorflow
+ OpenCV
+ Streamlit (for deployment)
+ VGG16

## Model
*The architecture I chose was underfitting the data ( probably due to less training examples ) so then I decided to use pretrained model*
+ VGG16 architecture (in our model, the input shape was changed to **(256,256,3)** from **(224,224,3)** )
![VGG16](helper/vgg16_image.png)
+ A pretrained VGG16 was used to carry out the classification task.
+ Since the VGG16 model was trained **ImageNet** dataset, it is capable of extracting minute details from the input images. This suits our purpose.
+ Output layer is a softmax layer outputting 4 class probabilities
![model](helper/model.png)

## Performance
+ Since the **total** dataset was of about **4-5k images**, the model is pretty accurate. And scope of improvement in performance exists.
+ The model was trained for **15 epochs**, and below are some training plots -
![loss](helper/Loss.png)
![accuracy](helper/accuracy.png)

## Result
+ The model achieved a **Precision** of **0.990595**, **Recall** of **0.913295**, **F1 Score** of **0.950376** and an **Accuracy** of **0.781385** on the **test** set.

## Working
![working](helper/streamlit-app-2023-07-13-19-07-81.gif)
![pg1](helper/Screenshot(309).png)
![pg2](helper/Screenshot(310).png)

## Running in local environment
+ clone this repo in local machine.
+ create and switch to virtual environment.
+ install the dependencies from `requirements.txt`.
+ in the terminal, type `streamlit run app.py`.

## References
+ [https://www.tensorflow.org/api_docs/python/tf](https://www.tensorflow.org/api_docs/python/tf)
+ [https://docs.streamlit.io/](https://docs.streamlit.io/)

## Future improvements to do
_This repo is open to contributions_ 😊
+ Adding a **cat vs non-cat classifier** before **cat breed classifier** model, to **avoid** prediction on non-cat uploaded images.
+ Training the base (cat breed classifier) model on a better and cleaner data, and increasing the classes.
