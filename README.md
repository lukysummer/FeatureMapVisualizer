# FeatureMapVisualizer - In-depth Visualization Tool for CNN-based Models

# Overview
FeatureMapVisualizer allows visualizations using individual feature maps of CNN-based image classification models to get insights about their predictions. It provides the following visualization techniques :

  - Find Most-Activated Feature Maps for Each Class
  - Visualize Patterns Captured by a Top Feature Map
  - Visualize Different Feature Maps' Activations on One Image
  - Visualize One Feature Map's Activations on Different Images 
  - Visualize Different Feature Maps' Activations on Different Images 
  - Plot Sum of Activations of Top Feature Maps for Each Class

## Usage

Here I will describe how you can get and use FeatureMapVisualizer for your own projects.

###  Getting it

To download FeatureMapVisualizer, either fork this github repo or simply use PyPi via pip.
```sh
$ pip install FeatureMapVisualizer
```

### Using it

Scrapeasy was programmed with ease-of-use in mind. First, import Website and Page from Scrapeasy

```Python
from FeatureMapVisualizer import visualizer
```

And you are ready to go! The visualizer class contains all visualization methods mentioned above. I will show you the sample code for each method below.

## Define Feature Map Visualizer Class
First things first, let's define the Feature Map Visualizer class.
```Python
FV = visualizer.FeatureMapVisualizer(model, model_type="resnet")
```

## **[Pre-Viz]** Find Most-Activated (Top) Feature Maps for Each Class
ResNet50 has 2,048 feature maps in the last convolutional layer, while vgg16 has 512. Popular visualization tools such as [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) looks at the *average* of those feature maps. But EACH individual feature map must be learning different characteristics of the classes, so why not look at the specific ones? So this function lets us figure out ***which specific feature maps from the model's last convolutional layer are activated the most when the model sees images of a particular class***. Then we can focus on those feature maps when visualizing.

```Python
top_feature_map_dict = FV.find_unique_filters(
                                layer = 12, 
                                train_dir = "my_folder/train/", 
                                classes = ["cat", "dog"], 
                                n_imgs_dict = [{"cat":1955, "dog":1857}],
                                n_each_img = 25,
                                n_each_class = 25,
                                            )
```
Let's look at each parameter:
* `layer`        : (int) index of the model's convoultional layer to investigate 
*To look at feature maps in the last concolutional layer (which are most sensitive to specific shapes rather than high-level texture), use `-2` for resnet or `12` for vgg16
* `train_dir`    : (str) address of the folder that contains training data including "/" at the end 
* `classes`      : (list of strs) list containing (at least two) class names in string
* `n_imgs_dict`  : (dict) key : class name (str), value : # of training images for that class (int) 
* `n_each_img` ***(optional)***  : (int) # of top feature maps to save for EACH IMAGE, ***default=25***
* `n_each_class` ***(optional)***  : (int) # of top feature maps to save for EACH CLASS, ***default=25***

As the output, you will get a dict of a form: 
```Pytyhon
{'cat': [452, 312, 327, 12, 114],
 'dog': [115, 23, 135, 203, 350, 132]}
```
which indicates the indices of feature maps that are most sensitive to each class.

## **Viz #1 -** Visualize Patterns Captured by a Top Feature Map
After narrowing down the specialized feature maps for each class, I investigated which prominent shapes or object parts were captured by each INDIVIDUAL top feature map. Inspired by [this article](https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030), the idea is to first generate a small random image (I used 33 by 33 pixels) and then iteratively adjust its pixels in a direction that maximizes the selected feature map’s activation. This is done by minimizing loss equal to the negative of the sum of the feature map’s activation. 

```Python
layer = -2        # last convolutional layer for ResNet50
filter = 452      # top #1 feature map for cat class
img, img_name = FV.visualize_patterns(
                            layer = layer, 
                            filter_n = filter,
                            init_size=33, 
                            lr = 0.2, 
                            opt_steps = 20,  
                            upscaling_steps = 20, 
                            upscaling_factor = 1.2, 
                            print_loss = False, 
                            plot = True,
                                        )
plt.figure(figsize=(15,15))
plt.imshow(np.array(Image.open(img_name)))
plt.show()
```
Parameters:
- `layer`            : (int) index of the convolutional layer to investigate feature maps 
*For the last convolutional layer, use `-2` for resent50 or `12` for vgg16.
- `filter_n`         : (int) index of the feature map to investigate in the layer
- `init_size`        ***(optional)***  : (int) intial length of the square random image, ***default=33***
- `lr`              ***(optional)***   : (float) learning rate for pixel optimization, ***default=0.2***
- `opt_steps`        ***(optional)***  : (int) number of optimization steps, ***default=20***
- `upscaling_steps`  ***(optional)***  : (int) # of upscaling steps, ***default=20***
- `upscaling_factor` ***(optional)***  : (float) >1, upscale factor, ***default=1.2***
- `print_loss`       ***(optional)***  : (bool) if True, log info at each optimizing iteration, ***default=False***
*If activation is 0 for all iterations, there is a problem.
- `plot` ***(optional)***  : (bool) if `True`, plot the generated image at each optimizing iteration, ***default=True***

You can see examples of generated images [here](https://medium.com/codex/ch-7-decoding-black-box-of-cnns-using-feature-map-visualizations-45d38d4db1b0#0492).

##  **Viz #2 -** Visualize Different Feature Maps' Activations on One Image
Here we find feature maps whose activations maximize for a single image and **highlighting each feature map’s most attended regions of the image** by overlaying its activation map on top of the image. As mentioned before, [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) answers a similar question of which parts of the image the entire last convolutional layer is paying attention to, but it doesn’t go as specific as to looking at ***each individual feature map***.

```Python
layer = -2
activations_list_cat = FV.one_image_N_top_feature_maps(
                                    layer, 
                                    img_path="my_folder/test/cat/cat_1.jpg", 
                                    n=60, 
                                    n_plots_horiz=6, 
                                    n_plots_vert=10, 
                                    plot_w=50, 
                                    plot_h=30,
                                                    )
```

- `layer`         : (int) index of the convolutional layer to investigate feature maps 
*For the LAST convolutional layer, use `-2` for resent50 and `12` for vgg16.
- `img_path`      : (str) path to the image to investigate
- `plot`         ***(optional)***  : (bool) if True, plot the top N feature maps' activation maps on the image, ***default=True***
            **/// MUST BE : n_plots_horiz * n_plots_vert = n ///**
- `n`              ***(optional)***  : (int) number of top feature maps to plot, ***default=100***
- `n_plots_horiz`  ***(optional)***  : (int) number of feature maps to plot horizontally, ***default=10***
- `n_plots_vert`   ***(optional)***  : (int) number of feature maps to plot vertically, ***default=10***
            **/// Recommended : (n_plots_horiz/n_plots_vert) = (plot_h/plot_w) ///**
- `plot_h`         ***(optional)***  : (int) height of the plot, ***default=50***
- `plot_w`         ***(optional)***  : (int) width of the plot, ***default=50***
- `print_logits`   ***(optional)***  : (bool) if True, print model logits (outputs) for the image, ***default=***
- `imagenet`       ***(optional)***  : (bool) if True, print_logits will print the logits for corresponding imagenet labels, ***default=False***
- `plot_overlay`   ***(optional)***  : (bool) if `True`, overlay the top feature map on top of the image and plot the overlaid image; if `False`, plot the original feature map only, ***default=True***

You can see examples of generated visualizations [here](https://medium.com/codex/ch-7-decoding-black-box-of-cnns-using-feature-map-visualizations-45d38d4db1b0#6fe6).

##  **Viz #3 -** Visualize One Feature Map's Activations on Different Images 
For this visualization, when you pass the index of the feature map you want to investigate (`filter_idx`), it plots its activation map for at most 100 images in `dataloader`.

```Python
activations_list_dog = FV.one_feature_map_N_images(
                                layer = -2, 
                                dataloader = dog_test_dataloader, 
                                filter_idx = 1239, 
                                plot=True, 
                                max_n_imgs_to_plot = 100, 
                                plot_overlay = True, 
                                normalize = True,
                                folder = "viz3/dog",
                                class_name = "dog",
                                                  )
```
- `layer`         : (int) index of the convolutional layer to investigate feature maps 
*For the LAST convolutional layer, use `-2` for resent50 and `12` for vgg16.
- `dataloader`   : (torch.utils.data.dataloader object) dataloader containing images to plot (usually images of a single class)
- `filter_idx`   : (int) index of the feature map to investigate in the layer
- `plot` ***(optional)***          : (bool) if True, plot the feature maps' activation maps on images in the dataloader, ***default=True***
- `max_n_imgs_to_plot` ***(optional)***      : (int) maximum number of images to plot, ***default=100***
- `plot_overlay` ***(optional)***      : (bool) if `True`, overlay the top feature map on top of the image and plot the overlaid image; if `False`, plot the original feature map only, ***default=True***
- `normalize` ***(optional)***     : (bool) if True, normalize the mask feature map by dividing by maximum value, ***default=True***
- `folder` ***(optional)*** : (str) name of the folder to save images (only if you want to save the visualizations), ***default=""***
- `class_name`  ***(optional)***  : (str) name of the class the images belong to, ***default=""***

You can see examples of generated visualizations [here](https://medium.com/codex/ch-7-decoding-black-box-of-cnns-using-feature-map-visualizations-45d38d4db1b0#7b0a).

##  **Viz #4 -** Visualize Different Feature Maps' Activations on Different Images 
This visualization is the same as Viz #3 except that you pass the indices of **multiple feature maps** you want to investigate together (`filter_idxs`). It plots the **SUM of activation maps** for all those feature maps for images in the `dataloader`.

```Python
activations_list_dog = FV.M_feature_map_N_images(
                                layer = -2, 
                                dataloader = dog_test_dataloader, 
                                filter_idxs = top_feature_map_dict['dog'],
                                plot = True, 
                                max_n_imgs_to_plot = 100, 
                                plot_overlay = True,
                                                )
```
- `layer`         : (int) index of the convolutional layer to investigate feature maps 
*For the LAST convolutional layer, use `-2` for resent50 and `12` for vgg16.
- `dataloader`   : (torch.utils.data.dataloader object) dataloader containing images to plot (usually images of a single class)
- `filter_idxs`   : (list of ints) indices of feature maps to investigate in the layer
- `plot` ***(optional)***          : (bool) if True, plot the feature maps' activation maps on images in the dataloader, ***default=True***
- `max_n_imgs_to_plot` ***(optional)***      : (int) maximum number of images to plot, ***default=100***
- `plot_overlay` ***(optional)***      : (bool) if `True`, overlay the top feature map on top of the image and plot the overlaid image; if `False`, plot the original feature map only, ***default=True***

##  **Viz #5 -** Plot Sum of Activations of Top Feature Maps for Each Class
Here, for each class, the activations of all class-wise top feature maps (found in **Pre-Viz** section) is added up for EACH test image, and plotted on the same graph with different colours representing different classes.

```Python
sum_dicts = FV.sum_top_feature_maps_by_class(
                        layer = -2, 
                        transform = transform, 
                        img_dir = "xray_sh", 
                        top_feature_maps_dict = {"cat":[1,3,9], "dog":[2,7]},
                        plot = True,
                                            )
```
- `layer`         : (int) index of the convolutional layer to investigate feature maps 
*For the LAST convolutional layer, use `-2` for resent50 and `12` for vgg16.
- `transform` : (torchvision.transforms object) transform to be applied to each test image
- `img_dir`   : (str) address of the folder containing image folders
        *Image folders' names must be the same as target class names.
***/// You MUST either pass `top_feature_maps_dict` or ALL of `train_dir`, `classes`, and  `n_imgs_dict`. ///***
- `top_feature_maps_dict` ***(conditionally optional)*** : (dict) (key, value)=(class name, list of top feature maps for that class) e.g. {"cat":[1,3,5], "dog":[2,4,8]}, ***default=None***
- `train_dir`     ***(conditionally optional)*** : (str) address of the folder that contains training data including "/" at the end  e.g. "train_data/", ***default=None***
- `classes`       ***(conditionally optional)*** : (list of strs) list containing (at least two) class names in string e.g. ["cat", "dog"], ***default=None***
- `n_imgs_dict`   ***(conditionally optional)*** : (dict) key : class name (str), value : # of training images for that class (int) e.g. {"dog":955, "cat":1857}, ***default=None***
- `plot`       ***(optional)*** : (bool) show plots if True, ***default=True***

You can see examples of generated visualizations [here](https://medium.com/codex/ch-7-decoding-black-box-of-cnns-using-feature-map-visualizations-45d38d4db1b0#d4b0).

License
----

MIT License

Copyright (c) 2022 Jahyun Shin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Hire me!: [LinkedIn](https://www.linkedin.com/in/lucrece-shin/)