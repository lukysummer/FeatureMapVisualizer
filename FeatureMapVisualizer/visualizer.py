from .save_features import SaveFeatures
       

class FeatureMapVisualizer():
    def __init__(self, 
                 model,  
                 model_type="resnet", 
                 ec=False): 
        '''
        ### Feature Map Visualization class:  ###
          Contains various functions for visualization methods using convolutional feature maps
          || PARAMETERS ||
            model       : (PyTorch model)
            model_type  : (str) must be "resnet" or "vgg"
            ec          : (bool) True if using encoder, False if using the whole model (encoder + classifier)
        '''
        assert model_type in ["resnet", "vgg"], 'mode_type must be either "resnet" or "vgg"!'
        self.model = model.eval().cuda() if use_cuda else model.eval()
        for p in self.model.parameters(): p.requires_grad=False
        self.model_type = model_type
        self.ec = ec
    

    def register_hook(self, layer):
        ''' Register hook in the requested layer '''
        if self.model_type == "vgg":
            conv_layers = [c for c in list(self.model.children())[0] if isinstance(c, nn.Conv2d)]
            activations = SaveFeatures(conv_layers[layer])  # register hook

        elif self.model_type == "resnet":
            if self.ec:
                activations = SaveFeatures(self.model[-2][-2])
            else:
                activations = SaveFeatures(self.model.layer4[layer])
        return activations
    

    def find_unique_filters(self, 
                            layer, 
                            train_dir, 
                            classes, 
                            n_imgs_dict, 
                            n_each_img=25, 
                            n_each_class=25):
        '''
        || PARAMETERS ||
          layer        : (int) if using last convolutional layer, use -2 for resnet & 12 for vgg16
          train_dir    : (str) address of the folder that contains training data including "/" at the end  e.g. "train_data/"
          classes      : (list of strs) list containing (at least two) class names in string e.g. ["cat", "dog"]
          n_imgs_dict  : (dict) key : class name (str), value : # of training images for that class (int) e.g. {"dog":955, "cat":1857}
          n_each_img   : (int) # of top feature maps to save for EACH IMAGE
          n_each_class : (int) # of top feature maps to save for EACH CLASS
        '''

        cls_dirs = [train_dir + cls for cls in classes]
        top_feature_maps_dict_each_image = {}  # dict to save top feature maps for ALL images for each class
        n_maps_last_layer = 2048 if self.model_type=="resnet" else 512

        ##########  Top Feature maps for EACH IMAGE  ##########
        for dir in cls_dirs: # iterate over class
          top_filters = []  

          ### for EACH IMAGE of the class ###
          for img_path in os.listdir(dir): 
            ### Save activations of ALL feature maps for the image ###
            activations_list = self.one_image_N_top_feature_maps(layer, os.path.join(dir, img_path), plot=False, print_logits=False)
            
            ### Add top n_each_img most activated feature maps of the image to the "top filters" list ###
            top_filters.extend(list(activations_list.detach().cpu().numpy().argsort()[::-1][:n_each_img]))
          cls = dir.split("/")[-1]  # class name

          ### Add the aggregated list of the class to the dict ###
          top_feature_maps_dict_each_image[cls] = top_filters
          print(cls + " done.")

        ##########  Top Feature maps for EACH CLASS  ##########
        top_feature_map_dict_each_class = {}  # dict to save top feature maps for each class
        for cls in classes:
          ### Count the feature maps appearing in each class's aggregated list of top feature maps for ALL images ###
          frequency_counters = Counter(top_feature_maps_dict_each_image[cls])

          ### Calculate the frequency ratio for each feature map
          frequency_ratios = [frequency_counters[i]/n_imgs_dict[cls] if i in frequency_counters.keys() else 0. for i in range(n_maps_last_layer)]

          ### Add top n_each_class most frequent feature maps of the class to the dict ###
          top_feature_map_dict_each_class[cls] = np.argsort(frequency_ratios)[::-1][:n_each_class]

        ###  Eliminate feature maps that exist in more than one classes' top feature map lists  ###
        unique_top_feature_map_dict_each_class = {}
        for cls in classes:
          dict_without_this_class = {key:list(val) for key, val in top_feature_map_dict_each_class.items() if key != cls}
          if len(classes) > 2:  
            unique_top_feature_map_dict_each_class[cls] = [map for map in top_feature_map_dict_each_class[cls] if map not in set(sum(dict_without_this_class.values(), []))]
          elif len(classes) == 2:  
            unique_top_feature_map_dict_each_class[cls] = [map for map in top_feature_map_dict_each_class[cls] if map not in list(dict_without_this_class.values())[0]]

        print("# of top feature maps:", {key:len(val) for key, val in unique_top_feature_map_dict_each_class.items()})

        return  unique_top_feature_map_dict_each_class


    def visualize_patterns(self, 
                          layer,  
                          filter_n, 
                          init_size=33, 
                          lr=0.2, 
                          opt_steps=20,  
                          upscaling_steps=20, 
                          upscaling_factor=1.2, 
                          print_loss=False, 
                          plot=True):
        ''' 
        ###  VISUALIZATION #1 :  ###
          Visualize patterns captured by a single feature map
        
          || PARAMETERS ||
            layer            : (int) index of the convolutional layer to investigate feature maps 
                               *For the last convolutional layer, use -2 for resent50 & 12 for vgg16
            filter_n         : (int) index of the feature map to investigate in the layer
            init_size        : (int) intial length of the square random image
            lr               : (float) learning rate for pixel optimization
            opt_steps        : (int) number of optimization steps
            upscaling_steps  : (int) # of upscaling steps
            upscaling_factor : (float) >1, upscale factor
            log_info         : (bool) if True, log info at each optimizing iteration
                               *if activation: 0 for all iterations, there's a problem
            plot             : (bool) if True, plot the generated image at each optimizing iteration
        '''
        activations = self.register_hook(layer)

        ### Generate a random image ###
        img = np.uint8(np.random.uniform(150, 180, (init_size, init_size, 3)))/255  
        sz = init_size
        if print_loss:
            plt.imshow(img)
            plt.title("original random image")
            plt.show()

        ### Upscale the image (upscaling_steps) times ###
        for upscale_i in range(upscaling_steps):  
            ### Attach graients to the optimized image ###
            img_var = torch.autograd.Variable(torch.Tensor(img.transpose((2,0,1))).cuda().unsqueeze(0), requires_grad=True)
            
            ### Define Optimizer to update the image pixels ###
            optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)

            ### Update the image's pixel values for (opt_steps) times ### 
            for n in range(opt_steps):  
                optimizer.zero_grad()

                ### Pass the image through the model ###
                # Use sigmoid to restrict the image pixels between 0 and 1. 
                # Without sigmoid, the pixels can become negative. 
                self.model(torch.sigmoid(img_var)) 

                ### Maximize the activation of the (filter_n)th feature map of the requested layer ###
                loss = -activations.features[0, filter_n].mean() 
                if plot:
                    plt.imshow(activations.features[0, filter_n].detach().cpu().numpy(), cmap="gray")
                    plt.show()
                if print_loss: 
                    print("whole layer shape:", activations.features.shape) # [1, n_filter, intermediate_H, intermediate_W]
                    print("intermediate feature shape:", activations.features[0, filter_n].shape)
                    print("parameters shape:", activations.params.shape)         
                    print("activation:", activations.features[0, filter_n].mean().item())
                loss.backward()
                optimizer.step()

            if print_loss: 
                print()

            if upscale_i < upscaling_steps - 1: 
                img = img_var.detach().cpu().numpy()[0].transpose(1,2,0)
                ### Scale the optimized image up ###
                sz = int(upscaling_factor * sz)  # calculate new image size
                img = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)  
            else:
                ### for the last iteration, convert img_var into a numpy array ###
                img = torch.sigmoid(img_var).detach().cpu().numpy()[0].transpose(1,2,0)

        ### Remove hook ###
        activations.close()  

        ### Save the generated image ###
        img_name = "layer_"+str(layer)+"_filter_"+str(filter)+".jpg"
        plt.imsave(img_name, img)

        return img, img_name


    def make_img_var(self, img_path):
        ''' 
        Given a path to an image (str), convert the image into a PyTorch variable
        '''
        img = Image.open(img_path).convert('RGB')    
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        img = transform(img)[:3, :, :].unsqueeze(0)    
        img_var = torch.autograd.Variable(img.cuda(), requires_grad=True) if use_cuda else torch.autograd.Variable(img, requires_grad=True)
        
        return img, img_var


    def one_image_N_top_feature_maps(self, 
                                    layer,
                                    img_path,  
                                    plot=True, 
                                    n=100, 
                                    n_plots_horiz=10, 
                                    n_plots_vert=10, 
                                    plot_h=50, 
                                    plot_w=50,
                                    print_logits=False, 
                                    imagenet=False, 
                                    plot_overlay=True,
                                    n_top_classes=5):  
        '''
        ###  VISUALIZATION #2 :  ###
          1. Find top n feature maps for a single image.
          2. Highlight each top feature map's most attended regions of the image 
             by overlaying its activation map on top of the image.

        || PARAMETERS ||
          layer         : (int) index of the convolutional layer to investigate feature maps 
                                *For the LAST convolutional layer, use -2 for resent50 & 12 for vgg16
          img_path      : (str) path to the image to investigate
          plot          : (bool) if True, plot the top N feature maps' activation maps on the image
            /// MUST BE : n_plots_horiz * n_plots_vert = n ///
          n             : (int) # of top feature maps to plot
          n_plots_horiz : (int) # of feature maps to plot horizontally
          n_plots_vert  : (int) # of feature maps to plot vertically
            /// It's recommended that (n_plots_horiz/n_plots_vert) = (plot_h/plot_w) ///
          plot_h        : (int) height of the plot
          plot_w        : (int) width of the plot
          print_logits  : (bool) if True, print model logits (outputs) for the image
          imagenet      : (bool) if True, print_logits will print the logits for corresponding imagenet labels
          plot_overlay  : (bool) if True, overlay the top feature map on top of the image and plot the overlaid image
                                    if False, plot the original feature map only
        '''

        activations = self.register_hook(layer)

        ### Convert the image into a pytorch variable ###
        img, img_var = self.make_img_var(img_path)

        ### Pass the image through the model ###
        logits = self.model(img_var)

        ### Save the activations of ALL feature maps in the requested convolutional layer ###
        activations_list = activations.features[0].mean((1,2)).detach().cpu()

        ### Save only the top N most activated feature maps, in order of largest to smallest activations ###
        topN_activated_feature_maps = np.array(activations_list).argsort()[::-1][:n]

        if plot:
            assert n_plots_horiz*n_plots_vert==n, "n_plots_horiz*n_plots_vert must be equal to n!"

            ### Show the input image ###
            plt.imshow(np.transpose(img.squeeze(0).numpy(), (1,2,0)))
            plt.title("original image")
            plt.show()

            ### Print model outputs (logits) ###
            if print_logits:
                if imagenet:
                    ### Download imagenet labels ###
                    from urllib.request import urlretrieve
                    os.makedirs("attention_data", exist_ok=True)
                    if not os.path.isfile("attention_data/ilsvrc2012_wordnet_lemmas.txt"):
                        urlretrieve("https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt", "attention_data/ilsvrc2012_wordnet_lemmas.txt")
                    if not os.path.isfile("attention_data/ViT-B_16-224.npz"):
                        urlretrieve("https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz", "attention_data/ViT-B_16-224.npz")
                    imagenet_labels = dict(enumerate(open('attention_data/ilsvrc2012_wordnet_lemmas.txt')))
                    probs = torch.nn.Softmax(dim=-1)(logits)
                    top = torch.argsort(probs, dim=-1, descending=True)
                    for idx in top[0, :n_top_classes]:
                        print(f'{probs[0, idx.item()]:.5f} : {imagenet_labels[idx.item()]}', end='')
                else:
                    print("prediction: ", logits)

            plt.figure(figsize=(plot_w, plot_h))
            for top_i in range(n):    
                plt.subplot(n_plots_horiz, n_plots_vert, top_i+1)
                plt.title("layer "+str(layer)+" filter "+str(topN_activated_feature_maps[top_i]))
              
                if plot_overlay:
                    ### Upscale the feature maps to match the image size ###
                    img_dim = img.size(-1)
                    mask = np.array(cv2.resize(activations.features[0, topN_activated_feature_maps[top_i]].detach().cpu().numpy(), (img_dim,img_dim)))
                    if self.model_type == "resnet":
                        mask = mask*2 ### double the mask signal for resnet50
                    
                    ### Overlay the mask on top of the image ###
                    overlay = np.array([ch * mask for ch in img.detach().cpu().squeeze(0).numpy()])
                    plt.imshow(np.transpose(np.clip(overlay,0,1), (1,2,0)), cmap="gray")
                else:
                    mask = activations.features[0, topN_activated_feature_maps[top_i]].detach().cpu().numpy()
                    plt.imshow(mask, cmap="gray")
            plt.show()

        ### Plot a line plot of average activations of ALL feature maps ###
        if plot:
            plt.plot(activations_list)
            plt.xlabel("filter in layer "+str(layer))
            plt.ylabel("mean activation")
            plt.show()
        
        ### Return the activations of ALL feature maps in the requested convolutional layer ###
        return activations_list


    def one_feature_map_N_images(self, 
                                layer,
                                dataloader,  
                                filter_idx,  
                                plot=True, 
                                plot_all=False, 
                                plot_overlay=True, 
                                normalize=True,
                                folder="", 
                                class_name=""):
        '''
        ###  VISUALIZATION #3 :  ###
          Given the indice of the feature map to investigate (filter_idx), 
          plot its activation map for images in the dataloader.

        || PARAMETERS ||
          layer        : (int) index of the convolutional layer to investigate feature maps 
                              *For the last convolutional layer, use -2 for resent50 & 12 for vgg16
          dataloader   : (torch.utils.data.dataloader object) dataloader containing images to plot (usually images of a single class)
          filter_idx   : (int) index of the feature map to investigate in the layer
          plot         : (bool) if True, plot the feature maps' activation maps on images in the dataloader
          plot_all     : (bool) if True, plot ALL images in the dataloader
                                if False, plot only half of the images in the dataloader (if data too large)
          plot_overlay : (bool) if True, overlay the top feature map on top of the image and plot the overlaid image
                                if False, plot the original feature map only
          normalize    : (bool) if True, normalize the mask feature map by dividing by maximum value
          folder       : (str) name of the folder to save images (if not saving, leave it as "")
          class_name   : (str) name of the class the images belong to 
        '''

        activations = self.register_hook(layer)
        mean_activations_list = []

        if plot:
            n_imgs = len(dataloader.dataset) if plot_all else int(len(dataloader.dataset)/2)
            n_plots_vert, n_plots_horiz = 10, 2*(int(n_imgs/10)+1)
            plot_w, plot_h = 50, (50*n_plots_horiz/10) + 1
            plt.figure(figsize=(plot_w, plot_h))
        
        plot_i = 1
        for batch_i, (img_batch, _) in enumerate(dataloader):
            if (plot is False) or (plot_all is True) or (batch_i%2 != 0):  # only do odd batch (not enough RAM)
                b = img_batch.size(0)
                if use_cuda:
                    img_batch = img_batch.cuda() 
                
                ### Pass the batch of images through the model ###
                self.model(img_batch)

                ### Save only the requested feature map's activation for the images ###
                feat = activations.features[:, filter_idx] 
                for img_i in range(b):
                    ### Compute the average of the 7x7 activation map ###
                    mean_activation = feat[img_i].mean((0,1)).item()
                    mean_activations_list.append(mean_activation)
                    if plot:
                        plt.subplot(n_plots_horiz, n_plots_vert, plot_i)
                        plt.imshow(np.transpose(img_batch[img_i].detach().cpu().numpy(), (1,2,0)))
                        plot_i += 1
                        plt.subplot(n_plots_horiz, n_plots_vert, plot_i)
                        plt.title(str(mean_activation), fontdict={'fontsize':20})

                        ### Upscale the feature maps to match the image size ###
                        img_dim = img_batch[img_i].size(-1)
                        mask = np.array(cv2.resize(feat[img_i].detach().cpu().numpy(), (img_dim, img_dim)))
                        plt.axis("off")
                        if plot_overlay:
                            if self.model_type == "resnet":
                                mask = mask*2  ### double the mask signal for resnet50
                            else:
                                if normalize:
                                    mask = mask/mask.max()

                            ### Overlay the mask on top of the image ###
                            overlay = np.array([ch * mask for ch in img_batch[img_i].detach().cpu().squeeze(0).numpy()])
                            plt.imshow(np.transpose(np.clip(overlay, 0, 1), (1,2,0)), cmap="gray")

                            ### Save the masked images ###
                            if folder:
                                if not os.path.exists(folder):
                                   os.makedirs(folder)
                                if not os.path.exists(folder+ "/masked_" + class_name):
                                   os.makedirs(folder+ "/masked_" + class_name)
                                plt.imsave(folder + "/masked_" + class_name + "_" + str(plot_i) + ".jpg", 
                                           np.transpose(np.clip(overlay, 0, 1), (1,2,0)))
                        else:
                            plt.imshow(mask, cmap="gray")
                        plot_i += 1
        if plot:
            plt.show()

        return mean_activations_list


    def M_feature_maps_N_images(self, 
                                layer, 
                                dataloader, 
                                filter_idxs, 
                                plot=True, 
                                plot_all=False, 
                                plot_overlay=True):
        '''
        ###  VISUALIZATION #4 :  ###
          Given the indices of MULTIPLE feature maps to investigate (filter_idxs), 
          plot the SUM of their activation maps (one on top of each other) for images in the dataloader.
        
        || PARAMETERS ||
          layer        : (int) index of the convolutional layer to investigate feature maps 
                              *For the last convolutional layer, use -2 for resent50 & 12 for vgg16
          dataloader   : (torch.utils.data.dataloader object) dataloader containing images to plot (usually images of a single class)
          filter_idxs  : (list of ints) index of the feature map to investigate in the layer
          plot         : (bool) if True, plot the feature maps' activation maps on images in the dataloader
          plot_all     : (bool) if True, plot ALL images in the dataloader
                                if False, plot only half of the images in the dataloader (if data too large)
          plot_overlay : (bool) if True, overlay the top feature map on top of the image and plot the overlaid image
                                if False, plot the original feature map only
          normalize    : (bool) if True, normalize the mask feature map by dividing by maximum value
          folder       : (str) name of the folder to save images (if not saving, leave it as "")
          class_name   : (str) name of the class the images belong to 
        '''

        activations = self.register_hook(layer)
        mean_activations_list = []

        if plot:
            n_imgs = len(dataloader.dataset) if plot_all else int(len(dataloader.dataset)/2)
            n_plots_vert, n_plots_horiz = 10, 2*(int(n_imgs/10)+1)
            plot_w, plot_h = 50, (50*n_plots_horiz/10) + 1
            plt.figure(figsize=(plot_w, plot_h))

        plot_i = 1
        save_i = 1
        for batch_i, (img, _) in enumerate(dataloader):
            if (plot is False) or (plot_all is True) or (batch_i%2 != 0):  # only do odd batch (not enough RAM)
                b = img.size(0)
                if use_cuda:
                    img = img.cuda() 
                self.model(img)
                
                for img_i in range(b):
                    mask = np.zeros((224,224))
                    mean_activation = 0
                    for filter_idx in filter_idxs:
                        feat = activations.features[:, filter_idx] 
                        mask += np.array(cv2.resize(feat[img_i].detach().cpu().numpy(), (224, 224)))
                        mean_activation += feat[img_i].mean((0,1)).item()
                    mean_activations_list.append(mean_activation)
                    overlay = np.array([ch * np.clip(mask, 0, 1) for ch in img[img_i].detach().cpu().squeeze(0).numpy()])
                    plt.imsave("masked_with_gun_filters/knife/masked_{}.jpg".format(save_i), np.transpose(np.clip(overlay, 0, 1)))
                    save_i+=1
                    if plot:
                        plt.subplot(n_plots_horiz, n_plots_vert, plot_i)
                        plt.imshow(np.transpose(img[img_i].detach().cpu().numpy(), (1,2,0)))
                        plot_i += 1
                        plt.subplot(n_plots_horiz, n_plots_vert, plot_i)
                        plt.title(str(mean_activation), fontdict={'fontsize':20})
                        plt.axis("off")
                        if plot_overlay:
                            #overlay = np.array([ch * np.clip(mask, 0, 1) for ch in img[img_i].detach().cpu().squeeze(0).numpy()])
                            plt.imshow(np.transpose(np.clip(overlay, 0, 1), (1,2,0)), cmap="gray")
                            plt.imsave("benign_inch/masked_{}.jpg".format(plot_i), np.transpose(np.clip(overlay, 0, 1)))
                        else:
                            plt.imshow(mask, cmap="gray")
                        plot_i += 1
        if plot:
            plt.show()

        return mean_activations_list


    def sum_top_feature_maps_by_class(self,
                                      layer,
                                      transform, 
                                      img_dir,
                                      top_feature_maps_dict=None, 
                                      training_imgs_dir=None,
                                      classes=None,
                                      n_imgs_dict=None,
                                      plot=True,
                                      colours=[c[4:] for c in list(mcolors.TABLEAU_COLORS)]*1000):
      '''
      ### Visualization #5 ###
        Plot the SUM of activations of each class's top feature maps for each image, 
        for all classes in the same plot
        || PARAMETERS ||
          layer     : (int) if using last convolutional layer, use -2 for resnet & 12 for vgg16
          transform : (torchvision.transforms object) transform to be applied to each test image
          img_dir   : (str) address of the folder containing image folders
                      *image folders' names must be the same as target class names 
             /// Either pass top_feature_maps_dict OR (train_dir, classes, and n_imgs_dict). ///
          top_feature_maps_dict : (dict) (key, value)=(class name, list of top feature maps for that class)
                                  e.g. {"cat":[1,3,5], "dog":[2,4,8]}
          train_dir     : (str) address of the folder that contains training data including "/" at the end  e.g. "train_data/"
          classes       : (list of strs) list containing (at least two) class names in string e.g. ["cat", "dog"]
          n_imgs_dict   : (dict) key : class name (str), value : # of training images for that class (int) e.g. {"dog":955, "cat":1857}
          plot      : (bool) show plots if True
      '''

      if top_feature_maps_dict is None:
          top_feature_maps_dict = self.find_unique_filters(layer=layer, 
                                                          train_dir=training_imgs_dir, 
                                                          classes=classes, 
                                                          n_imgs_dict=n_imgs_dict)

      sum_dicts_dict = {}  # will become a dict of dicts
      classes = os.listdir(img_dir)
      for cls_i, cls in enumerate(classes):
        sum_lists_dict = {_cls:[] for _cls in top_feature_maps_dict.keys()}   
        
        for img_path in os.listdir(os.path.join(img_dir, cls)):
          # read in the image and transform it into a torch tensor
          full_img_path = os.path.join(img_dir, cls, img_path)
          img = Image.open(full_img_path).convert('RGB')     
          img_var = transform(img)[:3, :, :].unsqueeze(0).cuda()

          # compute the activations of all feature maps for the image
          activations_list = self.one_image_N_top_feature_maps(layer, img_path=full_img_path, plot=False)

          # save the sum of only the class top feature maps' activations for each class
          for top_feature_map_cls in top_feature_maps_dict.keys():
            sum_lists_dict[top_feature_map_cls].append(sum(activations_list[top_feature_maps_dict[top_feature_map_cls]]))

        for top_feature_map_cls in top_feature_maps_dict.keys():
          sum_dicts_dict[cls] = sum_lists_dict

      if plot:
        c = {cls:colour for cls, colour in zip(classes, colours)}
        for top_feature_map_cls in top_feature_maps_dict.keys():
          plt.figure(figsize=(10,7))
          for cls in classes:
            plt.plot(sum_dicts_dict[cls][top_feature_map_cls], marker=".", color=c[cls])
          plt.title(top_feature_map_cls+" activations")
          plt.legend(classes)
          plt.show()

      return sum_dicts_dict