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


    def visualize(self, 
                  layer,  
                  filter_n, 
                  init_size=33, 
                  lr=0.2, 
                  opt_steps=20,  
                  upscaling_steps=20, 
                  upscaling_factor=1.2, 
                  print_loss=False, 
                  plot=False):
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


    def one_image(self, 
                  layer, 
                  img_path, 
                  n=5, 
                  plot=True,
                  n_plots_horiz=1, 
                  n_plots_vert=5, 
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
          layer            : (int) index of the convolutional layer to investigate feature maps 
                              *For the last convolutional layer, use -2 for resent50 & 12 for vgg16
          img_path      : (str) path to the image to investigate
          n             : (int) # of top feature maps to plot
          plot          : (bool) if True, plot the top N feature maps' activation maps on the image
          n_plots_horiz : (int) # of feature maps to plot horizontally
          n_plots_vert  : (int) # of feature maps to plot vertically
            /// n_plots_horiz * n_plots_vert MUST be equal to n ///
          plot_h        : (int) height of the plot
          plot_w        : (int) width of the plot
            /// it's recommended that (n_plots_horiz/n_plots_vert) = (plot_h/plot_w) ///
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

        ### Save the activations of the feature maps ###
        activations_list = activations.features[0].mean((1,2)).detach().cpu()

        ### Save only the top N most activated feature maps, in order of largest to smallest activations ###
        topN_activated_filters = np.array(activations_list).argsort()[::-1][:n]

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
                plt.title("layer "+str(layer)+" filter "+str(topN_activated_filters[top_i]))
              
                if plot_overlay:
                    ### Upscale the feature maps to match the image size ###
                    img_dim = img.size(-1)
                    mask = np.array(cv2.resize(activations.features[0, topN_activated_filters[top_i]].detach().cpu().numpy(), (img_dim,img_dim)))
                    if self.model_type == "resnet":
                        mask = mask*2 ### double the mask signal for resnet50
                    
                    ### Overlay the mask on top of the image ###
                    overlay = np.array([ch * mask for ch in img.detach().cpu().squeeze(0).numpy()])
                    plt.imshow(np.transpose(np.clip(overlay,0,1), (1,2,0)), cmap="gray")
                else:
                    mask = activations.features[0, topN_activated_filters[top_i]].detach().cpu().numpy()
                    plt.imshow(mask, cmap="gray")
            plt.show()

        ### Plot a line plot of average activations of ALL feature maps ###
        if plot:
            plt.plot(activations_list)
            plt.xlabel("filter in layer "+str(layer))
            plt.ylabel("mean activation")
            plt.show()
        
        return activations_list


    def one_filter(self, 
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


    def multiple_filter(self, 
                        layer, 
                        dataloader, 
                        filter_idxs, 
                        plot=True, 
                        plot_all=False, 
                        plot_overlay=True):
        '''
        ###  VISUALIZATION #4 :  ###
          Given the indices of the feature maps to investigate (filter_idxs), 
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
