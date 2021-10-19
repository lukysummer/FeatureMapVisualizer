def unique_filters(FV, 
                   layer, 
                   train_dir, 
                   classes, 
                   n_imgs_dict, 
                   model_type="resnet", 
                   n1=25, 
                   n2=25):
    '''
    || PARAMETERS ||
      FV          : (FeatureMapVisualizaer class)
      layer       : (int) if using last convolutional layer, use -2 for resnet & 12 for vgg16
      train_dir   : (str) address of the folder that contains training data including "/" at the end  e.g. "train_data/"
      classes     : (list of strs) list containing (at least two) class names in string e.g. ["cat", "dog"]
      n_imgs_dict : (dict) key : class name (str), value : # of training images for that class (int) e.g. {"dog":955, "cat":1857}
      model_type  : (str) "resnet" for ResNet50 or "vgg" for VGG16
      n1          : (int) # of top feature maps to save for EACH IMAGE
      n2          : (int) # of top feature maps to save for EACH CLASS
      ec          : (bool) True if using encoder, False if using the whole model (encoder + classifier)
    '''

    cls_dirs = [train_dir + cls for cls in classes]
    top_feature_maps_dict_each_image = {}  # dict to save top feature maps for ALL images for each class
    n_maps_last_layer = 2048 if model_type=="resnet" else 512

    ##########  Top Feature maps for EACH IMAGE  ##########
    for dir in cls_dirs: # iterate over class
      top_filters = []  

      ### for EACH IMAGE of the class ###
      for img_path in os.listdir(dir): 
        ### Save activations of ALL feature maps for the image ###
        activations_list = FV.one_image(layer, os.path.join(dir, img_path), plot=False, print_logits=False)
        
        ### Add top n1 most activated feature maps of the image to the "top filters" list ###
        top_filters.extend(list(activations_list.detach().cpu().numpy().argsort()[::-1][:n1]))
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

      ### Add top n2 most frequent feature maps of the class to the dict ###
      top_feature_map_dict_each_class[cls] = np.argsort(frequency_ratios)[::-1][:n2]

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
