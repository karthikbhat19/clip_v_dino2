image_dict_clip = {"image_name1": ["sim_image1", "sim_image2", "sim_image3", "sim_image4", "sim_image5", "sim_image6", "sim_image7"], 
                "image_name2": ["sim_image7", "sim_image8", "sim_image9", "sim_image10", "sim_image11", "sim_image12"]}

image_dict_dino = {"image_name1": ["sim_image1", "sim_image2", "sim_image7", "sim_image9", "sim_image5", "sim_image12", "sim_image3"], 
                "image_name2": ["sim_image7", "sim_image69", "sim_image9", "sim_image10", "sim_image420", "sim_image12"]}

clip_dino_dict = {}

for im_name, im_list in image_dict_clip.items():
    clip_dino_dict[im_name] = []
    # clip_dino_dict[im_name] = list(set(im_list) - set(image_dict_dino[im_name]))  # if ordering isn't important
    for idx, im in enumerate(image_dict_dino[im_name]):
        if im_list[idx] != im:
            clip_dino_dict[im_name].append([im_list[idx], im])
    
print(clip_dino_dict)