import os

img_dir = "himalayan_cat"
# class_list = os.listdir(img_dir)
# print(class_list)
class_list = ["himalayan_cat"]
image_exts = ['.jpeg','.jpg', '.bmp', '.png']
cnt = 1
for breed in class_list:
    for cnt, filename in enumerate(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, filename) # path of current image
        # print(img_path)
        img_path_ext = os.path.splitext(img_path)[1] # we get the extension of image at the path
        # print(img_path_ext)
        if(img_path_ext in image_exts):
            dst = f"{str(cnt)}{img_path_ext}"
            src = f"{img_dir}/{filename}"  # foldername/filename, if .py file is outside folder
            dst = f"{img_dir}/{dst}"
            os.rename(src, dst)
