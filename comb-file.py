import os
from imutils import paths
from tqdm import tqdm
from shutil import copyfile
from absl import app


def main(argv):
    if(len(argv)==2):
        src_dir = argv[1]
    else:
        raise app.UsageError('Need src dir.'+' or Too many command-line arguments.')
    dst_dir = src_dir + "/dst/"
    os.mkdir(dst_dir)
    images = list(paths.list_images(src_dir))
    print(len(images))
    for image in tqdm(images):
        image_file = image.split("/")[-1]
        parent_path = image.split("/")[-2]
        dst_file = dst_dir+parent_path+'_'+image_file
        print("src: "+src_dir+image, "  dst: "+dst_file)
        copyfile(image, dst_file)

if __name__ == "__main__":
    app.run(main)