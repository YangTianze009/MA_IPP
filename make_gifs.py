import imageio
from glob import glob
import os


def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def main():
    list_img_name = []
    list_read_img = []
    path = "gifs/ipp/28384"

    # filelist = os.listdir(path)

    # print(filelist)
    filelist = os.listdir(path)
    # filelist = glob(os.path.join(path, "*.png"))
    filelist.sort()
    for i in range(len(filelist)):
        filelist[i] = path + "/" + filelist[i]
    print(filelist)

    # print(image_list)
    gif_name = path + ".gif"
    duration = 0.30
    create_gif(filelist, gif_name, duration)


if __name__ == '__main__':
    main()
