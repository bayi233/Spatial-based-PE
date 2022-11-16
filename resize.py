import argparse
import os
from PIL import Image


def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)     ##第二项表示高质量

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):   ##查看是否有output_dir这个目录，如果没有就创建一个
        os.makedirs(output_dir)

    images = os.listdir(image_dir)   ##返回指定文件夹内文件名字的列表
    num_images = len(images)
    for i, image in enumerate(images):   ##根据名字依次打开文件
        with open(os.path.join(image_dir, image), 'r+b') as f: ##拼接路径
            with Image.open(f) as img:  ##打开图片
                img = resize_image(img, size)   ##resize
                img.save(os.path.join(output_dir, image), img.format)    ##图片保存到拼接的路径，其格式与原先一样
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))
'''
def main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]
    resize_images(image_dir, output_dir, image_size)
'''

def main(args):
    splits = ['train', 'val']
    years = ['2014']  ##2014

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for split in splits:
        for year in years:
            # build path for input and output dataset
            dataset = split + year
            image_dir = os.path.join(args.image_dir, dataset)
            output_dir = os.path.join(args.output_dir, dataset)

            image_size = [args.image_size, args.image_size]
            resize_images(image_dir, output_dir, image_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--image_dir', type=str, default= 'E:/datasets/COCO2014-2015/images',
    #                     help='directory for train images')
    # parser.add_argument('--output_dir', type=str, default='E:/datasets/COCO2014-2015/images/val_resized2014',
    #                     help='directory for saving resized images')
    # parser.add_argument('--image_size', type=int, default=256,
    #                     help='size for image after processing')

    # parser.add_argument('--image_dir', type=str, default= 'E:/Pytorch_room/Pytorch_image_caption/data/image_datang',
    #                     help='directory for train images')
    # parser.add_argument('--output_dir', type=str, default='E:/Pytorch_room/Pytorch_image_caption/data/image_datang_resized',
    #                     help='directory for saving resized images')
    # parser.add_argument('--image_size', type=int, default=256,
    #                     help='size for image after processing')
    parser.add_argument('--image_dir', type=str, default= '/media/huashuo/mydisk/cocodataset/coco2014',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='/media/huashuo/mydisk/cocodataset/coco2014/384_resized',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=384,
                        help='size for image after processing')

    args = parser.parse_args()
    main(args)