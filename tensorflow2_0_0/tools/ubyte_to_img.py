'''
将train-images.idx3-ubyte等格式文件转化为图片格式以及文本标签
'''

from PIL import Image
import struct


def read_image(filename):
    f = open(filename, 'rb')

    index = 0
    buf = f.read()

    f.close()

    magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    for i in range(images):
        # for i in xrange(2000):
        image = Image.new('L', (columns, rows))

        for x in range(rows):
            for y in range(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')

        # print('save ' + str(i) + 'image')
        image.save(r'E:\pycharm\tensorflow-learn\tensorflow2_0_0\datasets\MNIST\train_img\\' + str(i) + '.jpg')
    print('ok')


def read_label(filename, saveFilename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()

    f.close()

    magic, labels = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')

    labelArr = [0] * labels

    for x in range(labels):
        labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')

    save = open(saveFilename, 'w')

    save.write(','.join(map(lambda x: str(x), labelArr)))
    save.write('\n')

    save.close()
    print('save labels success')


if __name__ == '__main__':
    # 以下是1万个检测数据。
    read_image(r'E:\pycharm\tensorflow-learn\datasets\MNIST\train-images.idx3-ubyte')
    read_label(r'E:\pycharm\tensorflow-learn\datasets\MNIST\train-labels.idx1-ubyte', r'E:\pycharm\tensorflow-learn\tensorflow2_0_0\datasets\MNIST\train_label.txt')

# 以下是6万个训练数据。
# read_image('train-images.idx3-ubyte')
# read_label('train-labels.idx1-ubyte', 'test/label.txt')