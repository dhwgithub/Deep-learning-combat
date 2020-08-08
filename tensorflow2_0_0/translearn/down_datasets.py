import pathlib
import tensorflow as tf
'''
共用数据集下载链接（也可直接手动下载数据集）：
https://www.tensorflow.org/datasets/catalog/overview#top_of_page
'''
data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
data_root = pathlib.Path(data_root_orig)
print(data_root)

for item in data_root.iterdir():
  print(item)
