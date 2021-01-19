from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import preprocess_input
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import backend
import glob
import time
from tqdm import tqdm
import os
import numpy as np
import shutil
from sklearn.metrics import confusion_matrix
import cv2

CLASS_NUM = 10
learning_rate = 0.003
TRAIN_DIR = r"D:\Downloads\Tobacco-3482\Tobacco3482\train"
VALID_DIR = r"D:\Downloads\Tobacco-3482\Tobacco3482\val"
RANDOM_STATE = 2020
EPOCHS = 2


def get_callbacks(filepath, patience=2):
    '''
    返回回调列表
    :param filepath:
    :param patience:
    :return:
    '''
    # ReduceLROnPlateau: 当标准评估停止提升时，降低学习速率。
    # monitor: 被监测的数据。
    # factor: 学习速率被降低的因数。新的学习速率 = 学习速率 * 因数
    # patience: 没有进步的训练轮数，在这之后训练速率会被降低。
    # verbose: 整数。0：安静，1：更新信息。
    # min_lr: 学习速率的下边界。
    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, epsilon=1e-5, patience=patience, verbose=1,
                                  min_lr=0.00001)
    # 该回调函数将在（默认）每个epoch后保存模型到filepath
    # ModelCheckpoint: 在每个训练期之后保存模型。
    # filepath: 字符串，保存模型的路径。
    # monitor: 被监测的数据。
    # verbose: 详细信息模式，0 或者 1 。
    # save_best_only: 如果 save_best_only=True， 被监测数据的最佳模型就不会被覆盖。
    msave = ModelCheckpoint(
        filepath, monitor='val_accuracy', save_best_only=True, verbose=1)
    # EarlyStopping: 当被监测的数量不再提升，则停止训练。
    # monitor: 被监测的数据。
    # min_delta: 在被监测的数据中被认为是提升的最小变化， 例如，小于 min_delta 的绝对变化会被认为没有提升。
    # patience: 没有进步的训练轮数，在这之后训练就会被停止。
    earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0,
                              patience=patience * 3 + 2, verbose=1, mode='auto')
    return [lr_reduce, msave, earlystop]


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
        base_model: keras model excluding top
        nb_classes: # of classes
    Returns:
        new keras model with last layer
    """
    x = base_model.output
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='sigmoid')(x)
    model = Model(input=base_model.input, output=predictions)
    return model


def get_model(IN_WIDTH, IN_HEIGHT):
    # DenseNet169: 模型默认输入尺寸是 224x224。
    # include_top: 是否包括顶层的全连接层。
    # weights: None 代表随机初始化， 'imagenet' 代表加载在 ImageNet 上预训练的权值。
    # input_shape: 可选，输入尺寸元组，
    #              仅当 include_top=False 时有效（不然输入形状必须是 (224, 224, 3) （channels_last 格式）
    #              或 (3, 224, 224) （channels_first 格式），因为预训练模型是以这个大小训练的）。
    #              它必须为 3 个输入通道，且宽高必须不小于 32，比如 (200, 200, 3) 是一个合法的输入尺寸。
    # pooling: 可选，当 include_top 为 False 时，该参数指定了特征提取时的池化方式。
    #     None 代表不池化，直接输出最后一层卷积层的输出，该输出是一个四维张量。
    #     'avg' 代表全局平均池化（GlobalAveragePooling2D），相当于在最后一层卷积层后面再加一层全局平均池化层，输出是一个二维张量。
    #     'max' 代表全局最大池化.
    base_model = DenseNet169(
        include_top=False, weights='imagenet', input_shape=(IN_WIDTH, IN_HEIGHT, 3))

    model = add_new_last_layer(base_model, CLASS_NUM)
    model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9),
                  loss="binary_crossentropy", metrics=['accuracy'])
    model.summary()

    return model


def train_model(save_model_path, BATCH_SIZE, IN_SIZE):
    IN_WIDTH, IN_HEIGHT = IN_SIZE
    print('train_model')
    callbacks = get_callbacks(filepath=save_model_path, patience=3)
    print('get_callbacks')
    model = get_model(IN_WIDTH, IN_HEIGHT)
    print('get_model')

    # 角度训练模型修改
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # preprocessing_function: 应用于每个输入的函数。
                                                  # 这个函数会在任何其他改变之前运行。
                                                  # 这个函数需要一个参数：一张图像（秩为 3 的 Numpy 张量），
                                                  # 并且应该输出一个同尺寸的 Numpy 张量。
                                                  # preprocess_input类似于一个归一化的函数，对每个通道减均值
        horizontal_flip=False,
        vertical_flip=False,
        rotation_range=10,  # rotation_range: 整数。随机旋转的度数范围。
        shear_range=0.1  # shear_range: 浮点数。剪切强度（以弧度逆时针方向剪切角度）。
    )

    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        directory=TRAIN_DIR,  # directory: 目标目录的路径。每个类应该包含一个子目录。
                              # 任何在子目录树下的 PNG, JPG, BMP, PPM 或 TIF 图像，都将被包含在生成器中。
        target_size=(IN_WIDTH, IN_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',   # class_mode: "categorical", "binary", "sparse", "input" 或 None 之一。
                                    # 默认："categorical"。决定返回的标签数组的类型：
                                    # "categorical" 将是 2D one-hot 编码标签，
                                    # "binary" 将是 1D 二进制标签，"sparse" 将是 1D 整数标签，
                                    # "input" 将是与输入图像相同的图像（主要用于自动编码器）。
                                    # 如果为 None，不返回标签（生成器将只产生批量的图像数据，
                                    # 对于 model.predict_generator(), model.evaluate_generator() 等很有用）。
                                    # 请注意，如果 class_mode 为 None，那么数据仍然需要驻留在 directory 的子目录中才能正常工作。
        seed=RANDOM_STATE,  # seed: 可选随机种子，用于混洗和转换。
        interpolation='lanczos',  # interpolation: 在目标大小与加载图像的大小不同时，用于重新采样图像的插值方法。
                                  # PIL默认插值下采样的时候会模糊
    )

    valid_generator = valid_datagen.flow_from_directory(
        directory=VALID_DIR,
        target_size=(IN_WIDTH, IN_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=RANDOM_STATE,
        interpolation='lanczos',
        # PIL默认插值下采样的时候会模糊, Supported: nearest, bilinear, bicubic, hamming, box, lanczos
    )

    # fit_generator: 使用 Python 生成器（或 Sequence 实例）逐批生成的数据，按批次训练模型。
    # 生成器与模型并行运行，以提高效率。 例如，这可以让你在 CPU 上对图像进行实时数据增强，以在 GPU 上训练模型。
    # steps_per_epoch: 在声明一个 epoch 完成并开始下一个 epoch 之前从 generator 产生的总步数（批次样本）。
    #                  它通常应该等于你的数据集的样本数量除以批量大小。
    # max_queue_size: 整数。生成器队列的最大尺寸。 如未指定，max_queue_size 将默认为 10。
    # workers: 整数。使用的最大进程数量，如果使用基于进程的多线程。 如未指定，workers 将默认为 1。如果为 0，将在主线程上执行生成器。
    # verbose: 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。
    model.fit_generator(
        train_generator,
        steps_per_epoch=1 * (train_generator.samples // BATCH_SIZE + 1),
        epochs=EPOCHS,
        max_queue_size=1000,
        workers=2,
        verbose=1,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // BATCH_SIZE,
        # valid_generator.samples // BATCH_SIZE + 1, #len(valid_datagen)+1,
        callbacks=callbacks
    )


def load_weight(weight_path, IN_SIZE):
    '''
    Function :
        加载权重防止多次加载
    Args :
        weight_path : 权重路径
        IN_SIZE : 训练时输入尺度大小
    Return :
        权重模型
    Raises :
        无
    '''
    IN_WIDTH, IN_HEIGHT = IN_SIZE
    backend.set_learning_phase(0)
    model = get_model(IN_WIDTH, IN_HEIGHT)
    model.load_weights(weight_path)
    return model


def predict(model, image_path, IN_SIZE):
    '''
    Function :
        预测小票类别
    Args :
        model : 加载的权重模型
        image_path :小票路径
    Return :
        分类结果
    Raises :
        文件插值时出错，返回None
    '''
    IN_WIDTH, IN_HEIGHT = IN_SIZE
    try:
        # image.load_img用来加载文件，并没有形成numpy数组
        # 再使用image.img_to_array(x)可以达到和cv2.imread()一样的输出
        # interpolation='antialias'表示平滑的差值方式
        x = load_img(path=image_path, target_size=(IN_HEIGHT, IN_WIDTH, 3), interpolation='lanczos')
    except Exception:
        print('\n' + image_path + ": 文件出错 \n")
        return None
    x = img_to_array(x)
    x = preprocess_input(x)  # 对每个通道减均值
    x = x[None]  # 给x在最外层加了一个维度

    # 预测
    y = model.predict(x)
    print('(预测) y is ' + str(y))
    result = np.argmax(y[0])
    print('(预测) result is ' + str(result))
    if result == 2 and max(y[0]) < 0.8:
        y[0][2] = min(y[0])
        result = np.argmax(y[0])
    return result


def int2str(class_int):
    '''
    Function :
        类别int 转为 str
    Args :
        class_int :  类别的int
    Return :
        class_str
    Raises :
        无
    '''
    class_str = ''
    if class_int is None:
        return 'error'

    if class_int < 10:
        class_str = str(class_int // 4) + '_' + str(class_int % 4)
    else:
        class_str = 'error'
    return class_str


def evaluate(model, image_dir, IN_SIZE):
    path_pattern = image_dir + r'/*/*.jpg'  # 图片预处理
    image_paths = glob.glob(path_pattern, recursive=True)  # recursive=True表示递归的返回从给定路径开始的全路径

    total_cnt = 0
    true_cnt = 0
    y_true = []  # 真实分类列表
    y_predict = []  # 预测分类列表

    start = time.clock()

    for image_path in tqdm(image_paths):  # tqdm显示进度条提示
        print(image_path)
        wrong_save_path = './wrong'
        if not os.path.exists(wrong_save_path):
            os.mkdir(wrong_save_path)
        # result = np.where(y == np.max(y))[1][0]
        result_int = predict(model, image_path, IN_SIZE)
        if result_int is None:
            continue
        result = int2str(result_int)
        path, _ = os.path.split(image_path)
        _, gt = os.path.split(path)
        # gt = int(gt)
        total_cnt += 1
        y_true.append(gt)
        y_predict.append(result)
        # print(y)
        if gt == result \
                or (gt.split('_') == '4' and result.split('_') == '0') \
                or (gt.split('_') == '0' and result.split('_') == '4'):  # and 优先级大于 or
            true_cnt += 1
            # print(image_path + "\t\tTrue: " + str(gt) + " == " + str(result))
        else:
            # print(image_path + "\t\tFalse: " + str(gt) + " != " + str(result))
            # print(y)

            reName = os.path.basename(image_path).split('.')[
                         0] + '_gt_' + str(gt) + '_pr_' + str(result) + '.' + os.path.basename(image_path).split('.')[1]
            try:
                shutil.copy(image_path, os.path.join(wrong_save_path, reName))  # 将文件拷贝到指定文件或目录
            except Exception:
                pass

    end = time.clock()

    # 根据真实分类和预测分类生成混淆矩阵
    confusion_m = confusion_matrix(y_true, y_predict,
                                   labels=['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report',
                                           'Resume', 'Scientific'])
    print(confusion_m)
    print("Total is %d,Wrong is %d: " % (total_cnt, total_cnt - true_cnt))
    print("Precision: ", true_cnt / total_cnt)
    print('time:' + str(end - start))


def classfiy_move(model, image_dir, IN_SIZE, out_path):
    '''
    Function :
        旋转移动
    Args :
        model : 权重模型 预加载
        IN_SIZE : 训练的尺度大小
        image_dir : 源图片目录
    Return :
        无
    Raises :
        无
    '''
    path_pattern = image_dir + '/*.jpg'
    image_paths = glob.glob(path_pattern, recursive=True)
    for i in range(CLASS_NUM):
        abs_path = os.path.join(out_path, int2str(i))
        if not os.path.exists(abs_path):
            os.mkdir(abs_path)
    for filename in tqdm(image_paths):
        result_int = predict(model, filename, IN_SIZE)
        result_str = int2str(result_int)
        # 递归将一个文件或目录移至另一个位置
        shutil.move(filename, os.path.join(out_path, result_str, os.path.split(filename)[1]))


def classfiy_rotate(model, IN_SIZE, image_dir):
    '''
    Function :
        源目录的图片分类后旋转
    Args :
        model : 权重模型 预加载
        IN_SIZE : 训练的尺度大小
        image_dir : 源图片目录
    Return :
        无
    Raises :
        无
    '''
    path_pattern = image_dir + '/*.jpg'
    image_paths = glob.glob(path_pattern, recursive=True)
    cnt = 0
    for image_path in tqdm(image_paths):
        result = predict(model, image_path, IN_SIZE)
        # print(result,type(result))
        image_data = cv2.imread(image_path)
        image_data = np.rot90(image_data, result)  # 参数分别表示图形矩阵和旋转次数（正数：逆时针）
        if result != 0:
            cv2.imwrite(image_path, image_data)
            cnt += 1
    print(cnt)


def main():
    IN_SIZE = (256, 256)
    weights_path = r'.\model_weight.hdf5'
    train_model(save_model_path=weights_path, BATCH_SIZE=8, IN_SIZE=IN_SIZE)
    print('ok')
    # model = load_weight(weights_path, IN_SIZE)
    #
    # image_dir = r'D:\Downloads\Tobacco-3482\Tobacco3482\val'
    # evaluate(model, image_dir, IN_SIZE)

    # ==================================================================================================================
    # classfiy_rotate(model,IN_SIZE,r'E:\receipt_angle\val\4')
    #
    # out_path = r'D:\OCR\图片\image_classifiy'
    # classfiy_move(model, image_dir, IN_SIZE, out_path)
    pass


if __name__ == '__main__':
    main()
