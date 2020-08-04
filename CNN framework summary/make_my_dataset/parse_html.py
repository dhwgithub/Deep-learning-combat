import requests
import re
import datetime

import cnn_learn.make_my_dataset.download_pic as download

'''
.  代表匹配除了\n和\r之外的任意字符
*  代表匹配0次或多次
?  跟在限制符后面是代表使用非贪婪模式匹配，因为默认的正则匹配是贪婪匹配
\d  代表匹配一个数字
+  代表匹配1个或多个
'''


def request_dandan(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
         return None


def save_imgs(url, files_item):
    html = request_dandan(url)
    pattern_str = r'https:.*?jpg'
    pattern = re.compile(pattern_str)
    result = pattern.findall(html)

    for img_src in result:
        id = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        save_dir = r'.\my_dataset\download\{}\{}.jpg'.format(files_item, id)
        download.download_pic(img_src, save_dir)


url = r'https://image.baidu.com/search/index?ct=201326592&cl=2&st=-1&lm=-1&nc=1&ie=utf-8&tn=baiduimage&ipn=r&rps=1&pv=&fm=rs9&word=%E5%BE%B7%E5%9B%BD%E7%89%A7%E7%BE%8A%E7%8A%AC%E5%B9%BC%E7%8A%AC%E5%9B%BE%E7%89%87&oriquery=%E7%89%A7%E7%BE%8A%E7%8A%AC%E5%B9%BC%E7%8A%AC%E5%9B%BE%E7%89%87&ofr=%E7%89%A7%E7%BE%8A%E7%8A%AC%E5%B9%BC%E7%8A%AC%E5%9B%BE%E7%89%87&sensitive=0'
files_item = 'dog'
save_imgs(url, files_item)
