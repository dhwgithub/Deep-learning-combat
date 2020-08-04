import urllib.request

# url = 'https://ss2.bdstatic.com/70cFvnSh_Q1YnxGkpoWK1HF6hhy/it/u=2933546464,2570290538&fm=26&gp=0.jpg'
# response = urllib.request.urlopen(url)
# img = response.read()
#
# with open(r'.\my_dataset\test.jpg', 'wb') as f:
#     f.write(img)
#
# print(response.geturl())
# # http://***./13928177_195158772185_2.jpg
# print(response.info())
# # （各种信息）
# print(response.code)

'''
img_src  图像下载路径
save_path  图像保存位置（含图片名称）
'''
def download_pic(img_src, save_path):
    response = urllib.request.urlopen(img_src)
    if response.code != 200:
        print('false one')
    img = response.read()
    with open(save_path, 'wb') as f:
        f.write(img)
