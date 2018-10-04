from PIL import Image
import glob
import os

#images = os.listdir("./data/") #ディレクトリのパス
images = glob.glob("./data2/*/*") #ディレクトリのパス

for i in images:
    if i.endswith('.jpg' or '.jpeg'): #拡張子
        filename = "./" + i
        img = Image.open(filename)
        print("Load {}".format(filename))
        img = img.resize((256, 256))
        img.save(filename) #上書き保存
    else: continue
