import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    """MNISTデータをpathからロード"""
    # 引数に指定したパスを結合(ラベルや画像のパスを作成)
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    # ファイルを読み込む
    # 引数にファイル、モードを指定(rbは読み込みのバイナリモード)
    with open(labels_path, 'rb') as lbpath:
        # バイナリを文字列に変換:unpack関数の引数にフォーマット、8バイト分の
        # バイナリデータを指定してマジックナンバー、アイテムの個数を読み込む
        magic, n = struct.unpack('>II', lbpath.read(8))
        # ファイルからラベルを読み込み配列を構築:fromfile関数の引数に
        # ファイル、配列のデータ形式を指定
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        # 画像ピクセル情報の配列のサイズを変更
        # (行数:ラベルのサイズ、列数:特徴量の個数)
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
    return images, labels

