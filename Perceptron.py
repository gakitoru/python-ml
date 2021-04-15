import numpy as np

class Perceptron(object):
 
    """ パーセプトロンの分類器

    パラメータ
    -------------
    eta : float
    　　　学習率(0.0より大きく1.0以下の値)
    n_iter : int
    　　　トレーニングデータのトレーニング回数
    random_state : int
    　　　重みを初期化するための乱数シード


    属性
    -------------
    w_ : 一次元配列
    　　適合後の重み
    errors_ : リスト
    　　各エポックでの誤分類(更新)の数

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ トレーニングデータに適合させる

        パラメータ
        ----------
        X : 配列のようなデータ構造}, shape = [n_samples, n_features]
        　　トレーニングデータ
        　　n_samplesはサンプルの個数, n_featuresは特徴量の個数
        y : 配列のようなデータ構造, shape = [n_samples]
        　　目的変数
        
        戻り値
        ----------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter): # トレーニング回数分トレーニングデータを反復
            errors = 0
            for xi, target in zip(X, y): # 各サンプルで重みを更新
                # 重み w1, w2, ...の更新
                # dwj = eta*(y_t - y_p)*xj
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)





    
                
