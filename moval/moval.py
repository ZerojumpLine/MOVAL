from sklearn.base import BaseEstimator

class MOVAL(BaseEstimator):

    def __init__(
        self,
        model: str = "TS",
    ):
        self.__dict__.update(locals())
    
    def __len__(self):
        raise NotImplementedError
    
    def debug(self):
        print('I am use moval beta!')
        print(self.model)
    
    