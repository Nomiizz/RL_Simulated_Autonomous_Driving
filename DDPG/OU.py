import numpy as np 

class OU():

    @staticmethod
    def OUnoise(x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)