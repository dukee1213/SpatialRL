import numpy as np

class PolyUtility:
    def __init__(self, seek_para:int = 2, averse_para:int = 2, R_max: int = 100) -> None:
        if seek_para not in {2, 3, 4, 5}:
            raise ValueError(f"seek_para must be one of {{2, 3, 4, 5}}, got {seek_para}")
        if averse_para not in {2, 3, 4, 5}:
            raise ValueError(f"averse_para must be one of {{2, 3, 4, 5}}, got {averse_para}")
        #a = [1, 10, 100**(2/3), 100**(3/4), 100**(4/5)]
        #s = [1, 0.01, 0.0001, 0.000001, 0.00000001]
        self.averse_param = 100**((averse_para-1)/averse_para)
        self.averse_param2 = 1/100**(averse_para-1)
        self.seek_param = 1/100**(seek_para-1)
        self.seek_param2 = 100**((seek_para-1)/seek_para)
        self.seek_para = seek_para
        self.averse_para = averse_para
        if not isinstance(R_max, int):
            raise ValueError("R_max must be an integer.")
        self.R_max: int = R_max
        self.R_min: int = 0

    def calU(self, wealth: float, risk_type: int = 2) -> float:
        if risk_type == 0:
            beta = self.averse_param
            beta2 = self.averse_param2
            x = 1/self.averse_para
            x2 = self.averse_para
        elif risk_type == 1:
            beta = self.seek_param
            beta2 = self.seek_param2
            x = self.seek_para
            x2 = 1/self.seek_para
        elif risk_type == 2:
            return wealth
        else:
            raise ValueError("Invalid risk_type.")
        if wealth >= 0:
            return beta*wealth**x
        else:
            return -beta2*(-wealth)**x2

uu = PolyUtility()
u = np.vectorize(uu.calU)
x1 = np.linspace(-100, 100, 400)
y1 = u(x1, 0)