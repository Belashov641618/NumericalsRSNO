import torch
import numpy

from typing import Union

_numpy_float = [numpy.float16, numpy.float32, numpy.float64]
_numpy_complex = [numpy.complex64, numpy.complex128]
_torch_float = [torch.float, torch.float16, torch.float32, torch.float64]
_torch_complex = [torch.complex32, torch.complex64, torch.complex128]


_decimal_powers         = [-18,         -15,            -12,            -9,             -6,             -3,         0,      +3,         +6,         +9,         +12,        +15,        +18]
_decimal_prefixes_eng   = ['а',         'f',            'p',            'n',            'μ',            'm',        '',     'k',        'M',        'G',        'T',        'P',        'E']
_decimal_prefixes_rus   = ['а',         'ф',            'п',            'н',            'мк',           'м',        '',     'к',        'М',        'Г',        'Т',        'П',        'Э']
_decimal_scientific     = ['·10⁻¹⁸', '·10⁻¹⁵', '·10⁻¹²', '·10⁻⁹', '·10⁻⁶', '·10⁻³', '', '·10³', '·10⁶', '·10⁹', '·10¹²', '·10¹⁵', '·10¹⁸']

class EngineeringFormater:
    _decimal_powers:list[int] = [-18, -15, -12, -9,  -6,   -3,  0,  +3,  +6,  +9,  +12, +15, +18]
    _decimal_prefixes:list[str]

    def __init__(self):
        self.rus()
    def rus(self):
        self._decimal_prefixes = _decimal_prefixes_rus
    def eng(self):
        self._decimal_prefixes = _decimal_prefixes_eng
    def scientific(self):
        self._decimal_prefixes = _decimal_scientific

    def _float(self, x:float):
        sign = (2*(x > 0) - 1)
        x = numpy.abs(x)
        index = 0
        for power_ in self._decimal_powers[1:]:
            if x >= 10.**power_:
                index += 1
            else: break
        power = self._decimal_powers[index]
        letter = self._decimal_prefixes[index]
        x = sign * x * 10.**(-power)
        return x, letter, power
    def _complex(self, x:complex):
        return self._float(x.real), self._float(x.imag)
    def _numpy_float(self, x:numpy.ndarray):
        mean:float = numpy.mean(x**4.)**0.25
        mean, letter, power = self._float(mean)
        return x * 10.**(-power), letter, power
    def _numpy_complex(self, x:numpy.ndarray):
        return self._numpy_float(x.real), self._numpy_float(x.imag)
    def _torch_float(self, x:torch.Tensor):
        mean:float = (torch.mean(x**4)**0.25).item()
        mean, letter, power = self._float(mean)
        return x * 10.**(-power), letter, power
    def _torch_complex(self, x:torch.Tensor):
        return self._torch_float(x.real), self._torch_float(x.imag)

    def __call__(self, x:Union[float, complex, torch.Tensor, numpy.ndarray], unit:str='', nums:int=3, space:str=' '):
        if isinstance(x, float):
            x, letter, power = self._float(x)
            return f"{round(x, nums)}{space+letter+unit}"
        elif isinstance(x, complex):
            (x_r, letter_r, power_r), (x_i, letter_i, power_i) = self._complex(x)
            return f"{round(x_r, nums)}{space+letter_r+unit} + {round(x_i, nums)}i{space+letter_i+unit}"
        elif isinstance(x, numpy.ndarray):
            if x.dtype in _numpy_float:
                raise NotImplementedError
            elif x.dtype in _numpy_complex:
                raise NotImplementedError
            else: raise TypeError
        elif isinstance(x, torch.Tensor):
            if x.dtype in _torch_float:
                raise NotImplementedError
            elif x.dtype in _torch_complex:
                raise NotImplementedError
            else: raise TypeError
        else: raise TypeError

    def params(self, x:Union[float, complex, torch.Tensor, numpy.ndarray], unit:str='',):
        if isinstance(x, float):
            x, letter, power = self._float(x)
            return letter+unit, power
        elif isinstance(x, complex):
            return self.params(x.real, unit), self.params(x.imag, unit)
        elif isinstance(x, numpy.ndarray):
            if x.dtype in _numpy_float:
                x, letter, power = self._numpy_float(x)
                return letter+unit, power
            elif x.dtype in _numpy_complex:
                return self.params(x.real, unit), self.params(x.imag, unit)
            else: raise TypeError
        elif isinstance(x, torch.Tensor):
            if x.dtype in _torch_float:
                x, letter, power = self._torch_float(x)
                return letter+unit, power
            elif x.dtype in _torch_complex:
                return self.params(x.real, unit), self.params(x.imag, unit)
            else: raise TypeError
        else: raise TypeError
    @staticmethod
    def formatter(letter:str, power:float, nums:int=3, space:str=' ', include_letter:bool=False):
        if not include_letter:
            letter = ''
            space = ''
        def function(x:float, _:int):
            return f"{round(x*10**(-power), nums)}{space}{letter}"
        return function
    def autoformatter(self, x:Union[float, complex, torch.Tensor, numpy.ndarray], unit:str='', nums:int=3, space:str=' ', include_letter:bool=True, accept_real:bool=True):
        letter, power = self.params(x, unit)
        if isinstance(letter, tuple) and isinstance(power, tuple):
            if accept_real:
                letter, power = letter
            else:
                letter, power = power
        return self.formatter(letter, power, nums, space, include_letter)
    def separatedformatter(self, x:Union[float, complex, torch.Tensor, numpy.ndarray], unit:str='', nums:int=3, accept_real:bool=True):
        letter, power = self.params(x, unit)
        if isinstance(letter, tuple) and isinstance(power, tuple):
            if accept_real:
                letter, power = letter
            else:
                letter, power = power
        return self.formatter(letter, power, nums, '', False), letter
engineering = EngineeringFormater()

scientific = EngineeringFormater()
scientific.scientific()