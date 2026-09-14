# -*- coding: utf-8 -*-
import inspect
import sys
sys.path.insert(0, r'C:\Users\12951\Desktop\12323\shootsim')
import replay
print(replay.__file__)
src = inspect.getsource(replay._structured_episodes)
print(src[:2000])
