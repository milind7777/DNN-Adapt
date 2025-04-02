To install models:
```
cd models
pip install -r requirements.txt
python3 get_models.py
```

To run DNN-Adapt:
```
clang++ -std=c++17 DNN-Adapt/main.cpp -o run
./run -m models
```

WIP:
* Request processor - Milind
* Load simulator    - Milind
* GPU executor      - Venkat