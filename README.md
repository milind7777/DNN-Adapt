To install models:
```
cd models
pip install -r requirements.txt
python3 get_models.py
```

Before running DNN-Adapt:
Install onnx-runtime and cudnn
```
bash setup.sh
```

To run DNN-Adapt:
```
bash build.sh # builds using cmake
./run -m models
```

WIP:
* Request processor  - Milind - Done
* Load simulator     - Milind - Done
* Node Runner        - Milind - Done
* Nexus algorithm    - Milind - Done
* Run schedule smart - Milind - Done
* Update schedule    - Milind - In Progress
* Metric gathering   - Milind - In Progress