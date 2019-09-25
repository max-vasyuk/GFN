# GFN
**"Gated Fusion Network for Image Deblurring and Super-Resolution"**

## How to test:
1. Git clone this repository.
```bash
$git clone https://github.com/max-vasyuk/GFN.git
$cd GFN
```
2. Download the trained model ``model_gfn.pkl`` from [here](https://drive.google.com/open?id=1AC2t7f3-BMsDvWPsO2k4Sly9r4Pwtd9i) and move the model to ``GFN/models`` folder.

3. Create class TestModel with path to model.
```bash
from inference import TestModel
tm = TestModel(*path_to_model*)
```

4. Execute function of prediction for inference.
```bash
tm.predict('test.jpeg')
```
5. The result will be in the root with the input image in ``result/`` folder.
