# BKAI-IGH-NeoPolyp
[Model checkpoint link](https://drive.google.com/file/d/1OfOE4R0BZV2TvodWl4Ncjds2w4FffJ2c/view?usp=sharing). Make sure that the .pth checkpoint file is placed in the same folder as the infer.py script, and to download all required libraries in requirements.txt:
```
git clone https://github.com/lucasnguyen2k4/BKAI-IGH-NeoPolyp.git
```
```
cd BKAI-IGH-NeoPolyp
```
```
pip install -r requirements.txt
```
```
python3 infer.py --image_path image.jpeg
```
A new output folder will be created containing the output segmented image.
