# Improved Modules for Infrared Small Target Detection

This repository contains PyTorch implementations of several novel modules designed to enhance the detection of small targets, particularly in infrared imagery. These modules, including MSGRA, AFBPE, ADRC, and PHDE, leverage unique mathematical and physical principles to improve feature representation and target discrimination.

## Modules Overview

### 1.  MSGRA: Multi-Scale Green-Radial Attention

* **Principle:** MSGRA employs multi-scale Gaussian radial kernels to capture information at different spatial frequencies, enhancing the representation of small, bright targets.
* **Analogy:** Similar to using multiple lenses with varying focal lengths to observe a night sky, MSGRA combines different levels of blur to effectively highlight small stars (targets).
* **Code:** \[MSGRA.py](MSGRA.py)

### 2.  AFBPE: Adaptive Fractional Biharmonic Position Encoding

* **Principle:** AFBPE integrates fractional Laplacian and biharmonic operators to encode both edge details and broader structural information, improving target localization.
* **Analogy:** AFBPE is comparable to analyzing terrain by considering both water permeation (fractional Laplacian) and the curvature of the land (biharmonic), allowing for a comprehensive understanding of the landscape.
* **Code:** \[AFB-PE.py](AFB-PE.py)

### 3.  ADRC: Adaptive Discrete-Ricci Curvature Position Encoding

* **Principle:** ADRC approximates discrete Ollivier-Ricci curvature to differentiate between point-like targets (high curvature) and linear edges (low curvature), reducing false alarms.
* **Analogy:** ADRC functions like a topographical map analysis tool, emphasizing peaks and diminishing the significance of ridges, effectively highlighting small, elevated features.
* **Code:** \[ADRC-PE.py](ADRC-PE.py)

### 4.  PHDE: Persistent Heat-Diffusion Encoding

* **Principle:** PHDE simulates heat diffusion across feature maps and uses persistent homology to emphasize features that are stable over multiple scales, suppressing noise and enhancing small target representation.
* **Analogy:** PHDE is similar to observing how heat dissipates across a metal plate; small hot spots cool quickly, while larger areas remain warm longer, helping to distinguish transient noise from significant structures.
* **Code:** \[PHDE.py](PHDE.py)

## Experimental Results

The effectiveness of these modules has been validated through experiments on the IRSTD-1K dataset, using the YOLOv8n-p2 model as a baseline. Key findings include:

* **MSGRA:** Excels at enhancing both precision and recall when used in shallow layers.
* **AFBPE:** Primarily improves precision by emphasizing positional accuracy.
* **ADRC:** Significantly boosts precision by effectively discriminating targets from edges, though with a slight trade-off in recall.
* **PHDE:** Performs optimally in deeper layers, leveraging richer semantic information to achieve a superior balance of precision and recall.

A detailed analysis of the results, including comparative performance metrics, can be found in the following document:

* \[python.md](python.md)

## Performance Summary

| Model              | Layers | Params (M) | GFLOPs | Box P | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :---------------- | :----- | :--------- | :----- | :---- | :----- | :------ | :----------- |
| YOLOv8n-p2 (Base)  | 207    | 2.921      | 12.2   | 0.825 | 0.807  | 0.841   | 0.374        |
| YOLOv8n-MSGRA     | 234    | 2.922      | 12.2   | 0.860 | 0.821  | 0.850   | 0.384        |
| Yolov8n-AFBPE     | 323    | 2.927      | 12.6   | 0.853 | 0.807  | 0.853   | 0.380        |
| YOLOv8n-ADRC      | 325    | 2.926      | 12.4   | 0.875 | 0.792  | 0.869   | 0.383        |
| YOLOv8n-PHDE      | 321    | 2.926      | 12.4   | 0.855 | 0.764  | 0.829   | 0.382        |
| YOLOv8n-PHDE-n5   | 308    | 2.930      | 12.3   | 0.885 | 0.774  | 0.840   | 0.386        |
| YOLOv8n-PHDE-n7   | 308    | 2.944      | 12.3   | 0.866 | 0.814  | 0.843   | 0.390        |

## Usage

To integrate these modules into your PyTorch projects, simply import them and incorporate them into your network architectures. Examples of usage are provided within each module's code.

## Citation

If you use these modules in your research, please cite the original work.

## License

This project is licensed under the MIT License.

Copyright (c) \[Year] CrepuscularIRIS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
