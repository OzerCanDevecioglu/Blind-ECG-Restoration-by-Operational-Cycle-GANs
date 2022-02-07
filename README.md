
# Project Description

Continuous long-term monitoring of electrocardiography (ECG) signals is crucial for the early detection of cardiac abnormalities such as arrhythmia. Non-clinical ECG recordings acquired by Holter and wearable ECG sensors often suffer from severe artifacts such as baseline wander, signal cuts, motion artifacts, variations on QRS amplitude, noise, and other interferences. Usually, a set of such artifacts occur on the same ECG signal with varying severity and duration, and this makes an accurate diagnosis by machines or medical doctors extremely difficult.  Despite numerous studies that have attempted ECG denoising, they naturally fail to restore the actual ECG signal corrupted with such artifacts due to their simple and naive noise model. In this study, we propose a novel approach for blind ECG restoration using cycle-consistent generative adversarial networks (Cycle-GANs) where the quality of the signal can be improved to a clinical level ECG regardless of the type and severity of the artifacts corrupting the signal. To further boost the restoration performance, we propose 1D operational Cycle-GANs with the generative neuron model. The proposed approach has been evaluated extensively using one of the largest benchmark ECG datasets from the China Physiological Signal Challenge (CPSC-2020) with more than one million beats. Besides the quantitative and qualitative evaluations, a group of cardiologists performed medical evaluations to validate the quality and usability of the restored ECG, especially for an accurate arrhythmia diagnosis.
[Paper Link](https://arxiv.org/abs/2202.00589)


## Dataset

- [The China Physiological Signal Challenge 2020](http://2020.icbeb.org/CSPC2020), (CPSC-2020) dataset is used for training & testing.
- For training dataset, 4000 clean and corrupted segments with a duration of 10 seconds are selected. Training dataset can be downloaded from the given [link](https://drive.google.com/drive/folders/1XScA_USC8ewJ9uGVBQfCC4KvxUbwi4qp?usp=sharing)
- CPSC-2020 with R-peak label information can be downloaded from the given [link](https://drive.google.com/drive/folders/1DoXdJJ5RCwPvahX9DSUTuWdltyK6iv0m?usp=sharing)
## Run

#### Train
- Download train data to the "mats/" folder
- Start training
```http
  python 1D_Self_Operational_CycleGAN.py
```
- Start evaluation. You can download Pre-trained Network [weights](https://drive.google.com/drive/folders/1ezrWa6A69H5ccNV1y2hb_GuyLsmEk1ff?usp=sharing)
```http
  python test.py
```
- Save outputs to the "test_outputs/" folder 
- Visualize Results
```http
  python plot_outputs.py
```
## Prerequisites
- Pyton 3
- Pytorch
- Pytorch-Lightning
- [FastONN](https://github.com/junaidmalik09/fastonn) 


  
## Results

![image](https://user-images.githubusercontent.com/98646583/152834107-2a80eb37-0dc3-445e-97b9-75ef5c6a3eed.png)

![image](https://user-images.githubusercontent.com/98646583/152834186-2db88f29-199d-47c3-804c-fc21d0d34cab.png)

![image](https://user-images.githubusercontent.com/98646583/152834222-5cfd05f1-9745-40aa-af9c-8f6a9cb4ac5f.png)




  
## Citation
If you find this project useful, we would be grateful if you cite this paper：

```http
S. Kiranyaz, O. C. Devecioglu, T. Ince, J. Malik, M. Chowdhury, T. Hamid, R. Mazhar, A. Khandakar, A. Tahir, T. Rahman, and M. Gabbouj, “Blind ECG Restoration by Operational Cycle-GANs”, in arXiv Preprint, https://arxiv.org/abs/2202.00589 , Feb. 2022. 
```
If you use labeled CPSC-2020 dataset , please cite these following papers too:
```http
M. U. Zahid and S. Kiranyaz and T. Ince and O. C. Devecioglu and M. E. H. Chowdhury and A. Khandakar and A. Tahir and M. Gabbouj, “Robust R-Peak Detection in Low-Quality Holter ECGs using 1D Convolutional Neural Network”, IEEE Trans. on Biomedical Eng., vol. 69, pp. 119-128, June 2021. 
```
```http
S. Kiranyaz, J. Malik, M. U. Zahid, T. Ince, M. Chowdhury, A. Khandakar, A. Tahir, and M. Gabbouj, “Robust Peak Detection for Holter ECGs by Self-Organized Operational Neural Networks”, arXiv preprint arXiv:2110.02381. (in IEEE Trans. on Neural Networks and Learning Systems, (Minor Revision), Dec. 2021. 
```

- Implementation of this code is inspired from [Pytorch Lightning Implementation of CycleGANs](https://www.kaggle.com/bootiu/cyclegan-pytorch-lightning)
