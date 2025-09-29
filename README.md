# SGRRG

[CMIG 2025] This is the official implementation of [CAMANet: Class Activation Map Guided Attention Network for Radiology Report Generation](https://www.sciencedirect.com/science/article/pii/S0895611125001533) accepted to  IEEE Journal of Biomedical and Health Informatics (J-BHI), 2024.

## Abstract

Radiology report generation (RRG) methods often lack sufficient medical knowledge to produce clinically accurate reports. A scene graph provides comprehensive information for describing objects within an image. However, automatically generated radiology scene graphs (RSG) may contain noise annotations and highly overlapping regions, posing challenges in utilizing RSG to enhance RRG. To this end, we propose Scene Graph aided RRG (SGRRG), a framework that leverages an automatically generated RSG and copes with noisy supervision problems in the RSG with a transformer-based module, effectively distilling medical knowledge in an end-to-end manner. SGRRG is composed of a dedicated scene graph encoder responsible for translating the radiography into a RSG, and a scene graph-aided decoder that takes advantage of both patch-level and region-level visual information and mitigates the noisy annotation problem in the RSG. The incorporation of both patch-level and region-level features, alongside the integration of the essential RSG construction modules, enhances our framework’s flexibility and robustness, enabling it to readily exploit prior advanced RRG techniques. A fine-grained, sentence-level attention method is designed to better distill the RSG information. Additionally, we introduce two proxy tasks to enhance the model’s ability to produce clinically accurate reports. Extensive experiments demonstrate that SGRRG outperforms previous state-of-the-art methods in report generation and can better capture abnormal findings.

<img src='architecture.png'>

## Citations

If you use or extend our work, please cite our paper.
```
@article{wang2025sgrrg,
  title={SGRRG: Leveraging radiology scene graphs for improved and abnormality-aware radiology report generation},
  author={Wang, Jun and Zhu, Lixing and Bhalerao, Abhir and He, Yulan},
  journal={Computerized Medical Imaging and Graphics},
  pages={102644},
  year={2025},
  publisher={Elsevier}
}
```

## Prerequisites

The following packages are required to run the scripts:
- [Python >= 3.6]
- [PyTorch = 1.6]
- [Torchvision]
- [Pycocoevalcap]

* You can create the environment via conda:
```bash
conda env create --name [env_name] --file env.yml
```


## Download Trained Models
You can download the trained models [here](https://drive.google.com/drive/folders/1iv_SNS6GGHKImLrFITdScMor4hvwin77?usp=sharing).

## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://openi.nlm.nih.gov/faq).

For `MIMIC-CXR`, you can download the dataset from [here](https://physionet.org/content/mimic-cxr/2.0.0/).

After downloading the datasets, put them in the directory `data`.

## Pseudo Label Generation
You can generate the pesudo label for each dataset by leveraging the automatic labeler  [ChexBert](https://github.com/stanfordmlgroup/CheXbert).

We also provide the generated labels in the files directory.

Our experiments were done on RTX A6000 card.

## Train on IU X-Ray

Run `bash run_iu.sh` to train a model on the IU X-Ray data.

## Run on MIMIC-CXR

Run `bash run_mimic.sh` to train a model on the MIMIC-CXR data.

## Test on MIMIC-CXR

Run `bash test_mimic.sh` to train a model on the MIMIC-CXR data.

## Acknowledgment
Our project references the codes in the following repos. Thanks for their works and sharing.
- [R2GenCMN](https://github.com/cuhksz-nlp/R2GenCMN)
