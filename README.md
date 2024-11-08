# SUREv2
This provides an update of our previous tool [SURE (SUccinct REpresentation of cells)](https://github.com/ZengFLab/SURE).

![Cell Atlas Assembly](./img/figure1.jpg)

SURE and SUREMO are metacell-centric generative models. They use self-organizing map (SOM) of metacells to represent cell distributions within 
single-omics and multi-omics datasets. SUREv2 provides broad utilities, including metacell calling, cell atlas compression, cell atlas assembly, reference mapping, reference-based positioning,
out-of-reference population detection, etc. 

SUREv2 is designed as a user-friendly package for cell atlas assembly and compression. 
- It provides a Python class **SingleOmicsAtlas** for the construction and application of compressed cell atlases. 
- It includes Python classes **SURE** and **SUREMO** for building metacell models from single omics data and single multi-omics data.
- It also offers the shell commands that users can call metacells within their data. 

## Installation
1. Download SUREv2 and enter the director
```bash
git clone https://github.com/ZengFLab/SUREv2.git & cd SUREv2
```

2. Install SUREv2
```bash
pip install .
```

3. Test whether SUREv2 has been installed
- Test the metacell command for single omics data
    ```bash
    SURE --help
    ```
- Test the metacell command for multi-omics data
    ```bash
    SUREMO --help
    ```

## Tutorials

### [Tutorial 1: Building a codebook of metacells from a single omics data](./Tutorial/tutorial_1/metacell_call_for_single_omics_dataset.ipynb)
> This tutorial demonstrate the usage of SURE in Linux shell.

### Tutorial 2: Building a codebook of metacells from a multi-omics data
### [Tutorial 3: Distribution-preserved atlas compression](./Tutorial/tutorial_3/atlas_compression.ipynb)
> This tutorial gives an example of compressing an atlas into a codebook of 500 metacells.

### Tutorial 4: Identifying out-of-reference cell populations
### Tutorial 5: Simple reference-based sketching
### Tutorial 6: Distribution-based boosted sketching
### [Tutorial 7: Hierarchical cell atlas assembly](./Tutorial/tutorial_7/hierarchical_assembly.ipynb)
> This tutorial gives an example of building an assembled atlas using datasets from different species. 
![](./img/assembly.png)

### Tutorial 8: Sketching-based cell atlas integration
### Tutorial 9: Sketching-based cell atlas assembly
### Tutorial 10: Building a reference atlas for cell type annotation
### Tutorial 11: Drug response prediction


## Citations
- If you use **SURE** to call metacells for single omics datasets and assemble single-omics atlases, please kindly cite the work:
    
    >Feng Zeng and Jiahuai Han. Building a single cell transcriptome-based coordinate system for cell ID with SURE. Submitted (2024).
    

- If you use **SUREMO** to call metacells for multi-omics datasets and assemble multi-omics atlases, please cite the following work:
    
    >Shuang-Rong Sun, Zhihan Cai, Feng Zeng. Self-organizing map of metacells and distribution-based boosted sketching for assembling atlas-level single-cell datasets. Submitted (2024)
    

- If you use **SingleOmicsAtlas** to create compressed atlases and identify out-of-reference populations, please cite our work:
    
    >Zhihan Cai, Zhibin Hu, Shuang-Rong Sun, Zexu Wang, Fan Yang, Jiahuai Han, Feng Zeng. Distribution-preserved compression of single-cell atlases for privacy-protected data dissemination and novel cell type discovery. Submitted (2024).
    