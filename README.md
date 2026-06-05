# TopoSlide: Topologically-Informed Histopathology Whole Slide Image Representation Learning
[**Shahira Abousamra, Asmita Sood, Sylvia Plevritis, TopoSlide: Topologically-Informed Histopathology Whole Slide Image Representation Learning, CVPR 2026.**](https://openaccess.thecvf.com/content/CVPR2026/papers/Abousamra_TopoSlide_Topologically-Informed_Histopathology_Whole_Slide_Image_Representation_Learning_CVPR_2026_paper.pdf)
<!--a href="https://shahiraabousamra.github.io/resources/TopoSlide_abousamra_cvpr26.pdf">Open PDF File</a>-->

This repository will host the code and models for TopoSlide.

1. Set up the environment


2. Preprocessing
 
	2.1. Extract WSIs meta data.  \
	Check ```preprocessing/extract_meta_data/readme.md``` 

    2.2. Tiling:     \
	Check ```preprocessing/tiling/readme.md```  
    
	2.3. Cluster WSI patch embeddings.  \
	Check ```preprocessing/cluster/readme.md```  


3. Generate patch embeddings

4. Generate Whole Slide Image (WSI) embeddings:  \
Check ```eval_wsi_embeddings/readme.md```  


 ## 📄 Dual Licensing

This project is **dual-licensed** to support both research and commercial use:
© The Board of Trustees of The Leeland Stanford Junior University. 
This code and associated models are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the TopoSlide model(s) and their derivatives, which include models trained on outputs from the TopoSlide model(s) or datasets created from the TopoSlide model(s), is prohibited and requires prior approval. If you are a commercial entity, please contact the corresponding author.

### 🔬 Non-Commercial Use (FREE)
- **License**: [Creative Commons BY-NC 4.0](LICENSE-NONCOMMERCIAL)
- **Permitted**: Research, education, personal projects
- **Requirements**: Attribution required
- **Restrictions**: No commercial use

### 💼 Commercial Use
- **License**: [Commercial License Required](LICENSE-COMMERCIAL)  
- **Contact**: shsamra@stanford.edu, plevriti@stanford.edu for licensing terms

 ## Citation ###
    @inproceedings{abousamra2026TopoSlide,
    author      =  {Shahira Abousamra and Asmita Sood and Sylvia Plevritis},
    title       =  {TopoSlide: Topologically-Informed Histopathology Whole Slide Image Representation Learning},
    booktitle   =  {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year        =  {2026}}
