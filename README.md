<!-- <p align="center">
  <img src="asset/logo.png"  height=120>
</p> -->


# <div align="center"> Efficient Medical Image Segmentation with Rectifed Flow-Based Diffusion Transformer<div> 
### <div align="center"> ICONIP 2025 poster <div> 
<div align="center">

  <!-- Paper (Springer) -->
  <a href="https://link.springer.com/chapter/10.1007/978-981-95-4100-3_1">
    <img src="https://img.shields.io/static/v1?label=Paper&message=ICONIP%202025&color=1E5AA8&logo=springer" />
  </a> &ensp;

  <!-- DOI -->
  <a href="https://doi.org/10.1007/978-981-95-4100-3_1">
    <img src="https://img.shields.io/static/v1?label=DOI&message=10.1007%2F978-981-95-4100-3_1&color=orange&logo=doi" />
  </a> &ensp;

  <!-- Code -->
  <a href="https://github.com/kato0116/Efficient_Medical_Image_Segmentation_with_RF-Based_Diffusion_Transformer">
    <img src="https://img.shields.io/static/v1?label=Code&message=GitHub&color=black&logo=github" />
  </a>

</div>

> [**Efficient Medical Image Segmentation with Rectifed Flow-Based Diffusion Transformer**](https://link.springer.com/chapter/10.1007/978-981-95-4100-3_1)<br>
> [Shunichi Kato](https://scholar.google.co.jp/citations?user=EeT3vBcAAAAJ&hl=ja), [Masaki Nishimura](https://scholar.google.co.jp/citations?user=Uel-5SgAAAAJ&hl=ja&oi=sra), 
> Takaya Ueda, Yu Song,
> [Ikuko Nishikawa](https://scholar.google.co.jp/citations?view_op=list_works&hl=ja&hl=ja&user=cWumpokAAAAJ&pagesize=80);,
> <br>Ritsumeikan University<br>


## Model overview
![Model overview](assets/overview.png)

### Training Procedure (Rectified Flow for Segmentation)

```python
# Pseudocode
Input:
    D = {(x_i, y_i)} : Dataset of segmentation map and medical image pairs
Parameters:
    E_x : Pretrained VAE encoder (weights are frozen)
    v_theta : Trainable velocity network

repeat:
    (x_i, y_i) ← sample from D
    z0 ← sample from N(0, I)
    t ← Uniform(0, 1)
    
    z1 = E_x(x_i)
    zt = (1 - t) * z0 + t * z1
    
    L = || (z1 - z0) - v_theta(zt, y_i, t) ||^2
    Update v_theta using ∇L
    
until convergence
```

### Inference: Euler Method with Intermediate Ensembles

```python
# Pseudocode

Input:
    y : Conditioning medical image
    N : Number of Euler steps
    S : Ensemble size

Parameters:
    v_theta : Pretrained velocity network (fixed)

Initialize:
    dt = 1 / N
    x1 ~ N(0, I)

for i in {0 ... N-1}:
    t = i / N
    
    # Ensemble sampling
    for s in {0 ... S-1}:
        x0_s ~ N(0, I)
        xt_s = t * x1 + (1 - t) * x0_s
        u_s  = v_theta(xt_s, t, y)
    
    # Ensemble averaging
    u_bar  = (1/S) * sum(u_s)
    xt_bar = (1/S) * sum(xt_s)
    
    # Euler update
    x1 = xt_bar + u_bar * dt * (N - i)

return x1  # Predicted mask
```

## Requirement
Python >= 3.8.16
```bash
pip install -r requirement.txt
```
## TODO LIST

- [ ] Release REFUGE and DDIT dataloaders and examples
- [ ] Sample and Vis in training
- [ ] Release pre processing and post processing
- [ ] Deploy on HuggingFace





## Cite
Please cite
~~~
@inproceedings{kato2025efficient,
  title={Efficient Medical Image Segmentation with Rectified Flow-Based Diffusion Transformer},
  author={Kato, Shunichi and Nishimura, Masaki and Ueda, Takaya and Song, Yu and Nishikawa, Ikuko},
  booktitle={International Conference on Neural Information Processing},
  pages={3--16},
  year={2025},
  organization={Springer}
}
~~~















