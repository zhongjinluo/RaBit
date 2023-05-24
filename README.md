# *RaBit*

> ***RaBit***: Parametric Modeling of 3D Biped Cartoon Characters with a Topological-consistent Dataset
>
> Zhongjin Luo*, Shengcai Cai*, Jinguo Dong, Ruibo Ming, Liangdong Qiu, [Xiaohang Zhang](https://xiaohangzhan.github.io/) and [Xiaoguang Han](https://gaplab.cuhk.edu.cn/)

#### | [Paper](https://arxiv.org/abs/2303.12564) | [Project](https://gaplab.cuhk.edu.cn/projects/RaBit/) | [Dataset](https://gaplab.cuhk.edu.cn/projects/RaBit/dataset.html) |

## Introduction
Assisting people in efficiently producing visually plausible 3D characters has always been a fundamental research topic in computer vision and computer graphics. Recent learning-based approaches have achieved unprecedented accuracy and efficiency in the area of 3D real human digitization. However, none of the prior works focus on modeling 3D biped cartoon characters, which are also in great demand in gaming and filming. In this paper, we introduce *[3DBiCar](https://gaplab.cuhk.edu.cn/projects/RaBit/dataset.html)*, the first large-scale dataset of 3D biped cartoon characters, and *RaBit*, the corresponding parametric model. Our dataset contains 1,500 topologically consistent high-quality 3D textured models which are manually crafted by professional artists. Built upon the data, *RaBit* is thus designed with a SMPL-like linear blend shape model and a StyleGAN-based neural UV-texture generator, simultaneously expressing the shape, pose, and texture. To demonstrate the practicality of *[3DBiCar](https://gaplab.cuhk.edu.cn/projects/RaBit/dataset.html)* and *RaBit*, various applications are conducted, including single-view reconstruction, sketch-based modeling, and 3D cartoon animation. 

![gallery](./assets/fig_teaser.png)

## Citation

```
@inproceedings{luo2023rabit,
  title={RaBit: Parametric Modeling of 3D Biped Cartoon Characters with a Topological-consistent Dataset},
  author={Luo, Zhongjin and Cai, Shengcai and Dong, Jinguo and Ming, Ruibo and Qiu, Liangdong and Zhan, Xiaohang and Han, Xiaoguang},
  booktitle={CVPR},
  year={2023}
}
```
