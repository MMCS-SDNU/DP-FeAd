# DP-FeAd

PyTorch implementataion for Dual Prototypes-Based Personalized Federated Adversarial Cross-Modal Hashing

## Environment

`python 3.8+`

`pytorch 2.0+`

## Usage

`python main.py help` to get help information.

`python main.py train` for train and test DP-FeAd.

`python main.py test` for test only.

## References and Acknowledgements

$$
\begin{bibliography}{9}
\bibitem{kawar2022denoising}
Bahjat Kawar and Michael Elad and Stefano Ermon and Jiaming Song.
\newblock Denoising Diffusion Restoration Models.
\newblock In {\em Advances in Neural Information Processing Systems}, 2022.
\end{bibliography}
$$
@inproceedings{zhan2020supervised,
  title={Supervised hierarchical deep hashing for cross-modal retrieval},
  author={Zhan, Yu-Wei and Luo, Xin and Wang, Yongxin and Xu, Xin-Shun},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={3386--3394},
  year={2020} 
}

This implementation is based on / inspired by:

- https://github.com/SDU-MIMA/SHDCH
