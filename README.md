# GeomDeepONet
A novel DeepONet architecture that is specifically designed for generating predictions on different 3D geometries discretized by different number of mesh nodes.

The data supporting this study can be downloaded in https://uofi.box.com/s/4jf59rw8be3ecuqnkrwp22qzowashbhq

All DeepONet codes were implemented using a TF2 backend, which needs to be set as an environment variable in DeepXDE before the code can be run properly:
export DDE_BACKEND=tensorflow

If you find our work helpful to your research, please cite our work as follows:
@article{he2024geomdeeponet,
  title={Geom-DeepONet: A point-cloud-based deep operator network for field predictions on 3D parameterized geometries},
  author={He, Junyan and Koric, Seid and Abueidda, Diab and Najafi, Ali and Jasiuk, Iwona},
  journal={Computer Methods in Applied Mechanics and Engineering},
  doi={https://doi.org/10.1016/j.cma.2024.117130},
  year={2024},
  publisher={Elsevier}
}
