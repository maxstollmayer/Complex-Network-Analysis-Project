# Persistent Homology applied to Complex Network Analysis
by Maximilian Stollmayer & Manuel Wissiak

This is the home of our project in the seminar course _Complex Network Analysis_ by Jörg Menche and Felix Müller at the University of Vienna.

The repository holds folders named `presentation1`, `presentation2` and `video`, which are all made for the presentations done in the corresponding course. In the corresponding `presentationX.ipynb` Jupyter notebooks some of the work of the project is summarized, while the folder `video` holds the final product of the project.

The folder `VR` contains 3 prepared networks for the __VRNetzer__ platform at [Menche Lab](https://menchelab.com/). Each of those folders consists of the `.csv` files of the nodes and edges parsed correctly for the VRNetzer and diagrams of the analysis of its ring structures. And furthermore each contains a jupyter notebook documenting how the network is build and how the `.csv` and `.png` files are created.
- `fibro`: a network of fibroblasts of a rheumatoid arthritis patient, where nodes are linked by spatial proximity of $<=115 \ \mu m$; see [Cadherin-11 Induces Rheumatoid Arthritis Fibroblast-Like Synoviocytes to Form Lining Layers in Vitro, Kiener et al](https://www.sciencedirect.com/science/article/pii/S0002944010621724#cesec185) for further discussions of the data
- `toy`: an artifical sequence of networks made for the conceptual presentation of persistent homology
- `yeast`: a network of protein-protein interactions in budding yeast, found at [Netzschleuder](https://networks.skewed.de/net/collins_yeast)

To easily export to the VRNetzer platform from our experimentation notebooks, we created the interface `layout.py` for exporting and `preview.py` for quick and interactive previewing of the network layout, which can be found in the `utils` folder. Furthermore since in its current form VRNetzer does not support the rendering of spheres, we developed a workaround to show the filtration process on this platform. `spheres.py`, also inside the `utils` folder, implements spherical mesh generation as `NetworkX` graphs which can then easily be centered around each node of a given network and layout.

Lastly the folder `experiments` contains a wide range of networks analyzed in the course of this project.
