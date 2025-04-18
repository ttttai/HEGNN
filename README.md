# HEGNN
We propose Heterogeneous Equivariant Graph Neural Networks (HEGNNs). This model is designed for heterogeneous graphs with edge associated features and node coordinates. HEGNNs are capable of capturing complex and diverse relationships in graphs while preserving the equivariance properties with respect to geometric transformations. By incorporating node and edge heterogeneity into the model design, HEGNNs can learn by considering the spatial relationships between different types of nodes. Moreover, the equivariant properties allow it to generalize effectively to graphs with similar spatial relationships in different regions.

We consider a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$. In addition to the node embeddings $\bm{h}_i \in \mathbb{R}^d$, each node also has n-dimensional coordinates $\bm{x}_i \in \mathbb{R}^n$. 
For translation transformations $\bm{g} \in \mathbb{R}^n$ and rotation transformations represented by an orthogonal matrix $\bm{Q} \in \mathbb{R}^{n \times n}$, HEGCL(Heterogeneous Equivariant Graph Convolutional Layer) satisfies the following equation:

```math
\bm{Q}\bm{x}^{l+1} + \bm{g}, \bm{h}^{l+1} = \mathrm{HEGCL}(\bm{Q}\bm{x}^l + \bm{g}, \bm{h}^l)
```

<div align="center">
    <img src="./img/HEGCL.png" width=600px centering>
</div>

The color of the nodes and edges correspondes to the types of nodes and edges respectively. The color of the blocks corresponds to the embedding representation of each node. When applying HEGCL, the node embedding representations remain consistent even after transformations such as translation and rotation are applied. Futhermore, the node coordinates of the graph applied the coordinate transformation after applying HEGCL match those of the graph where HEGCL is applied first and the transformation is applied afterward.