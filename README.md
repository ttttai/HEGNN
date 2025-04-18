# HEGNN
We propose Heterogeneous Equivariant Graph Neural Networks (HEGNNs). This model is designed for heterogeneous graphs with edge associated features and node coordinates. HEGNNs are capable of capturing complex and diverse relationships in graphs while preserving the equivariance properties with respect to geometric transformations. By incorporating node and edge heterogeneity into the model design, HEGNNs can learn by considering the spatial relationships between different types of nodes. Moreover, the equivariant properties allow it to generalize effectively to graphs with similar spatial relationships in different regions.

<div align="center">
    <img src="./img/HEGCL.png" width=600px centering>
</div>

The color of the nodes and edges correspondes to the types of nodes and edges respectively. The color of the blocks corresponds to the embedding representation of each node. When applying HEGCL, the node embedding representations remain consistent even after transformations such as translation and rotation are applied. Futhermore, the node coordinates of the graph applied the coordinate transformation after applying HEGCL match those of the graph where HEGCL is applied first and the transformation is applied afterward.