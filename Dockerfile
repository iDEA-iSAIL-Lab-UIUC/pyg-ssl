# Use an pytorch 2.3.1 cuda 12.1 python 3.10.14 image as a parent image
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Install torch_geometric (https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
RUN pip install torch_geometric
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
# Successfully installed aiohappyeyeballs-2.4.4 aiohttp-3.11.11 aiosignal-1.3.2 async-timeout-5.0.1 frozenlist-1.5.0 multidict-6.1.0 propcache-0.2.1 pyparsing-3.2.0 torch_geometric-2.6.1 yarl-1.18.3
# Successfully installed pyg_lib-0.4.0+pt23cu121 scipy-1.14.1 torch_cluster-1.6.3+pt23cu121 torch_scatter-2.1.2+pt23cu121 torch_sparse-0.6.18+pt23cu121 torch_spline_conv-1.2.2+pt23cu121

# Install faiss
RUN pip install faiss-gpu
# Successfully installed faiss-gpu-1.7.2

# Install deep graph library (https://www.dgl.ai/pages/start.html)
RUN pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
# Successfully installed annotated-types-0.7.0 dgl-2.4.0+cu121 pandas-2.2.3 pydantic-2.10.4 pydantic-core-2.27.2 python-dateutil-2.9.0.post0 typing-extensions-4.12.1 tzdata-2024.2

# other common libraries
RUN pip install scikit-learn matplotlib seaborn yacs