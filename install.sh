#mamba create -n word2vec -c rapidsai -c conda-forge -c nvidia  rapids=23.10 python=3.9 cuda-version=11.8
#mamba activate word2vec
mamba install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia -y
mamba install -y -c bioconda -c conda-forge snakemake -y
mamba install -c conda-forge graph-tool scikit-learn numpy numba scipy pandas polars networkx seaborn matplotlib gensim ipykernel tqdm black faiss-gpu==1.7.3 -y
install GPUtil nvitop
