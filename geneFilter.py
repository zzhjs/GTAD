import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

#scRNA_data.csv:gene×cell
adata_cortex = sc.read_csv('./data/scRNA_data.csv').T
adata_cortex_meta = pd.read_csv('./data/scRNA_meta.csv', index_col=0)
adata_cortex_meta_ = adata_cortex_meta.loc[adata_cortex.obs.index,]
adata_cortex.obs = adata_cortex_meta_
adata_cortex.var_names_make_unique()  
#Preprocessing
adata_cortex.var['mt'] = adata_cortex.var_names.str.startswith('Mt-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata_cortex, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pp.normalize_total(adata_cortex)
#PCA and clustering : Known markers with 'cell_subclass'
sc.tl.pca(adata_cortex, svd_solver='arpack')
sc.pp.neighbors(adata_cortex, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata_cortex)
sc.tl.leiden(adata_cortex, resolution = 0.5)
sc.pl.umap(adata_cortex, color=['leiden','cell_subclass'])
sc.tl.rank_genes_groups(adata_cortex, 'cell_subclass', method='wilcoxon')
sc.pl.rank_genes_groups(adata_cortex, n_genes=30, sharey=False)
genelists=adata_cortex.uns['rank_genes_groups']['names']
df_genelists = pd.DataFrame.from_records(genelists)
num_markers=30
res_genes = []
for column in df_genelists.head(num_markers): 
    res_genes.extend(df_genelists.head(num_markers)[column].tolist())
res_genes_ = list(set(res_genes))
with open('list.txt', 'w') as f:
    # 将列表中的每个元素转换为字符串，并逐行写入文件
    for item in res_genes_:
        f.write(str(item) + '\n')
