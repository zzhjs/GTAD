rm(list=ls())
options(stringsAsFactors = F)
library(Seurat)
library(DropletUtils)

#scRNA_data.csv:gene×cell
sc.count <- read.csv("./data/scRNA_data.csv", header=T, row.names = 1,check.names=FALSE)
metadata <- read.csv("./data/scRNA_meta.csv", row.names=1, header=T,check.names=FALSE)
#ST_data.csv:gene×spot
st.count <- read.csv("./data/ST_data.csv", header=T, row.names = 1,check.names=FALSE)

# metadata <- metadata[metadata$cell_subclass != "", ]
metadata[,'cell_subclass'][metadata[,'cell_subclass']==""]="NAN"
intersect.genes <- intersect(rownames(sc.count),rownames(st.count))
sc.count <- sc.count[intersect.genes,]
st.count <- st.count[intersect.genes,]
count.list <- list(sc.count,st.count)
sc.count<-sc.count[,rownames(metadata)]
metadata<-metadata[colnames(sc.count),]

sel.features <- read.table('list.txt', header = FALSE)[, 1]
sel.features<-intersect(rownames(st.count),sel.features)
st_count_new <- list(sc.count[sel.features,],st.count[sel.features,])
lable<-matrix(data = metadata[,'cell_subclass'])
colnames(lable) <- 'cell_subclass'
rownames(lable) <- rownames(metadata)
st_label=list(lable)
tem.t1 <- Seurat::CreateSeuratObject(counts = st_count_new[[1]],meta.data=metadata);
Seurat::Idents(object = tem.t1) <- tem.t1@meta.data$cell_subclass
test_spot_fun = function (se_obj, clust_vr, n = 1000, verbose = TRUE){
  if (is(se_obj) != "Seurat") 
    stop("ERROR: se_obj must be a Seurat object!")
  if (!is.character(clust_vr)) 
    stop("ERROR: clust_vr must be a character string!")
  if (!is.numeric(n)) 
    stop("ERROR: n must be an integer!")
  if (!is.logical(verbose)) 
    stop("ERROR: verbose must be a logical object!")
  suppressMessages(require(DropletUtils))
  suppressMessages(require(purrr))
  suppressMessages(require(dplyr))
  suppressMessages(require(tidyr))
  se_obj@meta.data[, clust_vr] <- gsub(pattern = "[[:punct:]]|[[:blank:]]", 
                                       ".", x = se_obj@meta.data[, clust_vr], perl = TRUE)
  print("Generating synthetic test spots...")
  start_gen <- Sys.time()
  pb <- txtProgressBar(min = 0, max = n, style = 3)
  count_mtrx <- as.matrix(se_obj@assays$RNA@counts)
  ds_spots <- lapply(seq_len(n), function(i) {
    cell_pool <- sample(colnames(count_mtrx), sample(x = 2:10, 
                                                     size = 1))
    pos <- which(colnames(count_mtrx) %in% cell_pool)
    tmp_ds <- se_obj@meta.data[pos, ] %>% mutate(weight = 1)
    name_simp <- paste("spot_", i, sep = "")
    spot_ds <- tmp_ds %>% dplyr::select(all_of(clust_vr), 
                                        weight) %>% dplyr::group_by(!!sym(clust_vr)) %>% 
      dplyr::summarise(sum_weights = sum(weight)) %>% dplyr::ungroup() %>% 
      tidyr::pivot_wider(names_from = all_of(clust_vr), 
                         values_from = sum_weights) %>% dplyr::mutate(name = name_simp)
    syn_spot <- rowSums(as.matrix(count_mtrx[, cell_pool]))
    sum(syn_spot)
    names_genes <- names(syn_spot)
    if (sum(syn_spot) > 25000) {
      syn_spot_sparse <- DropletUtils::downsampleMatrix(Matrix::Matrix(syn_spot, 
                                                                       sparse = T), prop = 20000/sum(syn_spot))
    }
    else {
      syn_spot_sparse <- Matrix::Matrix(syn_spot, sparse = T)
    }
    rownames(syn_spot_sparse) <- names_genes
    colnames(syn_spot_sparse) <- name_simp
    setTxtProgressBar(pb, i)
    return(list(syn_spot_sparse, spot_ds))
  })
  ds_syn_spots <- purrr::map(ds_spots, 1) %>% base::Reduce(function(m1, 
                                                                    m2) cbind(unlist(m1), unlist(m2)), .)
  ds_spots_metadata <- purrr::map(ds_spots, 2) %>% dplyr::bind_rows() %>% 
    data.frame()
  ds_spots_metadata[is.na(ds_spots_metadata)] <- 0
  lev_mod <- gsub("[\\+|\\ |\\/]", ".", unique(se_obj@meta.data[, 
                                                                clust_vr]))
  colnames(ds_spots_metadata) <- gsub("[\\|\\ |\\/]", ".", 
                                      colnames(ds_spots_metadata))
  if (sum(lev_mod %in% colnames(ds_spots_metadata)) == (length(unique(se_obj@meta.data[, 
                                                                                       clust_vr])) + 1)) {
    ds_spots_metadata <- ds_spots_metadata[, lev_mod]
  }
  else {
    missing_cols <- lev_mod[which(!lev_mod %in% colnames(ds_spots_metadata))]
    ds_spots_metadata[missing_cols] <- 0
    ds_spots_metadata <- ds_spots_metadata[, lev_mod]
  }
  close(pb)
  print(sprintf("Generation of %s test spots took %s mins", 
                n, round(difftime(Sys.time(), start_gen, units = "mins"), 
                         2)))
  print("intersect")
  return(list(topic_profiles = ds_syn_spots, cell_composition = ds_spots_metadata))
}
test.spot.ls1<-test_spot_fun(se_obj=tem.t1,clust_vr='cell_subclass',n=10000)

test.spot.counts1 <- as.matrix(test.spot.ls1[[1]])
colnames(test.spot.counts1)<-paste("mixt",1:ncol(test.spot.counts1),sep="_");
metadata1 <- test.spot.ls1[[2]]
test.spot.metadata1 <- do.call(rbind,lapply(1:nrow(metadata1),function(i){metadata1[i,]/sum(metadata1[i,])}))
st_counts <- list(test.spot.counts1,st_count_new[[2]])

st_label[[1]] <- test.spot.metadata1
N1 <- ncol(st_counts[[1]]); N2 <- ncol(st_counts[[2]])
label.list2 <- do.call("rbind", rep(list(st_label[[1]]), round(N2/N1)+1))[1:N2,]
st_labels <- list(st_label[[1]],label.list2)
#' normalize function 规范化函数
normalize_data <- function(count.list){
  "//访问了名为list的向量"
  norm.list <- vector('list')
  var.features <- vector('list')
  for ( i in 1:length(count.list)){
    "//归一化数据"
    norm.list[[i]] <- as.matrix(Seurat:::NormalizeData.default(count.list[[i]],verbose=FALSE))
    "//发现高可变基因"
    hvf.info <- Seurat:::FindVariableFeatures.default(count.list[[i]],selection.method='vst',verbose=FALSE)
    hvf.info <- hvf.info[which(x = hvf.info[, 1, drop = TRUE] != 0), ]
    hvf.info <- hvf.info[order(hvf.info$vst.variance.standardized, decreasing = TRUE), , drop = FALSE]
    "//获取前n个高可变基因"
    var.features[[i]] <- head(rownames(hvf.info), n = 2000)
  }
  sel.features <- selectIntegrationFeature(count.list,var.features)
  return (list(norm.list,sel.features))}

#' scaling function  标度函数
scale_data <- function(count.list,norm.list,hvg.features){
  scale.list <- lapply(norm.list,function(mat){
    "//scale标准化"
    Seurat:::ScaleData.default(object = mat, features = hvg.features,verbose=FALSE)})
  scale.list <- lapply(1:length(count.list),function(i){
    return (scale.list[[i]][na.omit(match(rownames(count.list[[i]]),rownames(scale.list[[i]]))),])})
  return (scale.list)}

#' select HVG genes 选择高表达基因
selectIntegrationFeature <- function(count.list,var.features,nfeatures = 2000){
  "//unlist 列表转换为向量 unname 从对象中删除名称"
  var.features1 <- unname(unlist(var.features))
  "//table 以表格的形式创建具有变量名称和频率的数据的分类表示 sort 默认对向量按照从小到大排序"
  var.features2 <- sort(table(var.features1), decreasing = TRUE)
  for (i in 1:length(count.list)) {
    " //names 获取对象名称 %in% 判断前一个对象是否在后一个对象中"
    var.features3 <- var.features2[names(var.features2) %in% rownames(count.list[[i]])]}    
  tie.val <- var.features3[min(nfeatures, length(var.features3))]
  features <- names(var.features3[which(var.features3 > tie.val)])
  if (length(features) > 0) {
    feature.ranks <- sapply(features, function(x) {
      ranks <- sapply(var.features, function(y) {
        if (x %in% y) {
          return(which(x == y))
        }
        return(NULL)
      })
      median(unlist(ranks))
    })
    features <- names(sort(feature.ranks))
  }
  features.tie <- var.features3[which(var.features3 == tie.val)]
  tie.ranks <- sapply(names(features.tie), function(x) {
    ranks <- sapply(var.features, function(y) {
      if (x %in% y) {return(which(x == y))}
      return(NULL)
    })
    median(unlist(ranks))
  })
  features <- c(features, names(head(sort(tie.ranks), nfeatures - length(features))))
  return(features)
}

#' select variable genes 选择可变基因
select_feature <- function(data,label,nf=2000){
  M <- nrow(data); new.label <- label[,1]
  pv1 <- sapply(1:M, function(i){
    "//获取表格"
    mydataframe <- data.frame(y=as.numeric(data[i,]), ig=new.label)
    " //方差分析"
    fit <- aov(y ~ ig, data=mydataframe)
    " //生成各种模型拟合函数的结果汇总"
    summary(fit)[[1]][["Pr(>F)"]][1]})
  names(pv1) <- rownames(data)
  pv1.sig <- names(pv1)[order(pv1)[1:nf]]
  "//返回一个把重复元素或行给删除的向量、数据框或数组"
  egen <- unique(pv1.sig)
  return (egen)
}

res1 <- normalize_data(st_counts)
st_norm <- res1[[1]]; variable_gene <- res1[[2]];
st_scale <- scale_data(st_counts,st_norm,variable_gene)
st.scale<-st_scale
variable.genes<-variable_gene
step1<-list(st_counts,st_labels,st_norm,st_scale,variable_gene)
st.count <- step1[[1]];
st.label <- step1[[2]];
st.norm <- step1[[3]];
st.scale <- step1[[4]];
variable.genes <- step1[[5]]

#' create data folders
dir.create('Datadir'); dir.create('Output');
inforDir <- 'Infor_Data'; dir.create(inforDir)
#' save counts data to certain path: 'Datadir'
write.csv(t(st.count[[1]]),file='Datadir/Pseudo_ST1.csv',quote=F,row.names=T)
write.csv(t(st.count[[2]]),file='Datadir/Real_ST2.csv',quote=F,row.names=T)
if (!dir.exists(paste0(inforDir,'/ST_count'))){dir.create(paste0(inforDir,'/ST_count'))}
if (!dir.exists(paste0(inforDir,'/ST_label'))){dir.create(paste0(inforDir,'/ST_label'))}
if (!dir.exists(paste0(inforDir,'/ST_norm'))){dir.create(paste0(inforDir,'/ST_norm'))}
if (!dir.exists(paste0(inforDir,'/ST_scale'))){dir.create(paste0(inforDir,'/ST_scale'))}

for (i in 1:2){
  write.csv(st.count[[i]],file=paste0(inforDir,'/ST_count/ST_count_',i,'.csv'),quote=F)
  write.csv(st.label[[i]],file=paste0(inforDir,'/ST_label/ST_label_',i,'.csv'),quote=F)
  write.csv(st.norm[[i]],file=paste0(inforDir,'/ST_norm/ST_norm_',i,'.csv'),quote=F)
  write.csv(st.scale[[i]],file=paste0(inforDir,'/ST_scale/ST_scale_',i,'.csv'),quote=F)
}
c1<- Seurat::CreateSeuratObject(counts = st_counts[[1]])
c2<- Seurat::CreateSeuratObject(counts = st_counts[[2]])
c1<-NormalizeData(c1)
c2<-NormalizeData(c2)
pbmc.obj <- FindIntegrationAnchors(object.list = list(c1,c2))
pbmc.obj <- IntegrateData(anchorset = pbmc.obj)
integrated.mat <- GetAssayData(object = pbmc.obj, slot = "data", assay = "integrated")
write.csv(integrated.mat,file=paste0(inforDir,'/integrated.mat.csv'),quote=F)
