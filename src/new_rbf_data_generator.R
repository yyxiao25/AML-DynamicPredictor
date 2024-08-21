library(caret)
library(semiArtificial)
library(readxl)
library(dplyr)
library(openxlsx)

# 加载数据
# data <- read.csv("C:/Users/xyy/Desktop/filtered_patient_data.csv")
# data <- read.csv("C:/Users/xyy/Desktop/nodata1_2cluster_data.csv")
# data <- read_excel("C:/Users/xyy/Desktop/AML_allnew_data_newALT/nopart_data3_2cluster_data.xlsx")
data <- read_excel("C:/Users/xyy/Desktop/nodata4ASTALT/nodata2_2cluster_data.xlsx")
# data <- read.csv("C:/Users/xyy/Desktop/data3_filtered_set.csv")

# 使用createDataPartition进行分层抽样，确保训练集和测试集在分类标签上的分布相似
set.seed(123) # 确保可重复性
trainIndex <- createDataPartition(data$bsum_cluster_label, p = .7, list = FALSE)
original_training_set <- data[trainIndex, ]
testing_set <- data[-trainIndex, ]

# # 保存测试集
write.xlsx(testing_set, "C:/Users/xyy/Desktop/nodata4ASTALT/nodata2_2cluster_testing_set.xlsx")
write.xlsx(original_training_set, "C:/Users/xyy/Desktop/nodata4ASTALT/nodata2_2cluster_training_set.xlsx")

# 删除指定的列
training_set <- select(original_training_set, -c(id, bsum_cluster, HGB, WBC, NEUT, PLT,
                                                 B_WBC,	gamma_WBC,	ktr_WBC,	slopeA_WBC,	slopeD_WBC,	B_HGB,	gamma_HGB,	ktr_HGB,	slopeA_HGB,
                                                 slopeD_HGB,	B_NEUT,	gamma_NEUT,	ktr_NEUT,	slopeA_NEUT,	slopeD_NEUT,	B_PLT,	gamma_PLT,	ktr_PLT,
                                                 slopeA_PLT,	slopeD_PLT))
# 'Pre_treatment_Mean_WBC', 'Pre_treatment_Mean_HGB', 'Pre_treatment_Mean_PLT', 'Pre_treatment_Mean_NEUT', 

# # 查看修改后的数据框
# head(training_set)

discrete_columns <- c('DD','AST/ALT','ALP_categorized','ALT_categorized','AST_categorized','sex','HBsAg','HBeAg','Anti-HCV', 'HIV-Ab', 'Syphilis','BG', 'ABOZDX', 'ABOFDX', 'Rh', 'BGZGTSC','bsum_cluster_label')

# 将指定列转换为因子类型
training_set <- training_set %>% 
  mutate(across(all_of(discrete_columns), as.factor))

for(size in seq(500, 1000, by = 100)) {
  # 创建RBF生成器
  dataRBF <- rbfDataGen(bsum_cluster_label ~ ., training_set, nominal = "encodeBinary")
  
  # 使用生成器创建新数据
  dataNewRBF <- newdata(dataRBF, size = size)
  
  # 为模拟数据生成ID
  dataNewRBF$id <- paste0("sim", seq_len(nrow(dataNewRBF)))
  
  # 将id列移动到第一列
  dataNewRBF <- dataNewRBF[c("id", setdiff(names(dataNewRBF), "id"))]
  head(original_training_set)
  merge_original_training_set <- select(original_training_set, -c(bsum_cluster, HGB, WBC, NEUT, PLT,
                                                            B_WBC,	gamma_WBC,	ktr_WBC,	slopeA_WBC,	slopeD_WBC,	B_HGB,	gamma_HGB,	ktr_HGB,	slopeA_HGB,
                                                            slopeD_HGB,	B_NEUT,	gamma_NEUT,	ktr_NEUT,	slopeA_NEUT,	slopeD_NEUT,	B_PLT,	gamma_PLT,	ktr_PLT,
                                                            slopeA_PLT,	slopeD_PLT))
  
  # 合并训练集和模拟数据
  merged_data <- rbind(merge_original_training_set, dataNewRBF)
  
  # 保存合并后的数据集
  file_name <- paste0("C:/Users/xyy/Desktop/nodata4ASTALT/nodata2_2cluster_simulated_training_patient_set_", size, ".xlsx")  
  write.xlsx(merged_data, file_name)
}



# # 创建RBF生成器
# dataRBF <- rbfDataGen(bsum_cluster_label ~ ., training_set, nominal = "encodeBinary")
# 
# # 使用生成器创建新数据
# dataNewRBF <- newdata(dataRBF, size = 1000)
# 
# # 为模拟数据生成ID
# dataNewRBF$id <- paste0("sim", seq_len(nrow(dataNewRBF)))
# 
# # 将id列移动到第一列
# dataNewRBF <- dataNewRBF[c("id", setdiff(names(dataNewRBF), "id"))]
# head(dataNewRBF)
# 
# original_training_set <- select(original_training_set, -c(bsum_cluster, HGB, WBC, NEUT, PLT,
#                                                           B_WBC,	gamma_WBC,	ktr_WBC,	slopeA_WBC,	slopeD_WBC,	B_HGB,	gamma_HGB,	ktr_HGB,	slopeA_HGB,
#                                                           slopeD_HGB,	B_NEUT,	gamma_NEUT,	ktr_NEUT,	slopeA_NEUT,	slopeD_NEUT,	B_PLT,	gamma_PLT,	ktr_PLT,
#                                                           slopeA_PLT,	slopeD_PLT))
# head(original_training_set)
# # 合并训练集和模拟数据
# merged_data <- rbind(original_training_set, dataNewRBF)
# 
# # 保存合并后的数据集
# write.xlsx(merged_data, "C:/Users/xyy/Desktop/data3_simulated_training_patient_set_1000.xlsx")

