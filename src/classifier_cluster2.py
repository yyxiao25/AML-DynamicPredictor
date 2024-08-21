import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
import shap
import os

# Load the dataset
data = pd.read_excel('C:/Users/xyy/Desktop/nodata4ASTALT/nodata2_2cluster_simulated_training_patient_set_700.xlsx')  #data3_simulated_training_patient_set_1000.xlsx modify(nodata3)merged_simulated_training_patient_set_800_2.xlsx

# Define X and y
X = data.drop(['id', 'bsum_cluster_label'], axis=1)  # Features
y = data['bsum_cluster_label']  # Labels

# Encode the categorical labels into numbers
y_encoded = pd.factorize(y)[0]

# Feature importance evaluation with Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y_encoded)
rf_feature_importances = rf_model.feature_importances_

# Feature importance evaluation with Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X, y_encoded)
gb_feature_importances = gb_model.feature_importances_

# Combine the importances
features = X.columns
importances_df = pd.DataFrame({
    'Feature': features,
    'Importance': (rf_feature_importances + gb_feature_importances) / 2
    # 'Importance': rf_feature_importances
}).sort_values(by='Importance', ascending=False)

# Select the top n most important features
n = 20 # for example, select top 20 features
selected_features = importances_df.head(n)['Feature'].tolist()
print(selected_features)

test_df = pd.read_excel('C:/Users/xyy/Desktop/nodata4ASTALT/nodata2_2cluster_testing_set.xlsx')  # data3_filtered_testing_set.xlsx  modify(nodata3)patient_testing_set_2.xlsx
test_label = test_df['bsum_cluster_label']
external_validation_df = pd.read_excel('C:/Users/xyy/Desktop/nodata4ASTALT/data2_2cluster_data.xlsx')  #data4_nofilter_set.xlsx  external_validation_set.xlsx
external_validation_label = external_validation_df['bsum_cluster_label']

# 假设你的类别标签列名为 'bsum_cluster_label'
label_column = 'bsum_cluster_label'
selected_features_with_label = selected_features + [label_column]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
test_label_encoded=le.transform(test_label)
external_validation_encoded = le.transform(external_validation_label)

# Split the dataset into training and testing sets using only the selected features
X_selected = X[selected_features]
test_selected=test_df[selected_features]
external_validation_selected = external_validation_df[selected_features]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
test_scaled = scaler.transform(test_selected)
external_validation_scaled = scaler.transform(external_validation_selected)

# # 搜索参数组合，找到在训练集、测试集、外部验证集上auc值综合表现最优的参数
# # 准备测试的核函数列表和C值列表
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# C_values = [0.01,0.03,0.05,0.07,0.1, 0.2,0.3, 0.4,0.5,0.6, 0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
#
# # 用于存储符合条件的参数组合和准确率
# valid_parameters = []
#
# # 遍历每个核函数和C值的组合
# for kernel in kernels:
#     for C_val in C_values:
#         svm_model = SVC(kernel=kernel, C=C_val, class_weight='balanced', decision_function_shape='ovr', break_ties=True,
#                         gamma='auto', probability=True)
#         svm_model.fit(X_scaled, y_encoded)
#         train_pred = svm_model.predict(X_scaled)
#         accuracy_train = accuracy_score(y_encoded, train_pred)
#         test_pred = svm_model.predict(test_scaled)
#         accuracy_test = accuracy_score(test_label_encoded, test_pred)
#         external_validation_pred = svm_model.predict(external_validation_scaled)
#         accuracy_ex = accuracy_score(external_validation_encoded, external_validation_pred)
#         y_score = svm_model.predict_proba(X_scaled)[:, 1]
#         fpr, tpr, _ = roc_curve(y_encoded, y_score)
#         roc_auc = auc(fpr, tpr)
#         test_score = svm_model.predict_proba(test_scaled)[:, 1]
#         test_fpr, test_tpr, _ = roc_curve(test_label_encoded, test_score)
#         test_roc_auc = auc(test_fpr, test_tpr)
#         external_validation_score = svm_model.predict_proba(external_validation_scaled)[:, 1]
#         ex_fpr, ex_tpr, _ = roc_curve(external_validation_encoded, external_validation_score)
#         ex_roc_auc = auc(ex_fpr, ex_tpr)
#
#         # 打印结果
#         print(f"Kernel: {kernel}, C: {C_val}, Accuracy Train: {accuracy_train}, Accuracy In: {accuracy_test}, Accuracy Ex: {accuracy_ex}")
#         print(
#             f"Kernel: {kernel}, C: {C_val}, Auc Train: {roc_auc}, Auc In: {test_roc_auc}, Auc Ex: {ex_roc_auc}")
#
#         # 检查准确率是否符合目标范围
#         if 0.7 <= roc_auc and 0.70 <= test_roc_auc and 0.70<= ex_roc_auc:
#             valid_parameters.append({
#                 'kernel': kernel,
#                 'C': C_val,
#                 'Auc Train': roc_auc,
#                 'Auc In': test_roc_auc,
#                 'Auc Ex': ex_roc_auc
#             })
#
#         # # 检查准确率是否符合目标范围
#         # if 0.75 <= accuracy_train <= 0.95 and 0.75 <= accuracy_test and 0.75 <= accuracy_ex:
#         #     valid_parameters.append({
#         #         'kernel': kernel,
#         #         'C': C_val,
#         #         'accuracy_train': accuracy_train,
#         #         'accuracy_test': accuracy_test,
#         #         'accuracy_ex': accuracy_ex
#         #     })
#
# if valid_parameters:
#     # best_parameters = min(valid_parameters, key=lambda x: abs(x['accuracy_train']-0.85) + abs(x['accuracy_test']-0.75) + abs(x['accuracy_ex']-0.65))
#     print("找到符合条件的最优参数组合:", valid_parameters)
# else:
#     print("未找到符合条件的参数组合。")
# #


# # 通过上一步的搜索确定了kernel和C_value，接下来画结果图
# 计算每个类别的样本数量
class_counts = np.bincount(y_encoded)
# 计算类别权重（与样本数量成反比）
total_samples = len(y_encoded)
n_classes = len(le.classes_)
class_weights_balanced = {i: total_samples / (class_counts[i] * n_classes) for i in range(n_classes)}

# # 处理样本不平衡问题  手动设置更高的权重
Mild_index = list(le.classes_).index('Mild')   # Mild
Severe_index = list(le.classes_).index('Severe')
class_weights_balanced[Mild_index] *=1.8
class_weights_balanced[Severe_index] *= 1

# 确保所有权重按照LabelEncoder的顺序
class_weights = {le.transform([le.classes_[i]])[0]: weight for i, weight in class_weights_balanced.items()}

# Training and evaluating SVM
kernel = 'poly'   #'linear', 'poly', 'sigmoid', 'rbf'
svm_model = SVC(kernel=kernel, C=10, class_weight=class_weights, decision_function_shape='ovr', break_ties=True,
                gamma='auto', probability=True)  #,gamma='auto', class_weight='balanced',break_ties=True,
svm_model.fit(X_scaled, y_encoded)
# # 输出准确率、精确率、召回率等性能指标
y_pred = svm_model.predict(X_scaled)
cm_train = confusion_matrix(y_encoded, y_pred)
print("训练集：", classification_report(y_encoded, y_pred, target_names=le.classes_))

plt.rcParams['font.size'] = 15  # 设置colorbar字体大小
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["pdf.fonttype"] = 42
# 绘制混淆矩阵
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Training cohort', fontweight='bold')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.show()

test_pred = svm_model.predict(test_scaled)
cm_test = confusion_matrix(test_label_encoded, test_pred)
print("测试集：", classification_report(test_label_encoded, test_pred, target_names=le.classes_))
# 绘制混淆矩阵
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Internal validation cohort', fontweight='bold')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.show()

external_validation_pred = svm_model.predict(external_validation_scaled)
cm_ex = confusion_matrix(external_validation_encoded, external_validation_pred)
print("外部验证：",classification_report(external_validation_encoded, external_validation_pred, target_names=le.classes_))
# 绘制混淆矩阵
sns.heatmap(cm_ex, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('External validation cohort', fontweight='bold')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.show()

linewidth = 3

# 获取分类概率
y_score = svm_model.predict_proba(X_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_encoded, y_score)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc, lw=5)
plt.plot([0, 1], [0, 1], 'k--', lw=5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.title('Training cohort', fontweight='bold')
# plt.title('Multi-class ROC curve')
plt.legend(loc="lower right")
plt.show()

# 测试集
test_score = svm_model.predict_proba(test_scaled)[:, 1]
test_fpr, test_tpr, _ = roc_curve(test_label_encoded, test_score)
test_roc_auc = auc(test_fpr, test_tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(test_fpr, test_tpr, label='AUC = %0.2f' % test_roc_auc, lw=5)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.title('Internal validation cohort', fontweight='bold')
plt.legend(loc="lower right")
plt.show()

# 外部验证集
external_validation_score = svm_model.predict_proba(external_validation_scaled)[:, 1]
ex_fpr, ex_tpr, _ = roc_curve(external_validation_encoded, external_validation_score)
ex_roc_auc = auc(ex_fpr, ex_tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(ex_fpr, ex_tpr, label='AUC = %0.2f' % ex_roc_auc, lw=5)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.title('External validation cohort', fontweight='bold')
plt.legend(loc="lower right")
plt.show()

# # # 使用SHAP值展示特征对分类结果的贡献，每个特征的重要程度
# # 设置线程数以避免内存泄漏
# os.environ["OMP_NUM_THREADS"] = "4"
#
# # 使用K-means聚类算法来找到数据中的 K 个代表性群集中心,来近似整个数据集的特征分布。
# X_summary = shap.kmeans(X_scaled, 100)
#
# # 创建SHAP解释器，使用转换后的数据   由于SVM是一个概率模型，我们使用predict_proba方法而不是predict。
# explainer = shap.KernelExplainer(svm_model.predict_proba, X_summary.data, link="logit")
#
# # 计算Shapley值 shap_values函数计算每个特征对每个预测结果的贡献度。
# shap_values = explainer.shap_values(X_summary.data)
# # 展示整个数据集中每个特征的平均影响力
# shap.summary_plot(shap_values, X_selected, feature_names=selected_features)

















