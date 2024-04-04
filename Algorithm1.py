from multiprocessing import freeze_support
if __name__ == "__main__":
    freeze_support()
    
    import pandas as pd
    import numpy as np
    from rdkit import Chem
    
    from rdkit.Chem import Descriptors, Draw
    from mordred import Calculator, descriptors
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from genetic_selection import GeneticSelectionCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SequentialFeatureSelector as sfs
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE


    #Выделяем 20 тысяч строк из датасета
    df = pd.read_csv(r"\qm9.csv")
    random = df.sample(n = 1000, random_state = 9)
    random.to_csv(r"\dae_samples.csv", index = True)


    df_sample = pd.read_csv(r"\dae_samples.csv")

    
    list_of_smiles = df_sample['smiles']
    #Преобразование смайлз в молекулы
    mols = [Chem.MolFromSmiles(smiles) for smiles in list_of_smiles]

    img = Draw.MolToImage(mols[0])
    #img.show()

    #Создаем калькулятор для вычисления хим дескрипторов
    calc = Calculator(descriptors)

    #Рассчитываем дескрипторы 
    mordred_desc = list(calc.map(mols))
    df_mord = pd.DataFrame(mordred_desc)

    df_combined_mordred = pd.concat([df_sample, df_mord], axis = 1)

    #Вычисляем, сколько дескприторов рассчитал нам Mordred для каждой молекулы
    num_of_desc_mord = len(df_mord)
    print(num_of_desc_mord)

    descriptor_values = []

    for smiles in df_combined_mordred['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        
        descriptors = []
        computed_descriptors = Descriptors.descList
        
        for descriptor in computed_descriptors:
            value = descriptor[1](mol)
            descriptors.append(value)
        
        descriptor_values.append(descriptors)

    # Добавление дескрипторов RDKit к df_combined_mordred
    df_descriptors = pd.DataFrame(descriptor_values, columns=[descriptor[0] for descriptor in computed_descriptors])

    # Объединяем df_combined_mordred и df_descriptors
    df_combined = pd.concat([df_combined_mordred, df_descriptors], axis=1)

    # Получился датафрейм с 2053 дескрпиторами (1844 - mordred, 210 - rdkit)
    
    #Выводим основную информацию о датафрейме
    print(df_combined.info()) 

    #Удаляем не информативные столбцы
    df_combined = df_combined.drop(columns = ['Unnamed: 0', 'mol_id', 'smiles'])
    #Проверяем, сколько пропусков в каждом столбце
    df_combined.isna().sum()
    #Выяснилось, что пропусков нет, УРА! ПРОВЕРЯЕМ на дубликаты
    df_combined.duplicated().sum()

    #У нас есть столбцы, где нет никаких данных(с ошибками), их необходимо удалить
    deleted_cols = []
    for column in df_combined.columns:
        if any('invalid value' in str(x) or 'Unable to parse' in str(x) for x in df_combined[column]):
            deleted_cols.append(column) #Таким образом, мы нашли ячейки, в которых не произошел парсинг по названиям ошибок
            del df_combined[column]  #удалили эти ячейки, и чисто для себя занесли их в список, чтоб посмотреть
    print("Удаленные столбцы:", deleted_cols)

    '''Удаленные столбцы: [139, 140, 141, 142, 148, 149, 150, 151, 157, 158, 159, 160, 166, 167, 168, 169, 175, 176, 177, 178, 184, 185, 186, 187, 
    193, 194, 195, 196, 202, 203, 204, 205, 211, 212, 213, 214, 220, 221, 222, 223, 229, 230, 231, 232, 346, 347, 348, 349, 355, 356, 357, 358, 364, 
    365, 366, 367, 373, 374, 375, 376, 382, 383, 384, 385, 391, 392, 393, 394, 400, 401, 402, 403, 409, 410, 411, 412, 418, 419, 420, 421, 427, 428, 
    429, 430, 436, 437, 438, 439, 445, 446, 447, 448, 453, 454, 455, 456, 461, 462, 463, 464, 469, 470, 471, 472, 477, 478, 479, 480, 485, 486, 487, 
    488, 493, 494, 495, 496, 501, 502, 503, 504, 509, 510, 511, 512, 517, 518, 519, 520, 525, 526, 527, 528, 533, 534, 535, 536, 541, 542, 543, 544, 
    549, 550, 551, 552, 557, 558, 559, 560, 565, 566, 567, 568, 573, 574, 575, 576, 581, 582, 583, 584, 589, 590, 591, 592, 597, 598, 599, 600, 605, 
    606, 607, 608, 613, 614, 615, 616, 621, 622, 623, 624, 629, 630, 631, 632, 637, 638, 639, 640]'''

    #Посмотрим на наши данные с помощью гистограммы и виолиновых графиков
    colors = ['blue', 'green', 'red']
    data = df_combined['gap']
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Histogram for gap', 'Violin Plot for gap'])
    
    #Строим гистограмму
    hist_fig = px.histogram(df_combined, x="gap", nbins = 60,
                    color_discrete_sequence = colors,
                    opacity = 0.7)

    fig.add_trace(hist_fig['data'][0], row=1, col=1) 

    #Строим виолиновый график
    violin_fig = px.violin(df_combined, y="gap", color_discrete_sequence = colors, box = True)
    fig.add_trace(violin_fig['data'][0], row=1, col=2) 
    fig.update_layout(showlegend=False, title_text="Histogram and Violin Plot")
    fig.show()


    #Выясним выбросы используя Z-score метод
    def detect_outliers_zscore(data, threshold=3.0):
        mean = np.mean(data)
        std_dev = np.std(data)
        z_scores = [(x - mean) / std_dev for x in data]
        outliers = np.where(np.abs(z_scores) > threshold)[0]
        return outliers

    outliers_z = detect_outliers_zscore(df_combined['gap'])
    print("Indices of outliers in Z:", outliers_z)
    print("Outlier values in Z:", [df_combined['gap'][i] for i in outliers_z])

    '''Outlier values in Z: [0.0747, 0.0972, 0.4063, 0.1065, 0.042, 0.4051, 0.0925, 0.0623, 0.3957, 0.4802, 0.1035]'''

    #IQR метод
    def detect_outliers_iqr(data, threshold=1.5):
 
        quartile_1, quartile_3 = np.percentile(data, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (threshold * iqr)
        upper_bound = quartile_3 + (threshold * iqr)
        outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
        return outliers

    
    outliers_IQR = detect_outliers_iqr(df_combined['gap'])
    print("Indices of outliers in IQR:", outliers_IQR)
    print("Outlier values in IQR:", [df_combined['gap'][i] for i in outliers_IQR])

    '''Outlier values in IQR: [0.0747, 0.0972, 0.4063, 0.1065, 0.042, 0.4051, 0.0925, 0.0623, 0.3957, 0.4802, 0.1035]'''
    # Изобразим наши выбросы
    outliers = detect_outliers_iqr(df_combined['gap'])
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=np.arange(len(df_combined['gap'])), y=df_combined['gap'], mode='markers', name='Data Points', marker=dict(color=colors[2])))
    fig.add_trace(go.Scatter(x=outliers, y=[df_combined['gap'][i] for i in outliers], mode='markers', marker=dict(color=colors[0]), name='Outliers'))

    fig.update_layout(title='Outlier Detection Using IQR Method',
                    xaxis_title='Index',
                    yaxis_title='Value')

    fig.show()

    '''Для удаления выбросов из наших данных о растворимости выбираю метод удаления по межквартильному размаху (IQR). 
    Этот выбор обусловлен тем, что метод IQR более устойчив к выбросам и более подходит для данных, которые не обязательно следуют нормальному распределению. 
    Кроме того, этот метод не требует предположений о распределении данных и позволяет сохранить больше информации, поскольку он опирается на квартили.'''
    

    filtered_data = df_combined.drop(outliers_IQR)
    print(filtered_data[:15])

   
    #Для начала выясним, какие из столбцов имеют категориальные признаки
    categorical_columns = []
    numeric_columns = []

    for column in filtered_data.columns:
        if filtered_data[column].dtype == 'object':
            categorical_columns.append(column)
        else:
            numeric_columns.append(column)

    print("Категориальные:", categorical_columns)
    print("Числовые:", numeric_columns)
    
    '''Категориальные: ['mol_id', 'smiles', 0, 1, 780, 781, 782, 783, 784, 785, 786, 
    787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 
    803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 817, 818, 819, 820 . . .'''

        # Делим числовые и булевы типы данных

    am_bools = filtered_data.select_dtypes(np.bool_)
    am_num = filtered_data.select_dtypes(exclude = 'bool')
    filtered_data.columns = filtered_data.columns.astype(str)
    # Преобразование категориальных признаков с помощью one-hot encoding
    one_hot_encoded = pd.get_dummies(am_num['gap'], prefix='gap')

    '''Прямое кодирование (one-hot encoding) для преобразования категориальных признаков выбираю, 
    потому что оно создает бинарные переменные для каждой категории, что подходит для моделей мМО, не учитывающих порядок категорий.'''

    filtered_data = pd.concat([filtered_data, one_hot_encoded], axis=1)

    # Нормализация числовых признаков с помощью стандартизации (z-score normalization)
    scaler = StandardScaler()
    df_enc_zscore  = scaler.fit_transform(filtered_data)  # Замените на ваши числовые признаки
    df_enc_zscore  = pd.DataFrame(df_enc_zscore, columns=filtered_data.columns)

    # Результат
    print(filtered_data.head())
    filtered_data.to_csv(r"\filtered.csv", index = True)

   #Feature Selection
    filtered_data = pd.read_csv(r"\filtered.csv", low_memory=False)

   #Проверка типа данных дескрипторов
    print(list(np.unique(filtered_data.dtypes)))
    # Делим числовые и булевы типы данных
    am_bools = filtered_data.select_dtypes(np.bool_)
    am_num = filtered_data.select_dtypes(exclude = 'bool')
    print(am_bools.head())

    #Создаем box-plot для дескрипторов
    am_num = filtered_data.select_dtypes(include='number')

    # Создание ящиков с усами для первых 5 дескрипторов
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=am_num.iloc[:, :5])
    plt.title('Boxplot of Descriptors')
    plt.xlabel('Descriptor')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.show()
    
    #Виолиновые графики
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=am_num.iloc[:, :5], inner="quartile")
    plt.title('Violin Plot of Descriptors')
    plt.xlabel('Descriptor')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.show()

    #Считаем ковариацию
    cols = am_num.columns[:5]
    stdsc = StandardScaler()
    X_std = stdsc.fit_transform(am_num[cols].iloc[:, range(5)].values)
    cov_mat = np.cov(X_std.T)

    #Создаем тепловую карту ковариационной матрицы
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cov_mat,
                    cbar=True,
                    annot=True,
                    square=True,
                    fmt='.2f',
                    annot_kws={'size': 12},
                    cmap='coolwarm',
                    yticklabels=cols,
                    xticklabels=cols)
    plt.title('Covariance matrix', size = 18)
    plt.tight_layout()
    plt.show()

    #Профильтруем матрицу дескрипторов ковариации, задав пороговое значение
    FILTER_THRESHOLD = 0.9 #пороговое значение

    cov_mat_df = pd.DataFrame(cov_mat, columns=cols) #создаем колонки по матрице

    #Берем только верхний треугольник матрицы, содержащий уникальные пары без повторений
    upper_tri = cov_mat_df.where(
        np.triu(
            np.ones(cov_mat_df.shape), k=1).astype(np.bool_)
            )
    #Узнаем имена дескрипторов, где хотя бы одно значение в столбце превышает threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > FILTER_THRESHOLD)]
    # Удаляем эти столбцы, так как если коэф корр бизок к ед, значит дескприторы дают схожую информацию о молекуле
    df_after_FS = am_num.drop(to_drop, axis=1)
    print(upper_tri[:6])
        
    #WRAPPER METHOD
    #Генетический алгоритм выбора признаков
    X = am_num.drop(['gap'], axis=1)
    Y = am_num['gap'].astype(float)

    # Преобразование имен признаков в строковый тип данных
    X.columns = X.columns.astype(str)

    # estimator = RandomForestRegressor()

    # model = GeneticSelectionCV(
    #     estimator, cv=5, verbose=0,
    #     scoring="r2", max_features=5,
    #     n_population=5, crossover_proba=0.5,
    #     mutation_proba=0.2, n_generations=50,
    #     crossover_independent_proba=0.5,
    #     mutation_independent_proba=0.04,
    #     tournament_size=3, n_gen_no_change=20,
    #     caching=True, n_jobs=-1)
    
    # model = model.fit(X, y)
    # print('Features_model:', X.columns[model.support_])
    # plt.plot(model.generation_scores_, 'o', color = 'black')

    # #Forward Selection
    # sfs1 = sfs(estimator, n_features_to_select=5,
    #            scoring = 'r2',
    #            cv = 5)
    
    # sfs1 = sfs1.fit(X, y)
    # print('Features_sfs:', X.columns[sfs1.support_])

    # estimator_sfs = RandomForestRegressor()
    # estimator_sfs.fit(X.loc[:, sfs1.support_], y)

    # r2 = estimator_sfs.score(X.loc[:, sfs1.support_], y)
    # print('R-squared_sfs:', np.round(r2, decimals=4))


    #Линейная зависимость
    '''PCA выбран для уменьшения размерности данных и выявления наиболее информативных компонент 
    Он помогает сохранить максимум информации, сокращая количество признаков и выделяя основные направления в данных'''
    # Оставляю только отобранные признаки методом SFS
    X_norm = MinMaxScaler().fit_transform(X)
    X_rdkit = X.iloc[:, 1844:2053]
    X_rdkit_norm = X_norm[:, 1844:2053]

    def PCA_implementation(X, num_components):
        # Среднее значение каждой переменной
        X_meaned = X - np.mean(X, axis=0)

        # КОвариационная матрица
        cov_mat = np.cov(X_meaned , rowvar = False)

        # Cобственные значения и собственные векторы
        eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)

        # Сортировказначений по убыванию
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]
        # Выбираю подмножество из отсортированной матрицы собственных векторов
        eigenvector_subset = sorted_eigenvectors[:, 0:num_components]
        
        X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
        
        return X_reduced

    #  PCA
    pca = PCA()
    X_reduced = PCA_implementation(X, 2)
    fig = plt.figure(figsize=(14,10))
    ax  = fig.add_subplot(111)
    scatter = ax.scatter(X_reduced[:,0], -X_reduced[:,1], c=Y, s=45, edgecolors='green', cmap=cm.jet_r, alpha=0.5)
    colorbar = fig.colorbar(scatter, ax=ax, label = "E")
    plt.xlabel(r'$PC_1$')
    plt.ylabel(r'$PC_2$')
    plt.title('PCA')
    sns.despine()
    plt.show()

    pca = PCA()
    X_reduced = pca.fit_transform(X_norm)
    y = np.cumsum(pca.explained_variance_ratio_)
    # Считаем главные компоненты
    xi = np.arange(1, y.shape[0]+1, step=1)
    # Выводим результаты
    plt.ylim(0.5,1.1)
    plt.xlim(0.0,15.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 21, step=2))
    plt.ylabel('Cumulative variance (%)')
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 1, 'Explained variance: 95%', color = 'red', fontsize=12)
    plt.grid(axis='x')
    plt.show()


    pca = PCA(n_components=12)
    X_reduced = pca.fit_transform(X_norm)
    # Проверяю корреляции с исходными данными
    df_pc = pd.DataFrame(data = X_reduced, columns = [f'PC{i}' for i in range(1,13)])
    df_col = pd.concat([df_pc[['PC1','PC2','PC3']], pd.DataFrame(X_rdkit_norm, columns=X_rdkit.columns)], axis=1)

    # Считаем корреляцию между исходными данными и PCA
    corMatrix = pd.DataFrame.corr(df_col)

    # Plot the results
    sns.set(rc={'figure.figsize':(14,6)})
    sns.heatmap(corMatrix, annot=True, fmt='.3f')
    plt.figure(figsize=(28,18))
    plt.show()

    #Выбираем дескрипторы из rdkit
    features = X_rdkit.columns.tolist()

    #Считаем метрики
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    #Визуализируем результаты
    fig = px.scatter(df_pc, x='PC1', y='PC2')
    for i, feature in enumerate(features):
        i += 1275
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0]*100,
            y1=loadings[i, 1]*100
        )
        fig.add_annotation(
            x=loadings[i, 0]*120,
            y=loadings[i, 1]*120,
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )
    fig.show()

    #t-SNE
    tsne = TSNE(n_components=2, perplexity=50)
    X_tsne = tsne.fit_transform(X_norm)

    #Визуализируем
    fig = plt.figure(figsize=(14,10))
    ax  = fig.add_subplot(111)
    scatter = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=Y, s=45, edgecolors='black', cmap=cm.jet_r, alpha=0.5)
    colorbar = fig.colorbar(scatter, ax=ax, label = "E")
    plt.xlabel(r'$Z_1$')
    plt.ylabel(r'$Z_2$')
    sns.despine()
    plt.show()

    print('Новая форма X: ', X_tsne.shape)
    print('Дивергенция Кульбака-Лейблера после оптимизации:', tsne.kl_divergence_)
    print('Количество итераций: ', tsne.n_iter_)

    '''Значение 1.0182533264160156 для дивергенции Кульбака-Лейблера вышло небольшим, что означает, 
        что t-SNE  алгоритм смог достаточно хорошо сохранить глобальную структуру данных'''