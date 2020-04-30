#correlation checker
def correlationBuilder(df, target):

    #Person's Correlation with Target
    targetCorr = df.corr(method = 'pearson')[target].abs()[:-1]

    #Pearson's Inner Feature Correlation (in absolute value)
    df = df.drop([target], axis = 1)
    innerCorr = df.corr(method = 'pearson').abs()
    innerCorr = innerCorr.where(np.tril(np.ones(innerCorr.shape), k=1).astype(np.bool))

    return innerCorr, targetCorr

innerCorr, targetCorr = correlationBuilder(df_loaded, 'Latencia')
corr_triu = innerCorr[all_features].stack()
corr_triu.name = 'pearson_Corr'
corr_triu.index.names = ['Col1', 'Col2']
corr_threshold = 0
df_feature_reduction = corr_triu[(corr_triu != 1.0)&(corr_triu > corr_threshold)].reset_index().rename(columns={"Col1": "features", "Col2": "features_comp"})

df_target_correlation = targetCorr.reset_index().rename(columns={"index": "features", "Latencia": "target_corr"})
df_feature_reduction = df_feature_reduction.merge(df_target_correlation).sort_values('target_corr', ascending=False)

print(df_feature_reduction)
