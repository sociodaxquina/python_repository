#correlation checker
def correlationBuilder(df, target):

    #Person's Correlation with Target
    targetCorr = df.corr(method = 'pearson')[target].abs()[:-1]

    #Pearson's Inner Feature Correlation (in absolute value)
    df = df.drop([target], axis = 1)
    innerCorr = df.corr(method = 'pearson').abs()
    innerCorr = innerCorr.where(np.tril(np.ones(innerCorr.shape), k=1).astype(np.bool))

    return innerCorr, targetCorr
