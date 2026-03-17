# Changelog

## 1.0.0 (2026-03-17)


### Features

* **config:** add training and ensemble config structs ([88ade2e](https://github.com/feza-ai/metee/commit/88ade2ec146300c26962b084d9f947048bde3d6d))
* **cv:** add CrossValidate with per-fold scoring ([efce870](https://github.com/feza-ai/metee/commit/efce8701510117e364e9ffedaaa0745aed6f67bd))
* **cv:** add era-level KFold and WalkForward splits ([6fee49f](https://github.com/feza-ai/metee/commit/6fee49f977dc75c8f7094c4760b39f3085d98ba3))
* **data:** add Parquet loader with streaming and load options ([a3712d3](https://github.com/feza-ai/metee/commit/a3712d36123acd055c40591bee1bcdf6bc1b9cca))
* **ensemble:** add rank-based blending ([32ebe11](https://github.com/feza-ai/metee/commit/32ebe11ee8499fd80396409233cef989fddbc6b1))
* **ensemble:** add stacking ensemble ([008c6f7](https://github.com/feza-ai/metee/commit/008c6f7ddedc6e24504d5498053971d601a6462c))
* **lightgbm:** register backend via init ([26e43b8](https://github.com/feza-ai/metee/commit/26e43b8e797239ec57e759407167007156429ada))
* **metrics:** add FeatureNeutralCorrelation ([1613c5b](https://github.com/feza-ai/metee/commit/1613c5b87088d1df39a51a66a0bca1c34a6ba60c))
* **metrics:** add PerEraReport and SpearmanPerEra ([d7bbe73](https://github.com/feza-ai/metee/commit/d7bbe73ec555ea84ef893c9848b0a1e17cd54d89))
* **model:** add Validator and Configurable interfaces ([5abad03](https://github.com/feza-ai/metee/commit/5abad03de6455c678a01bd2f429c60db397f4176))
* **registry:** add backend registry ([697c24f](https://github.com/feza-ai/metee/commit/697c24f8769c37288c7430f72edc6489a998a250))
* **trainer:** add callback support ([092caae](https://github.com/feza-ai/metee/commit/092caaeec55bc39528d00bd171c37dd905c89730))
* **trainer:** add training orchestrator ([767b60a](https://github.com/feza-ai/metee/commit/767b60ac0f0de5d12ad40d3137fcf37988424354))
* **transform:** add exposure computation ([86795d3](https://github.com/feza-ai/metee/commit/86795d334db0025104b095587c514def88ca1fee))
* **transform:** add neutralization ([5db3385](https://github.com/feza-ai/metee/commit/5db338556826e1420dc64567851dec3dc2d981af))
* **transform:** add neutralization ([878356f](https://github.com/feza-ai/metee/commit/878356f0082a5de0e3a8541a9df4599ff52c1eb8))
* **transform:** add rank normalization ([e933b63](https://github.com/feza-ai/metee/commit/e933b63dd95b986541a157edeeb0543242b8f1f4))
* **transform:** add transform pipeline ([712aff5](https://github.com/feza-ai/metee/commit/712aff559b905bf0a28cd17b6f3a740312cd31ff))
* **tuning:** add GridSearch and RandomSearch ([7d0db9c](https://github.com/feza-ai/metee/commit/7d0db9c2a052fa9d97f714ef7ce35a8a347a0c20))
* **tuning:** add parameter space with grid and random sampling ([0be26d6](https://github.com/feza-ai/metee/commit/0be26d6a948f81156ec59a9e0380b891ae0b33e6))
* **xgboost:** add CGO bindings for XGBoost C API ([9593ffe](https://github.com/feza-ai/metee/commit/9593ffe13b22480347cabfe69bf77a8642b70e99))
* **xgboost:** add params and stub ([ab1910d](https://github.com/feza-ai/metee/commit/ab1910d3c8f81c0402fd52141dda89662e222588))


### Bug Fixes

* **data:** treat non-numeric parquet eras as 0 instead of erroring ([eec4e36](https://github.com/feza-ai/metee/commit/eec4e36dff3ad4e43775bd6b6929b932ca4208d0))
* **lightgbm:** free previous booster on re-Train, fix CString leaks, add SetParams ([d568029](https://github.com/feza-ai/metee/commit/d5680298ab2861b967b177fcf0321de8aebf612f))
