import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from glob import iglob
from tqdm import tqdm


def mse(df, true_cols, pred_cols):
    errors = (df[true_cols].values - df[pred_cols].values) ** 2
    return np.mean(errors), pd.DataFrame(data=errors, columns=true_cols)


def mape(df, true_cols, pred_cols):
    errors = 100 * (df[true_cols].values - df[pred_cols].values) / df[true_cols].values
    return np.mean(np.abs(errors)), pd.DataFrame(data=errors, columns=true_cols)


def cross_validation(path):
    dfs = []
    for file in iglob(path):
        df = pd.read_csv(file)
        df['fold'] = int(file[-6:-4])
        if 'Training' in file:
            df['Iteration'] = list(range(len(df)))
        dfs.append(df)

    return pd.concat(dfs)


def calculate_hp_trial_errors(paradigm, metric=mape, num_files=500, targets=['c1', 'c2'], base_path='/baldig/physicstest/NeutronStarsData/SherpaResults/{paradigm}/Predictions/validation_*.csv'):
    path = base_path.format(paradigm=paradigm)
    pred_cols = ['pred_' + c for c in targets]
    error_dict = {}
    validation_files = list(iglob(path))[:num_files]

    for val_file in tqdm(validation_files):
        trial_id = int(val_file.replace('_01.csv', '')[-5:])
        df = pd.read_csv(val_file, index_col=0)[[*targets, *pred_cols]]
        error_dict[trial_id], _ = metric(df, targets, pred_cols)
        del df

    error_df = pd.DataFrame(error_dict, index=['error']).T
    return error_df


def groupby_poisson_noise(paradigm, trial, base_path='Results/{paradigm}/Predictions/poisson_{trial}_10.csv'):
    path = base_path.format(
        paradigm=paradigm,
        trial='%05d' % trial
    )
    poisson_noise = pd.read_csv(path, index_col=0)

    all_dfs = []
    all_stds = []

    for idx in range(0, poisson_noise.shape[0], 100):
        group = poisson_noise.iloc[idx:idx + 100]
        # SAVE THE MEAN OF 100 PREDICTIONS WITH POISSON NOISE
        all_dfs.append(group.mean().to_frame().T)
        # SAVE THE STD OF 100 PREDICTIONS WITH POISSON NOISE
        all_stds.append(group[['pred_c1', 'pred_c2']].std().values)

    all_dfs = pd.concat(all_dfs)
    _, all_mapes = mape(all_dfs, ['c1', 'c2'], ['pred_c1', 'pred_c2'])

    std_df = pd.DataFrame(data=np.array(all_stds), columns=['c1', 'c2'])
    plot_mape_std(
        mape_df=all_mapes.abs(),
        std_df=std_df,
        suptitle='For each sample make 100 copies with Poisson noise',
        mape_title='Mean of Augmented Predictions'
    )


def groupby_unique_eos(df):
    all_dfs = []
    all_stds = []

    for name, group in df.groupby(['fold', 'c1', 'c2']):
        # SAVE THE MEAN OF UNIQUE EOS PARAMS
        all_dfs.append(group.mean().to_frame().T)
        # SAVE THE STD OF UNIQUE EOS PARAMS
        all_stds.append(group[['pred_c1', 'pred_c2']].std().values)

    all_dfs = pd.concat(all_dfs)
    _, all_mapes = mape(all_dfs, ['c1', 'c2'], ['pred_c1', 'pred_c2'])

    std_df = pd.DataFrame(data=np.array(all_stds), columns=['c1', 'c2'])
    plot_mape_std(
        mape_df=all_mapes.abs(),
        std_df=std_df,
        suptitle='Groupby Unique EOS Parameters',
        mape_title='Mean of Predictions for Unique EOS'
    )


def plot_mape_std(mape_df, std_df, suptitle, mape_title):
    # PLOT MAPE OF MEAN PREDICTIONS WITH POISSON NOISE
    plt.subplot(1, 2, 1)
    ax = sns.histplot(data=mape_df);
    plt.setp(ax.get_legend().get_texts(), fontsize='25')
    plt.xlabel(r'Absolute Percent Error: $100 * |(Y - \hat{Y}) / Y|$', fontsize=25);
    plt.title(mape_title)

    # PLOT STD
    plt.subplot(1, 2, 2)
    ax = sns.histplot(data=std_df);
    plt.setp(ax.get_legend().get_texts(), fontsize='25')
    plt.xlabel('Standard Deviation');

    plt.suptitle(suptitle, fontsize=25)
    plt.show()


def get_crossval_command(paradigm, trial):
    trial = '%05d' % trial
    command = f'tf2 main.py --load_settings_from /baldig/physicstest/NeutronStarsData/SherpaResults/{paradigm}/Settings/{trial}.json'
    print('To run 10 fold cross-validation for this model:\n', command)


def get_poisson_command(paradigm, trial):
    trial = '%05d' % trial
    command = f'tf2 main.py --run_type uncertain --model_dir Results/{paradigm}/Models/{trial}/ ' \
                f'--paradigm {paradigm} --load_settings_from Results/{paradigm}/Settings/{trial}.json'
    print('To run poisson uncertainty augmentation:\n', command)
