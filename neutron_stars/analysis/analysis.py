import numpy as np
import pandas as pd
import neutron_stars as ns
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from glob import iglob
from tqdm import tqdm


def get_named_target_for_paradigm(paradigm, num_coefficients=2):
    target_names = []
    opts = ns.get_paradigm_opts(num_coefficients)
    for target in paradigm.split('2')[-1].split('+'):
        target_names.extend(opts[target]['columns'])

    pred_names = ['pred_' + tn for tn in target_names]
    return target_names, pred_names


def cross_validation(path):
    dfs = []
    for file in iglob(path):
        df = pd.read_csv(file)
        df['fold'] = int(file[-6:-4])
        if 'Training' in file:
            df['Iteration'] = list(range(len(df)))
        dfs.append(df)

    return pd.concat(dfs)


def calculate_hp_trial_errors(paradigm, metric_type='mape', num_files=500, base_path='/baldig/physicstest/NeutronStarsData/SherpaResults/{paradigm}/Predictions/validation_*.csv'):
    error_dict = {}
    path = base_path.format(paradigm=paradigm)
    validation_files = list(iglob(path))[:num_files]
    metric = ns.analysis.AVAILABLE_METRICS[metric_type]()
    target_names, pred_names = get_named_target_for_paradigm(paradigm)

    for val_file in tqdm(validation_files):
        trial_id = int(val_file.replace('_01.csv', '')[-5:])
        df = pd.read_csv(val_file, index_col=0)[[*target_names, *pred_names]]
        error_dict[trial_id], _ = metric(df, target_names, pred_names)
        del df

    error_df = pd.DataFrame(error_dict, index=['error']).T
    return error_df


def plot_overall(df, paradigm, metric_type='mape'):
    metric = ns.analysis.AVAILABLE_METRICS[metric_type]()
    target_names, pred_names = get_named_target_for_paradigm(paradigm)
    _, df_errors = metric(df, target_names, pred_names)

    ax = sns.histplot(data=df_errors.abs())
    plt.setp(ax.get_legend().get_texts(), fontsize='25')
    plt.xlabel(metric.label, fontsize=25)
    plt.title('Histogram by Coefficient', fontsize=25)

    plt.savefig(f'Figures/{paradigm}/overall_{metric_type}_hist.png')
    plt.show()


def groupby_poisson_noise(paradigm, trial, metric_type='mape', base_path='Results/{paradigm}/Predictions/poisson_{trial}_10.csv'):
    path = base_path.format(
        paradigm=paradigm,
        trial='%05d' % trial
    )
    poisson_noise = pd.read_csv(path, index_col=0)
    metric = ns.analysis.AVAILABLE_METRICS[metric_type]()
    target_names, pred_names = get_named_target_for_paradigm(paradigm)

    all_dfs = []
    all_stds = []

    for idx in range(0, poisson_noise.shape[0], 100):
        group = poisson_noise.iloc[idx:idx + 100]
        # SAVE THE MEAN OF 100 PREDICTIONS WITH POISSON NOISE
        all_dfs.append(group.mean().to_frame().T)
        # SAVE THE STD OF 100 PREDICTIONS WITH POISSON NOISE
        all_stds.append(group[pred_names].std().values)

    all_dfs = pd.concat(all_dfs)
    _, all_mapes = metric(all_dfs, target_names, pred_names)

    std_df = pd.DataFrame(data=np.array(all_stds), columns=target_names)
    plot_mape_std(
        mape_df=all_mapes.abs(),
        std_df=std_df,
        metric=metric,
        suptitle='For each sample make 100 copies with Poisson noise',
        mape_title='Mean of Augmented Predictions'
    )
    plt.savefig(f'Figures/{paradigm}/gb_poisson_noise_{metric_type}_hist.png')
    plt.show()


def groupby_unique_eos(df, paradigm, metric_type='mape'):

    all_dfs = []
    all_stds = []
    metric = ns.analysis.AVAILABLE_METRICS[metric_type]()
    target_names, pred_names = get_named_target_for_paradigm(paradigm)

    for name, group in df.groupby(['fold', *target_names]):
        # SAVE THE MEAN OF UNIQUE EOS PARAMS
        all_dfs.append(group.mean().to_frame().T)
        # SAVE THE STD OF UNIQUE EOS PARAMS
        all_stds.append(group[pred_names].std().values)

    all_dfs = pd.concat(all_dfs)
    _, all_mapes = metric(all_dfs, target_names, pred_names)

    std_df = pd.DataFrame(data=np.array(all_stds), columns=['c1', 'c2'])
    plot_mape_std(
        mape_df=all_mapes.abs(),
        std_df=std_df,
        metric=metric,
        suptitle='Groupby Unique EOS Parameters',
        mape_title='Mean of Predictions for Unique EOS',
    )
    plt.savefig(f'Figures/{paradigm}/gb_eos_{metric_type}_hist.png')
    plt.show()


def plot_mape_std(mape_df, std_df, metric, suptitle, mape_title):
    # PLOT MAPE OF MEAN PREDICTIONS WITH POISSON NOISE
    plt.subplot(1, 2, 1)
    ax = sns.histplot(data=mape_df);
    plt.setp(ax.get_legend().get_texts(), fontsize='25')
    plt.xlabel(metric.label, fontsize=25);
    plt.title(mape_title)

    # PLOT STD
    plt.subplot(1, 2, 2)
    ax = sns.histplot(data=std_df)
    plt.setp(ax.get_legend().get_texts(), fontsize='25')
    plt.xlabel('Standard Deviation')

    plt.suptitle(suptitle, fontsize=25)


def get_crossval_command(paradigm, trial):
    trial = '%05d' % trial
    command = f'tf2 main.py --load_settings_from /baldig/physicstest/NeutronStarsData/SherpaResults/{paradigm}/Settings/{trial}.json'
    print('To run 10 fold cross-validation for this model:\n', command)


def get_poisson_command(paradigm, trial):
    trial = '%05d' % trial
    command = f'tf2 main.py --run_type uncertain --model_dir Results/{paradigm}/Models/{trial}/ ' \
                f'--paradigm {paradigm} --load_settings_from Results/{paradigm}/Settings/{trial}.json'
    print('To run poisson uncertainty augmentation:\n', command)
